/**
 * Automated ML research loop for the Oxi chess model.
 * Runs iterations of: propose change → compile check → train → evaluate → keep/discard.
 *
 * v3 changes:
 * - Migrated from Python to TypeScript (Bun runtime)
 * - Inline statistical helpers replace numpy/scipy
 * - Chart generation via canvas
 */
import { call_tool } from "./shadesmar_tools";
import { join } from "path";
import { existsSync, mkdirSync, readdirSync, statSync, rmSync, readFileSync, writeFileSync, appendFileSync } from "fs";

const WORKSPACE = "/Users/marcusbuffett/projects/chessbook/oxi";
const RESEARCH_LOG = join(WORKSPACE, "research_log.md");
const AUTORESEARCH_DIR = join(WORKSPACE, "autoresearch");
const RESEARCH_RUNS = join(WORKSPACE, "research_runs");
const STATE_FILE = join(WORKSPACE, "research_state.json");
const TRAINING_TIMEOUT = 3600; // 60 minutes
const MAX_ITERATIONS = 50;
const BATCH_SIZE = 512;

// Composite metric weights
const METRIC_WINDOW = 100;
const WEIGHT_TOP1 = 1.0;
const WEIGHT_WDL = 0.5;
const WEIGHT_AUX = 0.2;
const WEIGHT_CALIBRATION = 0.4;
const CALIBRATION_MIN_LABELED_FRACTION = 0.01;

// Acceptance: block bootstrap p-value + Cohen's d
const BOOTSTRAP_SAMPLES = 5000;
const BLOCK_SIZE = 20;
const MIN_COHEN_D = 0.3;
const ALPHA = 0.05;

// Error handling
const MAX_CONSECUTIVE_ERRORS = 5;
const INITIAL_BACKOFF_SECS = 30;
const MAX_BACKOFF_SECS = 600;

// ---------------------------------------------------------------------------
// Array helpers (replacing numpy)
// ---------------------------------------------------------------------------

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr: number[], ddof = 0): number {
  const m = mean(arr);
  const variance = arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / (arr.length - ddof);
  return Math.sqrt(variance);
}

function percentile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function randInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min)) + min;
}

function shuffle<T>(arr: T[]): T[] {
  const out = [...arr];
  for (let i = out.length - 1; i > 0; i--) {
    const j = randInt(0, i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Metric loading
// ---------------------------------------------------------------------------

function normalizeWdl(value: number): number {
  return value > 1.0 ? value / 100.0 : value;
}

function loadMetricArray(logDir: string, metricName: string): number[] {
  const logFile = join(logDir, "metrics_logs", `${metricName}.log`);
  if (!existsSync(logFile)) return [];
  const text = readFileSync(logFile, "utf-8");
  const values: number[] = [];
  for (const line of text.split("\n")) {
    if (line.includes("\t")) {
      const val = parseFloat(line.trim().split("\t")[1]);
      if (!isNaN(val)) values.push(val);
    }
  }
  return values.length >= METRIC_WINDOW ? values.slice(-METRIC_WINDOW) : values;
}

function loadCompositeArray(logDir: string): number[] {
  function loadMap(name: string): Map<number, number> {
    const logFile = join(logDir, "metrics_logs", `${name}.log`);
    if (!existsSync(logFile)) return new Map();
    const m = new Map<number, number>();
    const text = readFileSync(logFile, "utf-8");
    for (const line of text.split("\n")) {
      if (line.includes("\t")) {
        const parts = line.trim().split("\t");
        const step = parseInt(parts[0]);
        const val = parseFloat(parts[1]);
        if (!isNaN(step) && !isNaN(val)) m.set(step, val);
      }
    }
    return m;
  }

  const top1Map = loadMap("top1_accuracy");
  const wdlMap = loadMap("wdl_accuracy");
  const fromMap = loadMap("aux_from_square_accuracy");
  const toMap = loadMap("aux_to_square_accuracy");
  const calibrationMap = loadMap("cp_loss_calibration_overall");
  const labeledFractionMap = loadMap("cp_loss_labeled_fraction");

  const commonSteps = [...top1Map.keys()]
    .filter((s) => wdlMap.has(s) && fromMap.has(s) && toMap.has(s))
    .sort((a, b) => a - b);

  if (commonSteps.length === 0) return [];

  const values: number[] = [];
  for (const s of commonSteps) {
    const aux = (fromMap.get(s)! + toMap.get(s)!) / 2.0;
    const labeled = labeledFractionMap.get(s) ?? 0.0;
    const calibration =
      labeled >= CALIBRATION_MIN_LABELED_FRACTION ? (calibrationMap.get(s) ?? 0.0) : 0.0;
    const score =
      WEIGHT_TOP1 * top1Map.get(s)! +
      WEIGHT_WDL * normalizeWdl(wdlMap.get(s)!) +
      WEIGHT_AUX * aux +
      WEIGHT_CALIBRATION * calibration;
    values.push(score);
  }

  return values.length >= METRIC_WINDOW ? values.slice(-METRIC_WINDOW) : values;
}

function parseMetricLog(logDir: string, metricName: string): number {
  const arr = loadMetricArray(logDir, metricName);
  return arr.length > 0 ? mean(arr) : 0.0;
}

interface Components {
  top1: number;
  wdl: number;
  aux: number;
  calibration: number;
}

function parseCompositeScore(logDir: string): [number, Components] {
  const top1 = parseMetricLog(logDir, "top1_accuracy");
  const wdl = normalizeWdl(parseMetricLog(logDir, "wdl_accuracy"));
  const auxFrom = parseMetricLog(logDir, "aux_from_square_accuracy");
  const auxTo = parseMetricLog(logDir, "aux_to_square_accuracy");
  const aux = (auxFrom + auxTo) / 2.0;
  let calibration = parseMetricLog(logDir, "cp_loss_calibration_overall");
  const labeledFraction = parseMetricLog(logDir, "cp_loss_labeled_fraction");
  if (labeledFraction < CALIBRATION_MIN_LABELED_FRACTION) calibration = 0.0;
  const score =
    WEIGHT_TOP1 * top1 + WEIGHT_WDL * wdl + WEIGHT_AUX * aux + WEIGHT_CALIBRATION * calibration;
  return [score, { top1, wdl, aux, calibration }];
}

// ---------------------------------------------------------------------------
// Statistical testing
// ---------------------------------------------------------------------------

function blockBootstrapTest(
  a: number[],
  b: number[],
  nBootstrap = BOOTSTRAP_SAMPLES,
  blockSize = BLOCK_SIZE,
): [number, number, number, number] {
  const observedDelta = mean(b) - mean(a);

  function blockResample(x: number[]): number[] {
    const n = x.length;
    if (n <= blockSize) return [...x];
    const nBlocks = Math.max(1, Math.ceil(n / blockSize));
    const blocks: number[] = [];
    for (let i = 0; i < nBlocks; i++) {
      const start = randInt(0, n - blockSize + 1);
      blocks.push(...x.slice(start, start + blockSize));
    }
    return blocks.slice(0, n);
  }

  const pooled = [...a, ...b];
  const pooledMean = mean(pooled);
  const pooledCentered = pooled.map((x) => x - pooledMean);

  const deltas = new Float64Array(nBootstrap);
  for (let i = 0; i < nBootstrap; i++) {
    const perm = shuffle([...pooledCentered.keys()].map((i) => pooledCentered[i]));
    const nullA = blockResample(perm.slice(0, a.length));
    const nullB = blockResample(perm.slice(a.length));
    deltas[i] = mean(nullB) - mean(nullA);
  }

  let pCount = 0;
  const absObserved = Math.abs(observedDelta);
  for (let i = 0; i < nBootstrap; i++) {
    if (Math.abs(deltas[i]) >= absObserved) pCount++;
  }
  const pVal = pCount / nBootstrap;

  const bootDeltas = new Float64Array(nBootstrap);
  for (let i = 0; i < nBootstrap; i++) {
    const bootA = blockResample(a);
    const bootB = blockResample(b);
    bootDeltas[i] = mean(bootB) - mean(bootA);
  }
  const bootArr = Array.from(bootDeltas);
  const ciLow = percentile(bootArr, 2.5);
  const ciHigh = percentile(bootArr, 97.5);

  return [pVal, observedDelta, ciLow, ciHigh];
}

interface StatResult {
  delta?: number;
  p?: number;
  d?: number;
  ci_low?: number;
  ci_high?: number;
  mean_new?: number;
  mean_old?: number;
  n_new?: number;
  n_old?: number;
  error?: string;
}

function checkImprovement(currentDir: string, baselineDir: string): [boolean, StatResult] {
  const a = loadCompositeArray(baselineDir);
  const b = loadCompositeArray(currentDir);

  if (a.length < 10 || b.length < 10) {
    return [false, { error: "insufficient data" }];
  }

  const [pVal, delta, ciLow, ciHigh] = blockBootstrapTest(a, b);
  const pooledStd = Math.sqrt((std(a, 1) ** 2 + std(b, 1) ** 2) / 2);
  const d = pooledStd > 0 ? delta / pooledStd : 0.0;

  const passed = pVal < ALPHA && d >= MIN_COHEN_D && delta > 0;
  return [
    passed,
    {
      delta,
      p: pVal,
      d,
      ci_low: ciLow,
      ci_high: ciHigh,
      mean_new: mean(b),
      mean_old: mean(a),
      n_new: b.length,
      n_old: a.length,
    },
  ];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function bestRunDirByScore(): string | null {
  if (!existsSync(RESEARCH_RUNS)) return null;
  let bestDir: string | null = null;
  let bestScore = -1.0;
  for (const entry of readdirSync(RESEARCH_RUNS)) {
    const p = join(RESEARCH_RUNS, entry);
    if (statSync(p).isDirectory() && existsSync(join(p, "metrics_logs"))) {
      const arr = loadCompositeArray(p);
      if (arr.length > 0) {
        const s = mean(arr);
        if (s > bestScore) {
          bestScore = s;
          bestDir = p;
        }
      }
    }
  }
  return bestDir;
}

function compileCheck(): [boolean, string] {
  const result = Bun.spawnSync(["cargo", "check", "--features", "train,backend-tch"], {
    cwd: WORKSPACE,
    timeout: 300_000,
  });
  return [result.exitCode === 0, result.stderr.toString()];
}

function formatComponents(c: Components): string {
  return `(top1=${c.top1.toFixed(4)}, wdl=${c.wdl.toFixed(4)}, aux=${c.aux.toFixed(4)}, cal=${c.calibration.toFixed(4)})`;
}

function formatDelta(
  score: number,
  components: Components,
  prevScore: number,
  prevComponents: Components,
): string {
  const ds = score - prevScore;
  const dt = components.top1 - prevComponents.top1;
  const dw = components.wdl - prevComponents.wdl;
  const da = components.aux - prevComponents.aux;
  const dc = components.calibration - prevComponents.calibration;
  const fmt = (n: number) => (n >= 0 ? `+${n.toFixed(4)}` : n.toFixed(4));
  return `${fmt(ds)} (top1=${fmt(dt)}, wdl=${fmt(dw)}, aux=${fmt(da)}, cal=${fmt(dc)})`;
}

function buildBinary(): [boolean, string] {
  const result = Bun.spawnSync(
    ["cargo", "build", "--release", "--features", "backend-tch train"],
    { cwd: WORKSPACE, timeout: 600_000 },
  );
  return [result.exitCode === 0, result.stderr.toString()];
}

async function runTraining(runName: string, seed: number): Promise<[number, Components]> {
  const logDir = join(RESEARCH_RUNS, runName);
  const metricsDir = join(logDir, "metrics_logs");
  if (existsSync(metricsDir)) rmSync(metricsDir, { recursive: true });
  mkdirSync(logDir, { recursive: true });

  const binary = join(WORKSPACE, "target", "release", "oxi");
  const cmd = [
    binary,
    "train",
    "--pretrain-samples=0",
    "--data-path=../data",
    `--physical-batch-size=${BATCH_SIZE}`,
    `--seed=${seed}`,
    `--log-dir=${logDir}`,
    "--disable-tui",
    "--warmup-multiplier=1.0",
    "--log-gradient-breakdown",
    "--full-metrics-interval=200",
    "--gradnorm-interval=200",
    "--checkpoint-interval=400",
  ];

  console.log(`  Training seed=${seed} for ${TRAINING_TIMEOUT}s...`);
  const stderrLog = join(logDir, "stderr.log");

  const proc = Bun.spawn(cmd, {
    cwd: WORKSPACE,
    stdout: "ignore",
    stderr: "pipe",
  });

  // Stream stderr to file in background
  const stderrPromise = (async () => {
    const writer = Bun.file(stderrLog).writer();
    const reader = proc.stderr.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      writer.write(value);
    }
    writer.flush();
    writer.end();
  })();

  // Kill on timeout, then just await normal exit
  const timer = setTimeout(() => proc.kill(), TRAINING_TIMEOUT * 1000);
  const exitCode = await proc.exited;
  clearTimeout(timer);

  await stderrPromise;

  if (exitCode !== 0 && exitCode !== null) {
    let stderrTail = "";
    try {
      stderrTail = (await Bun.file(stderrLog).text()).slice(-1000);
    } catch {}
    console.log(`  Training exited with code ${exitCode}`);
    if (stderrTail) console.log(`  stderr (last 1000 chars):\n${stderrTail}`);
  }

  const [score, components] = parseCompositeScore(logDir);
  console.log(`  score = ${score.toFixed(6)} ${formatComponents(components)}`);
  return [score, components];
}

function gitRevert() {
  const diff = Bun.spawnSync(["git", "diff", "--quiet", "src/", "Cargo.toml"], {
    cwd: WORKSPACE,
  });
  if (diff.exitCode !== 0) {
    Bun.spawnSync(
      [
        "git", "stash", "push", "-m", "research_loop: auto-stash discarded changes",
        "--", "src/", "Cargo.toml",
      ],
      { cwd: WORKSPACE },
    );
  } else {
    const cached = Bun.spawnSync(
      ["git", "diff", "--cached", "--quiet", "src/", "Cargo.toml"],
      { cwd: WORKSPACE },
    );
    if (cached.exitCode !== 0) {
      Bun.spawnSync(
        [
          "git", "stash", "push", "-m", "research_loop: auto-stash discarded changes",
          "--", "src/", "Cargo.toml",
        ],
        { cwd: WORKSPACE },
      );
    }
  }
}

function gitCommit(message: string) {
  Bun.spawnSync(["git", "add", "src/", "Cargo.toml"], { cwd: WORKSPACE });
  Bun.spawnSync(["git", "commit", "-m", message], { cwd: WORKSPACE });
}

interface ResultEntry {
  iter: number;
  title: string;
  ao: number;
  status: string;
}

interface State {
  best_score: number;
  baseline_score: number;
  results: ResultEntry[];
  best_run_dir?: string;
  best_components?: Components;
}

function loadState(): State {
  let state: State;
  if (existsSync(STATE_FILE)) {
    state = JSON.parse(readFileSync(STATE_FILE, "utf-8"));
  } else {
    state = { best_score: 0.0, baseline_score: 0.0, results: [] };
  }
  state.best_score ??= 0.0;
  state.baseline_score ??= 0.0;
  state.results ??= [];
  if (!state.best_run_dir) {
    const d = bestRunDirByScore();
    state.best_run_dir = d ?? join(RESEARCH_RUNS, "baseline");
  }
  return state;
}

function saveState(state: State) {
  writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

function parseExistingLog(): ResultEntry[] {
  if (!existsSync(RESEARCH_LOG)) return [];
  const results: ResultEntry[] = [];
  const text = readFileSync(RESEARCH_LOG, "utf-8");
  for (const line of text.split("\n")) {
    const m = line.match(/\|\s*(\d+)\s*\|(.+?)\|\s*([\d.]+)\s*\|\s*(\S+)\s*\|/);
    if (m) {
      results.push({
        iter: parseInt(m[1]),
        title: m[2].trim(),
        ao: parseFloat(m[3]),
        status: m[4].trim(),
      });
    }
  }
  return results;
}

function generateChart(results: ResultEntry[], baselineAo: number) {
  if (!results.length) return;
  const filtered = results.filter((r) => r.status !== "error");
  if (!filtered.length) return;

  // Skip chart generation — canvas dependencies not available in script context.
  // The research_log.md table is the canonical record.
  try {
    call_tool("artifact_upload", {
      path: RESEARCH_LOG,
      kind: "file",
      description: "Research log progress",
    });
  } catch {
    // artifact_upload not available
  }
}

function extractStructuredOutput(text: string): [string, string, string] {
  const jsonBlocks = [...text.matchAll(/```json\s*(\{.*?\})\s*```/gs)].map((m) => m[1]);
  const fallbackBlocks =
    jsonBlocks.length === 0
      ? [...text.matchAll(/(\{"title".*?\})/gs)].map((m) => m[1])
      : jsonBlocks;

  if (fallbackBlocks.length > 0) {
    try {
      const data = JSON.parse(fallbackBlocks[fallbackBlocks.length - 1]);
      const title = (data.title ?? "")
        .trim()
        .replace(/\|/g, "-")
        .replace(/\n/g, " ")
        .slice(0, 80);
      const description = (data.description ?? "").trim();
      const changes = (data.changes ?? "").trim();
      if (title && validateTitle(title)) return [title, description, changes];
    } catch {}
  }

  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (trimmed && !trimmed.startsWith("#") && !trimmed.startsWith("```") && validateTitle(trimmed)) {
      return [trimmed.slice(0, 80).replace(/\|/g, "-"), "", ""];
    }
  }
  return ["Unknown change", "", ""];
}

function validateTitle(title: string): boolean {
  const noiseStarts = [
    "i've analyzed", "let me", "i'll ", "i will", "ok,", "alright",
    "i need to", "i should", "now i", "first,", "looking at",
    "i have", "i can", "i want", "let's",
  ];
  const lower = title.toLowerCase();
  return !noiseStarts.some((p) => lower.startsWith(p));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}


// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

async function main() {
  mkdirSync(AUTORESEARCH_DIR, { recursive: true });
  mkdirSync(RESEARCH_RUNS, { recursive: true });

  const state = loadState();
  let existingResults = parseExistingLog();

  if (state.results?.length) {
    state.results = state.results.filter((r) => r.status !== "error");
  }

  if (!existsSync(RESEARCH_LOG) || statSync(RESEARCH_LOG).size === 0) {
    await Bun.write(
      RESEARCH_LOG,
      "# OXI Research Log\n\n| Iter | Description | Score | Δ vs prev best | Kept |\n|------|-------------|-------|----------------|------|\n",
    );
  }

  // Run baseline if needed
  if (state.baseline_score === 0.0) {
    console.log("=== Running baseline training ===");
    gitRevert();

    console.log("  Building release binary...");
    const [ok, stderr] = buildBinary();
    if (!ok) {
      console.log(`  Build failed: ${stderr.slice(-500)}`);
      return;
    }

    const [baseline, baselineComponents] = await runTraining("baseline", 42);
    state.baseline_score = baseline;
    state.best_score = baseline;
    state.best_run_dir = join(RESEARCH_RUNS, "baseline");
    state.best_components = baselineComponents;
    if (!existingResults.some((r) => r.iter === 0)) {
      appendFileSync(RESEARCH_LOG,
          `| 0 | Baseline (no changes) | ${baseline.toFixed(6)} ${formatComponents(baselineComponents)} | — | baseline |\n`,
      );
      existingResults.push({ iter: 0, title: "Baseline", ao: baseline, status: "baseline" });
    }
    saveState(state);
    console.log(`Baseline score: ${baseline.toFixed(6)} ${formatComponents(baselineComponents)}`);
  } else {
    console.log(
      `Resuming with ${existingResults.length} prior results, best score: ${state.best_score.toFixed(6)}`,
    );
  }

  let best = state.best_score;
  const baseline = state.baseline_score;
  const allResults: ResultEntry[] = [
    ...(state.results?.length ? state.results : existingResults).filter(
      (r) => r.status !== "error",
    ),
  ];

  let consecutiveErrors = 0;

  for (let iterationCount = 0; iterationCount < MAX_ITERATIONS; iterationCount++) {
    const i = allResults.length;
    console.log(`\n${"=".repeat(60)}`);
    console.log(
      `=== Iteration ${i} | ${iterationCount + 1}/${MAX_ITERATIONS} | best: ${best.toFixed(6)} | baseline: ${baseline.toFixed(6)} ===`,
    );
    console.log("=".repeat(60));

    try {
      gitRevert();

      const bestComps: Components = state.best_components ?? {
        top1: 0.0,
        wdl: 0.0,
        aux: 0.0,
        calibration: 0.0,
      };

      const iterDir = join(AUTORESEARCH_DIR, String(i));
      mkdirSync(iterDir, { recursive: true });

      // --- Propose experiment ---
      const ideaPrompt = `You are an ML researcher improving a chess move-prediction transformer.

**Start by reading \`research/architecture.md\`** — it contains a comprehensive reference of the model architecture, input representation, all output heads, loss functions, GradNorm, optimizer config, LR scheduling, data pipeline, metrics logging, and evaluation scoring. This is your primary reference for understanding the codebase.

## Evaluation Metric

Your changes are evaluated on a **composite score** computed as:
  score = ${WEIGHT_TOP1}*top1_accuracy + ${WEIGHT_WDL.toFixed(4)}*wdl_accuracy + ${WEIGHT_AUX}*aux_accuracy + ${WEIGHT_CALIBRATION.toFixed(4)}*calibration
where each component is computed from the last ${METRIC_WINDOW} logged values during a ${TRAINING_TIMEOUT / 60}-minute training run.

Components:
- **top1_accuracy** (weight ${WEIGHT_TOP1}): Policy head move prediction accuracy (0-1).
- **wdl_accuracy** (weight ${WEIGHT_WDL.toFixed(4)}): Win/draw/loss prediction accuracy (0-1, normalized if logged as percentages).
- **aux_accuracy** (weight ${WEIGHT_AUX}): Average of from-square and to-square auxiliary head accuracy (0-1).
- **calibration** (weight ${WEIGHT_CALIBRATION.toFixed(4)}): \`cp_loss_calibration_overall\` — computed as \`exp(-abs_cpl_error / 15)\`, so it's very sensitive to centipawn-loss accuracy. This rewards matching the human centipawn-loss profile, not just predicting the exact move.

Important calibration details:
- The main human-facing diagnostics are the signed Elo-band CPL calibration metrics, but the loop score uses the bounded overall calibration metric.
- A change that improves top-1 while harming human-strength calibration can lose overall.

Current best score: ${best.toFixed(6)} ${formatComponents(bestComps)} | Baseline: ${baseline.toFixed(6)} | Iteration ${i}

## Important Context

- **Loss weighting is handled by GradNorm** — the model uses adaptive GradNorm reweighting across policy, value, aux, and calibration. You do NOT need to manually tune loss weights unless you are deliberately changing the training objective.
- **This is a time-constrained training run** (${TRAINING_TIMEOUT / 60} minutes). Throughput matters.
- **The goal is to prep for a larger production run.** We want the best architecture/training recipe for both move prediction and human-strength calibration. Improvements to training dynamics, loss functions, architecture, or calibration behavior are all valuable.
- **Architecture is not pinned.** The model's embed_dim, num_layers, and head_dim come from defaults in \`src/config.rs\`. If you want to change the architecture, modify those defaults. The training command does NOT pass these as CLI args, so your config.rs changes will take effect.

## Risk Tolerance

A change that scores 5% worse is free — it gets reverted automatically. A change that scores 2% better is kept forever. **The expected value of bold experiments is high.** Don't optimize for keep rate, optimize for finding large improvements.

If your proposed change would not plausibly move the score by at least 1%, it's too small. One logical change can still be large — replacing the attention mechanism is one change, adding a new auxiliary head is one change.

## What To Do

Read the experiment log at research_log.md to see what has been tried. Do not repeat failed ideas.

Explore the codebase, then propose **one logical change**. Don't bundle unrelated modifications, but the change itself can be ambitious — a new architecture component, a different training objective, a structural refactor of how something works.

Verify your change compiles: \`cargo check --features "train,backend-tch"\`

Feel free to use the ./research directory to make notes and see notes from previous agents.

## Acceptance Criteria

Training runs for ${TRAINING_TIMEOUT / 60} minutes then is killed. Changes are kept if:
- Block bootstrap p < ${ALPHA} (autocorrelation-corrected)
- Cohen's d >= ${MIN_COHEN_D}

## Rules

DO NOT TOUCH:
- CLI argument handling or data loading
- The LR scheduler type (reduce-on-plateau only, no cosine/cyclic). Adjusting LR values, warmup, or plateau parameters is fine.
- The evaluation metric or how accuracy is measured
- Loss weights by default — GradNorm handles this unless the experiment is specifically about the objective itself
- research_log.md and research_runs/ — read-only (except research_runs/run_${i}/conclusion.md which you must create)

## Required Output

Write a file at research_runs/run_${i}/conclusion.md describing:
1. **What you did** — short summary
2. **Why you chose this** — detailed reasoning, referencing prior experiments
3. **What you expect** — predicted outcome and confidence level

End your response with:
\`\`\`json
{"title": "<=10 word title", "description": "1-2 sentence description", "changes": "files changed and what was done"}
\`\`\``;

      console.log("  Requesting experiment idea from subagent...");
      const ideaResult = await call_tool("subagent_run", { task: ideaPrompt, role: "full" });
      const ideaOutput = ideaResult.output ?? "";

      await Bun.write(join(iterDir, "idea.md"), ideaOutput);

      const [title, description, changes] = extractStructuredOutput(ideaOutput);
      console.log(`  Title: ${title}`);
      if (description) console.log(`  Description: ${description}`);

      // --- Phase 3: Compile check ---
      console.log("  Checking compilation...");
      const [compiles, compileStderr] = compileCheck();
      if (!compiles) {
        console.log("  ❌ Compile failed!");
        console.log(`  ${compileStderr.slice(-500)}`);
        gitRevert();
        const result: ResultEntry = {
          iter: i,
          title: `[COMPILE FAIL] ${title}`,
          ao: 0.0,
          status: "fail",
        };
        allResults.push(result);
        appendFileSync(RESEARCH_LOG,
            `| ${i} | [COMPILE FAIL] ${title} | 0.000000 | — | ❌ |\n`,
        );
        state.results = allResults;
        saveState(state);
        consecutiveErrors = 0;
        continue;
      }

      console.log("  ✅ Compilation succeeded");

      // --- Phase 4: Build release binary ---
      console.log("  Building release binary...");
      const [buildOk, buildStderr] = buildBinary();
      if (!buildOk) {
        console.log("  ❌ Build failed!");
        console.log(`  ${buildStderr.slice(-500)}`);
        gitRevert();
        const result: ResultEntry = {
          iter: i,
          title: `[BUILD FAIL] ${title}`,
          ao: 0.0,
          status: "fail",
        };
        allResults.push(result);
        appendFileSync(RESEARCH_LOG,
            `| ${i} | [BUILD FAIL] ${title} | 0.000000 | — | ❌ |\n`,
        );
        state.results = allResults;
        saveState(state);
        consecutiveErrors = 0;
        continue;
      }

      // --- Phase 5: Training ---
      const seed = 42;
      const [score, components] = await runTraining(`run_${i}`, seed);

      // --- Phase 6: Statistical comparison ---
      const bestDir = state.best_run_dir!;
      const [improved, stat] = checkImprovement(join(RESEARCH_RUNS, `run_${i}`), bestDir);
      let statStr: string;
      if (!stat.error) {
        statStr = `Δ=${stat.delta!.toFixed(4)}  p=${stat.p!.toFixed(4)}  d=${stat.d!.toFixed(3)}  CI=[${stat.ci_low!.toFixed(4)}, ${stat.ci_high!.toFixed(4)}]`;
      } else {
        statStr = stat.error;
      }

      const prevComponents: Components = state.best_components ?? {
        top1: 0.0,
        wdl: 0.0,
        aux: 0.0,
        calibration: 0.0,
      };
      const deltaStr = formatDelta(score, components, best, prevComponents);

      let status: string;
      let statusEmoji: string;
      if (improved) {
        console.log(`  ✅ IMPROVEMENT: ${score.toFixed(6)}  (${statStr})`);
        best = score;
        state.best_run_dir = join(RESEARCH_RUNS, `run_${i}`);
        state.best_components = components;
        gitCommit(`research iter ${i}: ${title} (score=${score.toFixed(6)})`);
        status = "kept";
        statusEmoji = "✅";
      } else {
        console.log(`  ❌ No improvement: ${score.toFixed(6)}  (${statStr})`);
        gitRevert();
        status = "discarded";
        statusEmoji = "❌";
      }

      const resultEntry: ResultEntry = { iter: i, title, ao: score, status };
      allResults.push(resultEntry);
      appendFileSync(RESEARCH_LOG,
          `| ${i} | ${title} | ${score.toFixed(6)} ${formatComponents(components)} | ${deltaStr} | ${statusEmoji} |\n`,
      );

      state.best_score = best;
      state.results = allResults;
      saveState(state);
      generateChart(allResults, baseline);
      consecutiveErrors = 0;
    } catch (e: any) {
      gitRevert();
      consecutiveErrors++;
      const msg = String(e?.message ?? e);

      const backoff = Math.min(MAX_BACKOFF_SECS, INITIAL_BACKOFF_SECS * 2 ** (consecutiveErrors - 1));
      console.log(
        `  ⚠️ Error (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}): ${msg.slice(0, 120)}`,
      );
      console.error(e);

      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        console.log(`  🛑 ${MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping loop`);
        const result: ResultEntry = {
          iter: i,
          title: `[ERROR] ${msg.slice(0, 50)}`,
          ao: 0.0,
          status: "error",
        };
        allResults.push(result);
        appendFileSync(RESEARCH_LOG,
            `| ${i} | [ERROR x${consecutiveErrors}] ${msg.slice(0, 50)} | 0.000000 | — | ⚠️ |\n`,
        );
        state.results = allResults;
        saveState(state);
        break;
      }

      console.log(`  Backing off ${backoff}s before retry...`);
      await sleep(backoff * 1000);
    }
  }

  // Clean up any leftover changes from the last iteration
  gitRevert();

  console.log(`\n${"=".repeat(60)}`);
  console.log("=== RESEARCH COMPLETE ===");
  console.log(`  Baseline score: ${baseline.toFixed(6)}`);
  console.log(`  Best score:     ${best.toFixed(6)}`);
  console.log(`  Improvement:   ${(best - baseline >= 0 ? "+" : "")}${(best - baseline).toFixed(6)}`);
  const kept = allResults.filter((r) => r.status === "kept");
  const nonBaseline = allResults.filter((r) => r.status !== "baseline");
  console.log(`  Kept changes:  ${kept.length} / ${nonBaseline.length}`);
  console.log(`  Log: ${RESEARCH_LOG}`);

  generateChart(allResults, baseline);

  try {
    await call_tool("artifact_upload", {
      path: RESEARCH_LOG,
      kind: "file",
      description: "Final research log with all iterations",
    });
  } catch {
    console.log("  (artifact_upload not available)");
  }
}

await main();
