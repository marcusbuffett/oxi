"""
Automated ML research loop for the Oxi chess model.
Runs iterations of: propose change → compile check → train → evaluate → keep/discard.

v2 changes:
- Architecture search: agents modify config.rs defaults directly (no CLI pinning)
- 60-minute training timeout (from 30)
- Pre-builds binary and runs it directly (training time isn't eaten by compile)
- Better error handling: all errors with exponential backoff, max consecutive limit
- Better statistical test: block bootstrap to handle autocorrelation
- Better title extraction with validation
"""
from shadesmar_tools import call_tool
import subprocess
import json
import re
import os
import time
import traceback
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKSPACE = Path("/Users/marcusbuffett/projects/chessbook/oxi")
RESEARCH_LOG = WORKSPACE / "research_log.md"
AUTORESEARCH_DIR = WORKSPACE / "autoresearch"
RESEARCH_RUNS = WORKSPACE / "research_runs"
STATE_FILE = WORKSPACE / "research_state.json"
TRAINING_TIMEOUT = 3600  # 60 minutes
MAX_ITERATIONS = 50
BATCH_SIZE = 512

# Composite metric weights
METRIC_WINDOW = 100
WEIGHT_TOP1 = 1.0
WEIGHT_WDL = 0.5
WEIGHT_AUX = 0.2
WEIGHT_CALIBRATION = 0.4
CALIBRATION_MIN_LABELED_FRACTION = 0.01

# Acceptance: block bootstrap p-value + Cohen's d
BOOTSTRAP_SAMPLES = 5000
BLOCK_SIZE = 20  # block bootstrap block size to handle autocorrelation
MIN_COHEN_D = 0.3
ALPHA = 0.05

# Error handling
MAX_CONSECUTIVE_ERRORS = 5
INITIAL_BACKOFF_SECS = 30
MAX_BACKOFF_SECS = 600


def normalize_wdl(value):
    return value / 100.0 if value > 1.0 else value


def load_metric_array(log_dir, metric_name):
    log_file = Path(log_dir) / "metrics_logs" / f"{metric_name}.log"
    if not log_file.exists():
        return np.array([])
    values = []
    with open(log_file, 'r') as f:
        for line in f:
            if '\t' in line:
                try:
                    values.append(float(line.strip().split('\t')[1]))
                except (ValueError, IndexError):
                    pass
    arr = np.array(values)
    return arr[-METRIC_WINDOW:] if len(arr) >= METRIC_WINDOW else arr


def load_composite_array(log_dir):
    def load_map(name):
        log_file = Path(log_dir) / "metrics_logs" / f"{name}.log"
        if not log_file.exists():
            return {}
        m = {}
        with open(log_file, 'r') as f:
            for line in f:
                if '\t' in line:
                    try:
                        step, val = line.strip().split('\t')
                        m[int(step)] = float(val)
                    except (ValueError, IndexError):
                        pass
        return m

    top1_map = load_map("top1_accuracy")
    wdl_map  = load_map("wdl_accuracy")
    from_map = load_map("aux_from_square_accuracy")
    to_map   = load_map("aux_to_square_accuracy")
    calibration_map = load_map("cp_loss_calibration_overall")
    labeled_fraction_map = load_map("cp_loss_labeled_fraction")

    common = sorted(set(top1_map) & set(wdl_map) & set(from_map) & set(to_map))
    if not common:
        return np.array([])

    values = []
    for s in common:
        aux = (from_map[s] + to_map[s]) / 2.0
        labeled = labeled_fraction_map.get(s, 0.0)
        calibration = calibration_map.get(s, 0.0) if labeled >= CALIBRATION_MIN_LABELED_FRACTION else 0.0
        score = (
            WEIGHT_TOP1 * top1_map[s]
            + WEIGHT_WDL * normalize_wdl(wdl_map[s])
            + WEIGHT_AUX * aux
            + WEIGHT_CALIBRATION * calibration
        )
        values.append(score)

    arr = np.array(values)
    return arr[-METRIC_WINDOW:] if len(arr) >= METRIC_WINDOW else arr


def parse_metric_log(log_dir, metric_name):
    arr = load_metric_array(log_dir, metric_name)
    return float(arr.mean()) if len(arr) > 0 else 0.0


def parse_composite_score(log_dir):
    top1 = parse_metric_log(log_dir, "top1_accuracy")
    wdl = normalize_wdl(parse_metric_log(log_dir, "wdl_accuracy"))
    aux_from = parse_metric_log(log_dir, "aux_from_square_accuracy")
    aux_to = parse_metric_log(log_dir, "aux_to_square_accuracy")
    aux = (aux_from + aux_to) / 2.0
    calibration = parse_metric_log(log_dir, "cp_loss_calibration_overall")
    labeled_fraction = parse_metric_log(log_dir, "cp_loss_labeled_fraction")
    if labeled_fraction < CALIBRATION_MIN_LABELED_FRACTION:
        calibration = 0.0
    score = WEIGHT_TOP1 * top1 + WEIGHT_WDL * wdl + WEIGHT_AUX * aux + WEIGHT_CALIBRATION * calibration
    components = {
        "top1": top1,
        "wdl": wdl,
        "aux": aux,
        "calibration": calibration,
    }
    return score, components


def block_bootstrap_test(a, b, n_bootstrap=BOOTSTRAP_SAMPLES, block_size=BLOCK_SIZE):
    """
    Block bootstrap test for difference in means.
    Handles autocorrelation by resampling in contiguous blocks.
    Returns (p_value, observed_delta, ci_low, ci_high).
    """
    observed_delta = b.mean() - a.mean()

    def block_resample(x):
        n = len(x)
        if n <= block_size:
            return x.copy()
        n_blocks = max(1, (n + block_size - 1) // block_size)
        starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        blocks = [x[s:s+block_size] for s in starts]
        return np.concatenate(blocks)[:n]

    pooled = np.concatenate([a, b])
    pooled_centered = pooled - pooled.mean()

    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        perm = np.random.permutation(len(pooled_centered))
        null_a = block_resample(pooled_centered[perm[:len(a)]])
        null_b = block_resample(pooled_centered[perm[len(a):]])
        deltas[i] = null_b.mean() - null_a.mean()

    p_val = np.mean(np.abs(deltas) >= np.abs(observed_delta))

    boot_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot_a = block_resample(a)
        boot_b = block_resample(b)
        boot_deltas[i] = boot_b.mean() - boot_a.mean()
    ci_low = np.percentile(boot_deltas, 2.5)
    ci_high = np.percentile(boot_deltas, 97.5)

    return p_val, observed_delta, ci_low, ci_high


def check_improvement(current_dir, baseline_dir):
    """
    Test whether current run is a significant improvement over baseline.
    Uses block bootstrap on composite score arrays + Cohen's d.
    Returns (is_improvement: bool, stats: dict).
    """
    a = load_composite_array(baseline_dir)
    b = load_composite_array(current_dir)

    if len(a) < 10 or len(b) < 10:
        return False, {"error": "insufficient data"}

    p_val, delta, ci_low, ci_high = block_bootstrap_test(a, b)
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    d = delta / pooled_std if pooled_std > 0 else 0.0

    passed = (p_val < ALPHA) and (d >= MIN_COHEN_D) and (delta > 0)
    return passed, {
        "delta": delta, "p": p_val, "d": d,
        "ci_low": ci_low, "ci_high": ci_high,
        "mean_new": float(b.mean()), "mean_old": float(a.mean()),
        "n_new": len(b), "n_old": len(a),
    }


def best_run_dir_by_score():
    best_dir, best_score = None, -1.0
    for p in RESEARCH_RUNS.iterdir():
        if p.is_dir() and (p / "metrics_logs").is_dir():
            arr = load_composite_array(p)
            if len(arr) > 0:
                s = float(arr.mean())
                if s > best_score:
                    best_score, best_dir = s, p
    return best_dir


def compile_check():
    result = subprocess.run(
        ["cargo", "check", "--features", "train,backend-tch"],
        cwd=WORKSPACE, capture_output=True, text=True, timeout=300
    )
    return result.returncode == 0, result.stderr


def format_components(components):
    return (
        f"(top1={components['top1']:.4f}, wdl={components['wdl']:.4f}, "
        f"aux={components['aux']:.4f}, cal={components['calibration']:.4f})"
    )


def format_delta(score, components, prev_score, prev_components):
    ds = score - prev_score
    dt = components['top1'] - prev_components['top1']
    dw = components['wdl'] - prev_components['wdl']
    da = components['aux'] - prev_components['aux']
    dc = components['calibration'] - prev_components['calibration']
    return f"{ds:+.4f} (top1={dt:+.4f}, wdl={dw:+.4f}, aux={da:+.4f}, cal={dc:+.4f})"


def build_binary():
    """Pre-build release binary. Returns (success, stderr)."""
    result = subprocess.run(
        ["cargo", "build", "--release", "--features", "backend-tch train"],
        cwd=WORKSPACE, capture_output=True, text=True, timeout=600
    )
    return result.returncode == 0, result.stderr


def run_training(run_name, seed):
    """Run training for TRAINING_TIMEOUT seconds using pre-built binary.
    Architecture comes from config.rs defaults (agent modifies those directly).
    Returns (composite_score, components_dict)."""
    log_dir = RESEARCH_RUNS / run_name
    metrics_dir = log_dir / "metrics_logs"
    if metrics_dir.exists():
        import shutil
        shutil.rmtree(metrics_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    binary = WORKSPACE / "target" / "release" / "oxi"
    cmd = [
        str(binary),
        "train",
        "--pretrain-samples=0",
        "--data-path=../data",
        f"--physical-batch-size={BATCH_SIZE}",
        f"--seed={seed}",
        f"--log-dir={log_dir}",
        "--disable-tui",
        "--warmup-multiplier=1.0",
        "--log-gradient-breakdown",
        "--full-metrics-interval=200",
        "--gradnorm-interval=200",
        "--checkpoint-interval=400",
    ]

    print(f"  Training seed={seed} for {TRAINING_TIMEOUT}s...")
    stderr_log = log_dir / "stderr.log"
    with open(stderr_log, "w") as stderr_file:
        try:
            proc = subprocess.Popen(cmd, cwd=WORKSPACE, stdout=subprocess.DEVNULL, stderr=stderr_file)
            proc.wait(timeout=TRAINING_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    if proc.returncode is not None and proc.returncode != 0:
        stderr_tail = ""
        try:
            stderr_tail = stderr_log.read_text()[-1000:]
        except Exception:
            pass
        print(f"  Training exited with code {proc.returncode}")
        if stderr_tail:
            print(f"  stderr (last 1000 chars):\n{stderr_tail}")

    score, components = parse_composite_score(log_dir)
    print(f"  score = {score:.6f} {format_components(components)}")
    return score, components


def git_revert():
    result = subprocess.run(
        ["git", "diff", "--quiet", "src/", "Cargo.toml"],
        cwd=WORKSPACE, capture_output=True
    )
    if result.returncode != 0:
        subprocess.run(
            ["git", "stash", "push", "-m", "research_loop: auto-stash discarded changes",
             "--", "src/", "Cargo.toml"],
            cwd=WORKSPACE, capture_output=True
        )
    else:
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "src/", "Cargo.toml"],
            cwd=WORKSPACE, capture_output=True
        )
        if result.returncode != 0:
            subprocess.run(
                ["git", "stash", "push", "-m", "research_loop: auto-stash discarded changes",
                 "--", "src/", "Cargo.toml"],
                cwd=WORKSPACE, capture_output=True
            )


def git_commit(message):
    subprocess.run(["git", "add", "src/", "Cargo.toml"], cwd=WORKSPACE, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=WORKSPACE, capture_output=True)


def load_state():
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
    else:
        state = {"best_score": 0.0, "baseline_score": 0.0, "results": []}
    state.setdefault("best_score", 0.0)
    state.setdefault("baseline_score", 0.0)
    state.setdefault("results", [])
    if not state.get("best_run_dir"):
        d = best_run_dir_by_score()
        state["best_run_dir"] = str(d) if d else str(RESEARCH_RUNS / "baseline")
    return state


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def parse_existing_log():
    if not RESEARCH_LOG.exists():
        return []
    results = []
    with open(RESEARCH_LOG) as f:
        for line in f:
            m = re.match(r'\|\s*(\d+)\s*\|(.+?)\|\s*([\d.]+)\s*\|\s*(\S+)\s*\|', line)
            if m:
                results.append({
                    'iter': int(m.group(1)),
                    'title': m.group(2).strip(),
                    'ao': float(m.group(3)),
                    'status': m.group(4).strip(),
                })
    return results


def generate_chart(results, baseline_ao):
    if not results:
        return
    results = [r for r in results if r['status'] not in ('error',)]
    if not results:
        return
    chart_path = RESEARCH_RUNS / "progress.png"
    fig, ax = plt.subplots(figsize=(12, 6))

    iters = [r['iter'] for r in results]
    aos = [r['ao'] for r in results]
    colors = []
    for r in results:
        if r['status'] in ('kept', 'baseline'):
            colors.append('green')
        elif r['status'] == 'fail':
            colors.append('gray')
        else:
            colors.append('red')

    ax.scatter(iters, aos, c=colors, s=80, zorder=3)
    if baseline_ao > 0:
        ax.axhline(y=baseline_ao, color='blue', linestyle='--', alpha=0.7,
                   label=f'Baseline: {baseline_ao:.6f}')

    best_so_far = baseline_ao
    best_iters = [0]
    best_vals = [baseline_ao]
    for r in results:
        if r['ao'] > best_so_far and r['status'] in ('kept', 'baseline'):
            best_so_far = r['ao']
            best_iters.append(r['iter'])
            best_vals.append(best_so_far)
    if best_iters:
        best_iters.append(max(iters))
        best_vals.append(best_so_far)
        ax.step(best_iters, best_vals, where='post', color='green', alpha=0.5,
                linewidth=2, label=f'Best: {best_so_far:.6f}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Composite Score')
    ax.set_title('Oxi Research Loop Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    try:
        call_tool("artifact_upload", path=str(chart_path), kind="image",
                  description="Composite score progress chart")
    except Exception:
        print("  (chart saved locally, artifact_upload not available)")


def extract_structured_output(text):
    """Extract the last JSON block from subagent output for title/description/changes."""
    json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not json_blocks:
        json_blocks = re.findall(r'(\{"title".*?\})', text, re.DOTALL)
    if json_blocks:
        try:
            data = json.loads(json_blocks[-1])
            title = data.get("title", "").strip().replace("|", "-").replace("\n", " ")[:80]
            description = data.get("description", "").strip()
            changes = data.get("changes", "").strip()
            if title and validate_title(title):
                return title, description, changes
        except json.JSONDecodeError:
            pass
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("```") and validate_title(line):
            return line[:80].replace("|", "-"), "", ""
    return "Unknown change", "", ""


def validate_title(title):
    """Reject titles that are obviously agent internal monologue."""
    noise_starts = [
        "i've analyzed", "let me", "i'll ", "i will", "ok,", "alright",
        "i need to", "i should", "now i", "first,", "looking at",
        "i have", "i can", "i want", "let's",
    ]
    lower = title.lower()
    return not any(lower.startswith(p) for p in noise_starts)


def main():
    AUTORESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_RUNS.mkdir(parents=True, exist_ok=True)

    state = load_state()
    existing_results = parse_existing_log()

    if state.get("results"):
        state["results"] = [r for r in state["results"] if r.get('status') != 'error']

    if not RESEARCH_LOG.exists() or RESEARCH_LOG.stat().st_size == 0:
        with open(RESEARCH_LOG, 'w') as f:
            f.write("# OXI Research Log\n\n| Iter | Description | Score | Δ vs prev best | Kept |\n|------|-------------|-------|----------------|------|\n")

    # Run baseline if needed
    if state["baseline_score"] == 0.0:
        print("=== Running baseline training ===")
        git_revert()

        print("  Building release binary...")
        ok, stderr = build_binary()
        if not ok:
            print(f"  Build failed: {stderr[-500:]}")
            return

        baseline, baseline_components = run_training("baseline", seed=42)
        state["baseline_score"] = baseline
        state["best_score"] = baseline
        state["best_run_dir"] = str(RESEARCH_RUNS / "baseline")
        state["best_components"] = baseline_components
        if not any(r['iter'] == 0 for r in existing_results):
            with open(RESEARCH_LOG, 'a') as f:
                f.write(f"| 0 | Baseline (no changes) | {baseline:.6f} {format_components(baseline_components)} | — | baseline |\n")
            existing_results.append({'iter': 0, 'title': 'Baseline', 'ao': baseline, 'status': 'baseline'})
        save_state(state)
        print(f"Baseline score: {baseline:.6f} {format_components(baseline_components)}")
    else:
        print(f"Resuming with {len(existing_results)} prior results, best score: {state['best_score']:.6f}")

    best = state["best_score"]
    baseline = state["baseline_score"]
    all_results = [r for r in state.get("results", existing_results) if r.get('status') != 'error']

    consecutive_errors = 0

    for iteration_count in range(MAX_ITERATIONS):
        i = len(all_results)
        print(f"\n{'='*60}")
        print(f"=== Iteration {i} | {iteration_count + 1}/{MAX_ITERATIONS} | best: {best:.6f} | baseline: {baseline:.6f} ===")
        print(f"{'='*60}")

        try:
            git_revert()

            best_comps = state.get(
                "best_components",
                {"top1": 0.0, "wdl": 0.0, "aux": 0.0, "calibration": 0.0},
            )

            iter_dir = AUTORESEARCH_DIR / str(i)
            iter_dir.mkdir(parents=True, exist_ok=True)

            # --- Propose experiment ---
            idea_prompt = f"""You are an ML researcher improving a chess move-prediction transformer.

**Start by reading `research/architecture.md`** — it contains a comprehensive reference of the model architecture, input representation, all output heads, loss functions, GradNorm, optimizer config, LR scheduling, data pipeline, metrics logging, and evaluation scoring. This is your primary reference for understanding the codebase.

## Evaluation Metric

Your changes are evaluated on a **composite score** computed as:
  score = {WEIGHT_TOP1}*top1_accuracy + {WEIGHT_WDL:.4f}*wdl_accuracy + {WEIGHT_AUX}*aux_accuracy + {WEIGHT_CALIBRATION:.4f}*calibration
where each component is computed from the last {METRIC_WINDOW} logged values during a {TRAINING_TIMEOUT//60}-minute training run.

Components:
- **top1_accuracy** (weight {WEIGHT_TOP1}): Policy head move prediction accuracy (0-1).
- **wdl_accuracy** (weight {WEIGHT_WDL:.4f}): Win/draw/loss prediction accuracy (0-1, normalized if logged as percentages).
- **aux_accuracy** (weight {WEIGHT_AUX}): Average of from-square and to-square auxiliary head accuracy (0-1).
- **calibration** (weight {WEIGHT_CALIBRATION:.4f}): `cp_loss_calibration_overall` — computed as `exp(-abs_cpl_error / 15)`, so it's very sensitive to centipawn-loss accuracy. This rewards matching the human centipawn-loss profile, not just predicting the exact move.

Important calibration details:
- The main human-facing diagnostics are the signed Elo-band CPL calibration metrics, but the loop score uses the bounded overall calibration metric.
- A change that improves top-1 while harming human-strength calibration can lose overall.

Current best score: {best:.6f} {format_components(best_comps)} | Baseline: {baseline:.6f} | Iteration {i}

## Important Context

- **Loss weighting is handled by GradNorm** — the model uses adaptive GradNorm reweighting across policy, value, aux, and calibration. You do NOT need to manually tune loss weights unless you are deliberately changing the training objective.
- **This is a time-constrained training run** ({TRAINING_TIMEOUT//60} minutes). Throughput matters.
- **The goal is to prep for a larger production run.** We want the best architecture/training recipe for both move prediction and human-strength calibration. Improvements to training dynamics, loss functions, architecture, or calibration behavior are all valuable.
- **Architecture is not pinned.** The model's embed_dim, num_layers, and head_dim come from defaults in `src/config.rs`. If you want to change the architecture, modify those defaults. The training command does NOT pass these as CLI args, so your config.rs changes will take effect.

## Risk Tolerance

A change that scores 5% worse is free — it gets reverted automatically. A change that scores 2% better is kept forever. **The expected value of bold experiments is high.** Don't optimize for keep rate, optimize for finding large improvements.

If your proposed change would not plausibly move the score by at least 1%, it's too small. One logical change can still be large — replacing the attention mechanism is one change, adding a new auxiliary head is one change.

## What To Do

Read the experiment log at research_log.md to see what has been tried. Do not repeat failed ideas.

Explore the codebase, then propose **one logical change**. Don't bundle unrelated modifications, but the change itself can be ambitious — a new architecture component, a different training objective, a structural refactor of how something works.

Verify your change compiles: `cargo check --features "train,backend-tch"`

Feel free to use the ./research directory to make notes and see notes from previous agents.

## Acceptance Criteria

Training runs for {TRAINING_TIMEOUT//60} minutes then is killed. Changes are kept if:
- Block bootstrap p < {ALPHA} (autocorrelation-corrected)
- Cohen's d >= {MIN_COHEN_D}

## Rules

DO NOT TOUCH:
- CLI argument handling or data loading
- The LR scheduler type (reduce-on-plateau only, no cosine/cyclic). Adjusting LR values, warmup, or plateau parameters is fine.
- The evaluation metric or how accuracy is measured
- Loss weights by default — GradNorm handles this unless the experiment is specifically about the objective itself
- research_log.md and research_runs/ — read-only (except research_runs/run_{i}/conclusion.md which you must create)

## Required Output

Write a file at research_runs/run_{i}/conclusion.md describing:
1. **What you did** — short summary
2. **Why you chose this** — detailed reasoning, referencing prior experiments
3. **What you expect** — predicted outcome and confidence level

End your response with:
```json
{{"title": "<=10 word title", "description": "1-2 sentence description", "changes": "files changed and what was done"}}
```"""

            print(f"  Requesting experiment idea from subagent...")
            idea_output = call_tool("subagent_run", task=idea_prompt, role="full").get("output", "")

            (iter_dir / "idea.md").write_text(idea_output)

            title, description, changes = extract_structured_output(idea_output)
            print(f"  Title: {title}")
            if description:
                print(f"  Description: {description}")

            # --- Phase 3: Compile check ---
            print(f"  Checking compilation...")
            compiles, stderr = compile_check()
            if not compiles:
                print(f"  ❌ Compile failed!")
                print(f"  {stderr[-500:]}")
                git_revert()
                result = {'iter': i, 'title': f'[COMPILE FAIL] {title}', 'ao': 0.0, 'status': 'fail'}
                all_results.append(result)
                with open(RESEARCH_LOG, 'a') as f:
                    f.write(f"| {i} | [COMPILE FAIL] {title} | 0.000000 | — | ❌ |\n")
                state["results"] = all_results
                save_state(state)
                consecutive_errors = 0
                continue

            print(f"  ✅ Compilation succeeded")

            # --- Phase 4: Build release binary ---
            print(f"  Building release binary...")
            build_ok, build_stderr = build_binary()
            if not build_ok:
                print(f"  ❌ Build failed!")
                print(f"  {build_stderr[-500:]}")
                git_revert()
                result = {'iter': i, 'title': f'[BUILD FAIL] {title}', 'ao': 0.0, 'status': 'fail'}
                all_results.append(result)
                with open(RESEARCH_LOG, 'a') as f:
                    f.write(f"| {i} | [BUILD FAIL] {title} | 0.000000 | — | ❌ |\n")
                state["results"] = all_results
                save_state(state)
                consecutive_errors = 0
                continue

            # --- Phase 5: Training ---
            seed = 42
            score, components = run_training(f"run_{i}", seed)

            # --- Phase 6: Statistical comparison ---
            best_dir = Path(state["best_run_dir"])
            improved, stat = check_improvement(RESEARCH_RUNS / f"run_{i}", best_dir)
            if "error" not in stat:
                stat_str = f"Δ={stat['delta']:+.4f}  p={stat['p']:.4f}  d={stat['d']:+.3f}  CI=[{stat['ci_low']:+.4f}, {stat['ci_high']:+.4f}]"
            else:
                stat_str = stat["error"]

            prev_components = state.get("best_components", {"top1": 0.0, "wdl": 0.0, "aux": 0.0})
            delta_str = format_delta(score, components, best, prev_components)

            if improved:
                print(f"  ✅ IMPROVEMENT: {score:.6f}  ({stat_str})")
                best = score
                state["best_run_dir"] = str(RESEARCH_RUNS / f"run_{i}")
                state["best_components"] = components
                git_commit(f"research iter {i}: {title} (score={score:.6f})")
                status = "kept"
                status_emoji = "✅"
            else:
                print(f"  ❌ No improvement: {score:.6f}  ({stat_str})")
                git_revert()
                status = "discarded"
                status_emoji = "❌"

            result_entry = {'iter': i, 'title': title, 'ao': score, 'status': status}
            all_results.append(result_entry)
            with open(RESEARCH_LOG, 'a') as f:
                f.write(f"| {i} | {title} | {score:.6f} {format_components(components)} | {delta_str} | {status_emoji} |\n")

            state["best_score"] = best
            state["results"] = all_results
            save_state(state)
            generate_chart(all_results, baseline)
            consecutive_errors = 0

        except Exception as e:
            git_revert()
            consecutive_errors += 1
            msg = str(e)

            backoff = min(MAX_BACKOFF_SECS, INITIAL_BACKOFF_SECS * (2 ** (consecutive_errors - 1)))
            print(f"  ⚠️ Error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {msg[:120]}")
            traceback.print_exc()

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"  🛑 {MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping loop")
                result = {'iter': i, 'title': f'[ERROR] {msg[:50]}', 'ao': 0.0, 'status': 'error'}
                all_results.append(result)
                with open(RESEARCH_LOG, 'a') as f:
                    f.write(f"| {i} | [ERROR x{consecutive_errors}] {msg[:50]} | 0.000000 | — | ⚠️ |\n")
                state["results"] = all_results
                save_state(state)
                break

            print(f"  Backing off {backoff}s before retry...")
            time.sleep(backoff)

    print(f"\n{'='*60}")
    print(f"=== RESEARCH COMPLETE ===")
    print(f"  Baseline score: {baseline:.6f}")
    print(f"  Best score:     {best:.6f}")
    print(f"  Improvement:   {best - baseline:+.6f}")
    kept = [r for r in all_results if r['status'] == 'kept']
    print(f"  Kept changes:  {len(kept)} / {len([r for r in all_results if r.get('status') != 'baseline'])}")
    print(f"  Log: {RESEARCH_LOG}")

    generate_chart(all_results, baseline)

    try:
        call_tool("artifact_upload", path=str(RESEARCH_LOG), kind="file",
                  description="Final research log with all iterations")
    except Exception:
        print("  (artifact_upload not available)")


if __name__ == "__main__":
    main()
