"""
Automated ML research loop for the Oxi chess model.
Runs iterations of: propose change → compile check → train → evaluate → keep/discard.
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
TRAINING_TIMEOUT = 1800  # 30 minutes
MAX_ITERATIONS = 50

# Research model config (~57 GB memory, ~0.5-0.67x production on each axis)
MODEL_LAYERS = 12
MODEL_EMBED = 256
MODEL_BATCH = 512

# Composite metric: weighted combination of top-1 accuracy, WDL accuracy, and aux head accuracy
# All components are averaged over the last METRIC_WINDOW values from their respective log files.
# top1_accuracy: 0-1 fraction, wdl_accuracy: 0-100 percentage (divided by 100), aux: 0-1 fraction
METRIC_WINDOW = 100
WEIGHT_TOP1 = 1.0
WEIGHT_WDL = 1.0 / 3.0
WEIGHT_AUX = 0.1  # average of from-square and to-square aux accuracy

# Acceptance criterion: Welch's t-test on composite score, last METRIC_WINDOW iterations.
# A change is kept only if p < ALPHA and Cohen's d >= MIN_COHEN_D.
MIN_COHEN_D = 0.5   # medium effect size
ALPHA = 0.05


def load_metric_array(log_dir, metric_name):
    """Parse a metric TSV log, returning a numpy array of the last METRIC_WINDOW values."""
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
    """Return a numpy array of per-step composite scores (last METRIC_WINDOW aligned steps)."""
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

    common = sorted(set(top1_map) & set(wdl_map) & set(from_map) & set(to_map))
    if not common:
        return np.array([])

    values = []
    for s in common:
        aux = (from_map[s] + to_map[s]) / 2.0
        score = WEIGHT_TOP1 * top1_map[s] + WEIGHT_WDL * (wdl_map[s] / 100.0) + WEIGHT_AUX * aux
        values.append(score)

    arr = np.array(values)
    return arr[-METRIC_WINDOW:] if len(arr) >= METRIC_WINDOW else arr


def parse_metric_log(log_dir, metric_name):
    """Return the mean of the last METRIC_WINDOW values for a metric (scalar, for reporting)."""
    arr = load_metric_array(log_dir, metric_name)
    return float(arr.mean()) if len(arr) > 0 else 0.0


def parse_composite_score(log_dir):
    """Compute composite score from multiple metrics. Returns (score, components_dict)."""
    top1 = parse_metric_log(log_dir, "top1_accuracy")
    wdl_raw = parse_metric_log(log_dir, "wdl_accuracy")
    wdl = wdl_raw / 100.0
    aux_from = parse_metric_log(log_dir, "aux_from_square_accuracy")
    aux_to = parse_metric_log(log_dir, "aux_to_square_accuracy")
    aux = (aux_from + aux_to) / 2.0
    score = WEIGHT_TOP1 * top1 + WEIGHT_WDL * wdl + WEIGHT_AUX * aux
    components = {"top1": top1, "wdl": wdl, "aux": aux}
    return score, components


def check_improvement(current_dir, baseline_dir):
    """
    Test whether current_dir is a statistically significant improvement over baseline_dir.
    Uses Welch's t-test + Cohen's d on the composite score (last METRIC_WINDOW steps).
    Returns (is_improvement: bool, stats: dict).
    """
    a = load_composite_array(baseline_dir)
    b = load_composite_array(current_dir)
    if len(a) < 2 or len(b) < 2:
        return False, {"error": "insufficient data"}

    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    delta = b.mean() - a.mean()
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    d = delta / pooled_std if pooled_std > 0 else 0.0

    passed = (p_val < ALPHA) and (d >= MIN_COHEN_D)
    return passed, {"delta": delta, "p": p_val, "d": d, "mean_new": b.mean(), "mean_old": a.mean()}


def best_run_dir_by_score():
    """Scan research_runs/ and return the dir with the highest composite score."""
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
    """Run cargo check and return (success, stderr)."""
    result = subprocess.run(
        ["cargo", "check", "--features", "train,backend-tch"],
        cwd=WORKSPACE, capture_output=True, text=True, timeout=300
    )
    return result.returncode == 0, result.stderr


def format_components(components):
    """Format component scores for display: (top1=0.41, wdl=0.52, aux=0.38)"""
    return f"(top1={components['top1']:.4f}, wdl={components['wdl']:.4f}, aux={components['aux']:.4f})"


def run_training(run_name, seed):
    """Run training for TRAINING_TIMEOUT seconds and return (composite_score, components_dict)."""
    log_dir = RESEARCH_RUNS / run_name
    # Clean stale metrics from previous runs so parse never reads old data
    metrics_dir = log_dir / "metrics_logs"
    if metrics_dir.exists():
        import shutil
        shutil.rmtree(metrics_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Pre-build to cache compilation
    print(f"  Building release binary...")
    build_result = subprocess.run(
        ["cargo", "build", "--release", "--features", "backend-tch train"],
        cwd=WORKSPACE, capture_output=True, text=True, timeout=300
    )
    if build_result.returncode != 0:
        print(f"  Build failed: {build_result.stderr[-500:]}")
        return 0.0, {"top1": 0.0, "wdl": 0.0, "aux": 0.0}

    print(f"  Running training for {TRAINING_TIMEOUT}s (seed={seed})...")
    cmd = [
        "cargo", "run", "--release", "--features", "backend-tch train", "--",
        "train",
        "--pretrain-samples=0",
        "--data-path=../data",
        f"--physical-batch-size={MODEL_BATCH}",
        f"--num-layers={MODEL_LAYERS}",
        f"--embed-dim={MODEL_EMBED}",
        f"--seed={seed}",
        f"--log-dir={log_dir}",
        "--disable-tui",
        "--warmup-multiplier=0.1",
        "--log-gradient-breakdown",
        "--full-metrics-interval=100",
    ]
    try:
        proc = subprocess.Popen(cmd, cwd=WORKSPACE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait(timeout=TRAINING_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    score, components = parse_composite_score(log_dir)
    print(f"  score = {score:.6f} {format_components(components)}")
    return score, components


def git_revert():
    """Revert all uncommitted changes to source files."""
    subprocess.run(["git", "checkout", "--", "src/"], cwd=WORKSPACE, capture_output=True)
    subprocess.run(["git", "checkout", "--", "Cargo.toml"], cwd=WORKSPACE, capture_output=True)


def git_commit(message):
    """Commit source changes."""
    subprocess.run(["git", "add", "src/", "Cargo.toml"], cwd=WORKSPACE, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=WORKSPACE, capture_output=True)


def load_state():
    """Load persisted state for resumability."""
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
    else:
        state = {"best_score": 0.0, "baseline_score": 0.0, "results": []}
    # Infer best_run_dir from disk if not already stored
    if not state.get("best_run_dir"):
        d = best_run_dir_by_score()
        state["best_run_dir"] = str(d) if d else str(RESEARCH_RUNS / "baseline")
    return state


def save_state(state):
    """Persist state."""
    STATE_FILE.write_text(json.dumps(state, indent=2))


def parse_existing_log():
    """Parse existing research_log.md for results."""
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
    """Generate and upload a progress chart."""
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
            if title:
                return title, description, changes
        except json.JSONDecodeError:
            pass
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("```"):
            return line[:80].replace("|", "-"), "", ""
    return "Unknown change", "", ""


def main():
    AUTORESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_RUNS.mkdir(parents=True, exist_ok=True)

    state = load_state()
    existing_results = parse_existing_log()

    if state.get("results"):
        state["results"] = [r for r in state["results"] if r.get('status') != 'error']

    if not RESEARCH_LOG.exists() or RESEARCH_LOG.stat().st_size == 0:
        with open(RESEARCH_LOG, 'w') as f:
            f.write(f"# OXI Research Log\n\n| Iter | Description | Score | Kept |\n|------|-------------|-------|------|\n")

    # Run baseline if needed
    if state["baseline_score"] == 0.0:
        print("=== Running baseline training ===")
        git_revert()
        baseline, baseline_components = run_training("baseline", seed=42)
        state["baseline_score"] = baseline
        state["best_score"] = baseline
        state["best_run_dir"] = str(RESEARCH_RUNS / "baseline")
        if not any(r['iter'] == 0 for r in existing_results):
            with open(RESEARCH_LOG, 'a') as f:
                f.write(f"| 0 | Baseline (no changes) | {baseline:.6f} {format_components(baseline_components)} | baseline |\n")
            existing_results.append({'iter': 0, 'title': 'Baseline', 'ao': baseline, 'status': 'baseline'})
        save_state(state)
        print(f"Baseline score: {baseline:.6f} {format_components(baseline_components)}")
    else:
        print(f"Resuming with {len(existing_results)} prior results, best score: {state['best_score']:.6f}")

    best = state["best_score"]
    baseline = state["baseline_score"]
    all_results = [r for r in state.get("results", existing_results) if r.get('status') != 'error']

    for iteration_count in range(MAX_ITERATIONS):
        i = len(all_results)
        print(f"\n{'='*60}")
        print(f"=== Iteration {i} | {iteration_count + 1}/{MAX_ITERATIONS} | best: {best:.6f} | baseline: {baseline:.6f} ===")
        print(f"{'='*60}")

        try:
            git_revert()

            idea_prompt = f"""You are an ML researcher improving a chess move-prediction transformer.

Your goal is to improve a composite score computed as:
  score = {WEIGHT_TOP1}*top1_accuracy + {WEIGHT_WDL:.4f}*wdl_accuracy + {WEIGHT_AUX}*aux_accuracy
where each component is the average of its last {METRIC_WINDOW} logged values during a {TRAINING_TIMEOUT//60}-minute training run.
- top1_accuracy: policy head move prediction accuracy (0-1)
- wdl_accuracy: win/draw/loss prediction accuracy (0-1)
- aux_accuracy: average of from-square and to-square auxiliary head accuracy (0-1)

Current best score: {best:.6f} | Baseline: {baseline:.6f} | Iteration {i}

The experiment log is at research_log.md — read it to see what has been tried. Do not repeat failed ideas.

Explore the codebase (start with src/) to understand the architecture, then make up to 3 related changes. You are free to make architectural changes, hyperparameter changes, ablation tests, loss function modifications, or anything else you think could help.

Verify your change compiles: `cargo check --features "train,backend-tch"`

The training command that will be run to evaluate your change:
```
cargo run --release --features "backend-tch train" -- train --pretrain-samples=0 --data-path=../data --physical-batch-size={MODEL_BATCH} --num-layers={MODEL_LAYERS} --embed-dim={MODEL_EMBED} --warmup-multiplier=0.1 --log-gradient-breakdown --full-metrics-interval=100 --seed=<varies> --log-dir=<run_dir> --disable-tui
```
Training runs for {TRAINING_TIMEOUT} seconds then is killed, so the model must converge quickly. Changes are kept only if they achieve a statistically significant improvement (Welch's t-test p < {ALPHA}, Cohen's d >= {MIN_COHEN_D}) on the composite score compared to the current best run. Aim for meaningful changes — small tweaks that improve the mean by less than ~1 pooled std dev will be discarded.

Previous run logs are in research_runs/. Each run directory contains:
- `metrics_logs/` — per-metric TSV files (top1_accuracy.log, wdl_accuracy.log, aux_from_square_accuracy.log, aux_to_square_accuracy.log, policy_loss.log, total_loss.log, etc.)
- `train.log` — detailed training log including per-layer gradient norms, weight norms, update ratios, and per-head gradient statistics (logged every 100 iterations)

DO NOT TOUCH:
- CLI argument handling or data loading
- The LR scheduler type — we use reduce-on-plateau so training is duration-agnostic. Do not add cosine decay, cyclic schedules, or any schedule that depends on knowing total training steps. Adjusting the LR value itself, warmup, or plateau detector parameters is fine.
- The evaluation metric or how accuracy is measured
- research_log.md and research_runs/ — these are read-only. You may read them for context but do not write to, modify, or delete any files in them. The one exception is research_runs/run_{i}/conclusion.md which you must create (see below).

Before your final response, write a file at research_runs/run_{i}/conclusion.md describing:
1. **What you did** — a short summary of the change(s) made.
2. **Why you chose this** — a detailed breakdown of your reasoning: what prior results, architectural insights, or hypotheses led you to believe this was the best option to try. Reference specific prior experiments from research_log.md if relevant.
3. **What you expect** — what outcome you predict and why.

Be concise but thorough on the reasoning.

End your response with:
```json
{{"title": "<=10 word title", "description": "1-2 sentence description", "changes": "files changed and what was done"}}
```"""

            print(f"  Requesting experiment idea from subagent...")
            seed = 42
            idea_output = call_tool("subagent.run", task=idea_prompt).get("output", "")

            iter_dir = AUTORESEARCH_DIR / str(i)
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "idea.md").write_text(idea_output)

            title, description, changes = extract_structured_output(idea_output)
            print(f"  Title: {title}")
            if description:
                print(f"  Description: {description}")

            print(f"  Checking compilation...")
            compiles, stderr = compile_check()
            if not compiles:
                print(f"  ❌ Compile failed!")
                print(f"  {stderr[-500:]}")
                git_revert()
                result = {'iter': i, 'title': f'[COMPILE FAIL] {title}', 'ao': 0.0, 'status': 'fail'}
                all_results.append(result)
                with open(RESEARCH_LOG, 'a') as f:
                    f.write(f"| {i} | [COMPILE FAIL] {title} | 0.000000 | ❌ |\n")
                save_state(state)
                continue

            print(f"  ✅ Compilation succeeded")

            score, components = run_training(f"run_{i}", seed=seed)

            best_dir = Path(state.get("best_run_dir", str(RESEARCH_RUNS / "baseline")))
            improved, stat = check_improvement(RESEARCH_RUNS / f"run_{i}", best_dir)
            if "error" not in stat:
                stat_str = f"Δ={stat['delta']:+.4f}  p={stat['p']:.4f}  d={stat['d']:+.3f}"
            else:
                stat_str = stat["error"]

            if improved:
                print(f"  ✅ IMPROVEMENT: {score:.6f}  ({stat_str})")
                best = score
                state["best_run_dir"] = str(RESEARCH_RUNS / f"run_{i}")
                git_commit(f"research iter {i}: {title} (score={score:.6f})")
                status = "kept"
                status_emoji = "✅"
            else:
                print(f"  ❌ No improvement: {score:.6f}  ({stat_str})  need p<{ALPHA} and d>={MIN_COHEN_D}")
                git_revert()
                status = "discarded"
                status_emoji = "❌"

            result = {'iter': i, 'title': title, 'ao': score, 'status': status}

            all_results.append(result)
            with open(RESEARCH_LOG, 'a') as f:
                f.write(f"| {i} | {title} | {score:.6f} {format_components(components)} | {status_emoji} |\n")

            state["best_score"] = best
            state["results"] = all_results
            save_state(state)

            generate_chart(all_results, baseline)

        except Exception as e:
            print(f"  ⚠️ Error in iteration {i}: {e}")
            traceback.print_exc()
            git_revert()
            result = {'iter': i, 'title': f'[ERROR] {str(e)[:50]}', 'ao': 0.0, 'status': 'error'}
            all_results.append(result)
            with open(RESEARCH_LOG, 'a') as f:
                f.write(f"| {i} | [ERROR] {str(e)[:50]} | 0.000000 | ⚠️ |\n")
            save_state(state)

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