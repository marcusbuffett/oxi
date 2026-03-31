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
MODEL_BATCH = 1536

# Metric: average of last 100 top-1 accuracy values
AO_WINDOW = 100

def parse_ao(log_dir):
    """Parse ao metric from top1_accuracy.log (average of last AO_WINDOW values)."""
    log_file = Path(log_dir) / "metrics_logs" / "top1_accuracy.log"
    if not log_file.exists():
        return 0.0
    with open(log_file, 'r') as f:
        lines = f.readlines()
    recent = lines[-AO_WINDOW:] if len(lines) >= AO_WINDOW else lines
    values = []
    for line in recent:
        if '\t' in line:
            try:
                values.append(float(line.strip().split('\t')[1]))
            except (ValueError, IndexError):
                pass
    return sum(values) / len(values) if values else 0.0


def compile_check():
    """Run cargo check and return (success, stderr)."""
    result = subprocess.run(
        ["cargo", "check", "--features", "train,backend-tch"],
        cwd=WORKSPACE, capture_output=True, text=True, timeout=300
    )
    return result.returncode == 0, result.stderr


def run_training(run_name, seed):
    """Run training for TRAINING_TIMEOUT seconds and return ao metric."""
    log_dir = RESEARCH_RUNS / run_name
    # Clean stale metrics from previous runs so parse_ao never reads old data
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
        return 0.0

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
    ]
    try:
        proc = subprocess.Popen(cmd, cwd=WORKSPACE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait(timeout=TRAINING_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    ao = parse_ao(log_dir)
    print(f"  ao{AO_WINDOW} = {ao:.6f}")
    return ao


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
        return json.loads(STATE_FILE.read_text())
    return {"iteration": 1, "best_ao": 0.0, "baseline_ao": 0.0, "results": []}


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
    ax.set_ylabel(f'ao{AO_WINDOW} Accuracy')
    ax.set_title('Oxi Research Loop Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    try:
        call_tool("artifact_upload", path=str(chart_path), kind="image",
                  description=f"ao{AO_WINDOW} accuracy progress chart")
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
            f.write(f"# OXI Research Log\n\n| Iter | Description | ao{AO_WINDOW} | Kept |\n|------|-------------|------|------|\n")

    # Run baseline if needed
    if state["baseline_ao"] == 0.0:
        print("=== Running baseline training ===")
        git_revert()
        baseline = run_training("baseline", seed=42)
        state["baseline_ao"] = baseline
        state["best_ao"] = baseline
        if not any(r['iter'] == 0 for r in existing_results):
            with open(RESEARCH_LOG, 'a') as f:
                f.write(f"| 0 | Baseline (no changes) | {baseline:.6f} | baseline |\n")
            existing_results.append({'iter': 0, 'title': 'Baseline', 'ao': baseline, 'status': 'baseline'})
        save_state(state)
        print(f"Baseline ao{AO_WINDOW}: {baseline:.6f}")
    else:
        print(f"Resuming from iteration {state['iteration']}, best ao{AO_WINDOW}: {state['best_ao']:.6f}")

    best = state["best_ao"]
    baseline = state["baseline_ao"]
    start_iter = state["iteration"]
    all_results = [r for r in state.get("results", existing_results) if r.get('status') != 'error']
    end_iter = start_iter + MAX_ITERATIONS

    for i in range(start_iter, end_iter):
        print(f"\n{'='*60}")
        print(f"=== Iteration {i} | {i - start_iter + 1}/{MAX_ITERATIONS} | best: {best:.6f} | baseline: {baseline:.6f} ===")
        print(f"{'='*60}")

        try:
            git_revert()

            idea_prompt = f"""You are an ML researcher improving a chess move-prediction transformer.

Your goal is to improve top-1 move prediction accuracy (measured as "ao{AO_WINDOW}" — average of last {AO_WINDOW} accuracy values during a 30-minute training run).

Current best ao{AO_WINDOW}: {best:.6f} | Baseline: {baseline:.6f} | Iteration {i}

The experiment log is at research_log.md — read it to see what has been tried. Do not repeat failed ideas.

Explore the codebase (start with src/) to understand the architecture, then make EXACTLY ONE focused change. You are free to make architectural changes, hyperparameter changes, ablation tests, loss function modifications, or anything else you think could help.

Verify your change compiles: `cargo check --features "train,backend-tch"`

The training command that will be run to evaluate your change:
```
cargo run --release --features "backend-tch train" -- train --pretrain-samples=0 --data-path=../data --physical-batch-size={MODEL_BATCH} --num-layers={MODEL_LAYERS} --embed-dim={MODEL_EMBED} --seed=<varies> --log-dir=<run_dir> --disable-tui
```
Training runs for {TRAINING_TIMEOUT} seconds then is killed, so the model must converge quickly.

Don't change CLI argument handling or data loading.

End your response with:
```json
{{"title": "<=10 word title", "description": "1-2 sentence description", "changes": "files changed and what was done"}}
```"""

            print(f"  Requesting experiment idea from subagent...")
            # Use a different seed per iteration for training stochasticity
            seed = 42 + i
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
                state["iteration"] = i + 1
                save_state(state)
                continue

            print(f"  ✅ Compilation succeeded")

            ao = run_training(f"run_{i}", seed=seed)

            if ao > best:
                print(f"  ✅ IMPROVEMENT: {ao:.6f} > {best:.6f} (+{ao-best:.6f})")
                best = ao
                git_commit(f"research iter {i}: {title} (ao{AO_WINDOW}={ao:.6f})")
                status = "kept"
                status_emoji = "✅"
            else:
                print(f"  ❌ No improvement: {ao:.6f} <= {best:.6f}")
                git_revert()
                status = "discarded"
                status_emoji = "❌"

            result = {'iter': i, 'title': title, 'ao': ao, 'status': status}
            all_results.append(result)
            with open(RESEARCH_LOG, 'a') as f:
                f.write(f"| {i} | {title} | {ao:.6f} | {status_emoji} |\n")

            state["iteration"] = i + 1
            state["best_ao"] = best
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
            state["iteration"] = i + 1
            save_state(state)

    print(f"\n{'='*60}")
    print(f"=== RESEARCH COMPLETE ===")
    print(f"  Baseline ao{AO_WINDOW}: {baseline:.6f}")
    print(f"  Best ao{AO_WINDOW}:     {best:.6f}")
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
