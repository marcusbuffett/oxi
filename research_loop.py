"""
Automated ML research loop for the Oxi chess model.
Runs 20 iterations of: propose change → compile check → train → evaluate → keep/discard.
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
TRAINING_TIMEOUT = 600  # 10 minutes
MAX_ITERATIONS = 50  # 50 iterations from current start point

# Key source files for the subagent to know about
KEY_FILES = [
    "src/model.rs",
    "src/config.rs",
    "src/custom_training.rs",
    "src/factorized_policy.rs",
    "src/relative_position_transformer.rs",
    "src/smolgen.rs",
]


def parse_ao10(log_dir):
    """Parse ao10 from top1_accuracy.log (average of last 10 values)."""
    log_file = Path(log_dir) / "metrics_logs" / "top1_accuracy.log"
    if not log_file.exists():
        return 0.0
    with open(log_file, 'r') as f:
        lines = f.readlines()
    recent = lines[-10:] if len(lines) >= 10 else lines
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


def run_training(run_name):
    """Run training for TRAINING_TIMEOUT seconds and return ao10."""
    log_dir = RESEARCH_RUNS / run_name
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

    print(f"  Running training for {TRAINING_TIMEOUT}s...")
    cmd = [
        "cargo", "run", "--release", "--features", "backend-tch train", "--",
        "train",
        "--pretrain-samples=0",
        "--data-path=../data",
        "--physical-batch-size=512",
        "--num-layers=6",
        "--embed-dim=192",
        f"--log-dir={log_dir}",
        "--disable-tui",
    ]
    try:
        subprocess.run(cmd, cwd=WORKSPACE, capture_output=True, text=True, timeout=TRAINING_TIMEOUT)
    except subprocess.TimeoutExpired:
        pass  # Expected - training runs until timeout

    # Kill any lingering training processes
    subprocess.run(["pkill", "-f", "oxi.*train"], capture_output=True)
    time.sleep(2)

    ao10 = parse_ao10(log_dir)
    print(f"  ao10 = {ao10:.6f}")
    return ao10


def git_revert():
    """Revert all uncommitted changes to source files."""
    subprocess.run(["git", "checkout", "--", "src/"], cwd=WORKSPACE, capture_output=True)
    subprocess.run(["git", "checkout", "--", "Cargo.toml"], cwd=WORKSPACE, capture_output=True)


def git_commit(message):
    """Commit all changes."""
    subprocess.run(["git", "add", "-A"], cwd=WORKSPACE, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=WORKSPACE, capture_output=True)


def load_state():
    """Load persisted state for resumability."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"iteration": 1, "best_ao10": 0.0, "baseline_ao10": 0.0, "results": []}


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
                    'ao10': float(m.group(3)),
                    'status': m.group(4).strip(),
                })
    return results


def generate_chart(results, baseline_ao10):
    """Generate and upload a progress chart."""
    if not results:
        return
    # Filter to only real results (not errors from chart upload)
    results = [r for r in results if r['status'] not in ('error',)]
    if not results:
        return
    chart_path = RESEARCH_RUNS / "progress.png"
    fig, ax = plt.subplots(figsize=(12, 6))

    iters = [r['iter'] for r in results]
    ao10s = [r['ao10'] for r in results]
    colors = []
    for r in results:
        if r['status'] in ('kept', 'baseline'):
            colors.append('green')
        elif r['status'] == 'fail':
            colors.append('gray')
        else:
            colors.append('red')

    ax.scatter(iters, ao10s, c=colors, s=80, zorder=3)
    if baseline_ao10 > 0:
        ax.axhline(y=baseline_ao10, color='blue', linestyle='--', alpha=0.7,
                   label=f'Baseline: {baseline_ao10:.6f}')

    # Draw the "best so far" line
    best_so_far = baseline_ao10
    best_iters = [0]
    best_vals = [baseline_ao10]
    for r in results:
        if r['ao10'] > best_so_far and r['status'] in ('kept', 'baseline'):
            best_so_far = r['ao10']
            best_iters.append(r['iter'])
            best_vals.append(best_so_far)
    if best_iters:
        best_iters.append(max(iters))
        best_vals.append(best_so_far)
        ax.step(best_iters, best_vals, where='post', color='green', alpha=0.5,
                linewidth=2, label=f'Best: {best_so_far:.6f}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('ao10 Accuracy')
    ax.set_title('Oxi Research Loop Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    try:
        call_tool("artifact_upload", path=str(chart_path), kind="image",
                  description="ao10 accuracy progress chart")
    except Exception:
        print("  (chart saved locally, artifact_upload not available)")


def get_recent_context(max_recent=5):
    """Read recent autoresearch outputs for context."""
    context_parts = []
    if AUTORESEARCH_DIR.exists():
        dirs = sorted(AUTORESEARCH_DIR.iterdir(), key=lambda d: int(d.name) if d.name.isdigit() else 0)
        for d in dirs[-max_recent:]:
            idea_file = d / "idea.md"
            if idea_file.exists():
                content = idea_file.read_text()[:3000]
                context_parts.append(f"## Iteration {d.name}\n{content}")
    return "\n\n".join(context_parts) if context_parts else "(no prior experiments)"


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
    # Fallback: use first 80 chars of first non-empty line
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

    # Clean up state results to remove error duplicates from prior runs
    if state.get("results"):
        state["results"] = [r for r in state["results"] if r.get('status') != 'error']

    # Initialize research log if needed
    if not RESEARCH_LOG.exists() or RESEARCH_LOG.stat().st_size == 0:
        with open(RESEARCH_LOG, 'w') as f:
            f.write("# OXI Research Log\n\n| Iter | Description | ao10 | Kept |\n|------|-------------|------|------|\n")

    # Run baseline if needed
    if state["baseline_ao10"] == 0.0:
        print("=== Running baseline training ===")
        git_revert()  # Ensure clean state
        baseline = run_training("baseline")
        state["baseline_ao10"] = baseline
        state["best_ao10"] = baseline
        if not any(r['iter'] == 0 for r in existing_results):
            with open(RESEARCH_LOG, 'a') as f:
                f.write(f"| 0 | Baseline (no changes) | {baseline:.6f} | baseline |\n")
            existing_results.append({'iter': 0, 'title': 'Baseline', 'ao10': baseline, 'status': 'baseline'})
        save_state(state)
        print(f"Baseline ao10: {baseline:.6f}")
    else:
        print(f"Resuming from iteration {state['iteration']}, best ao10: {state['best_ao10']:.6f}")

    best = state["best_ao10"]
    baseline = state["baseline_ao10"]
    start_iter = state["iteration"]
    all_results = [r for r in state.get("results", existing_results) if r.get('status') != 'error']
    end_iter = start_iter + MAX_ITERATIONS  # 50 iterations from wherever we resume

    for i in range(start_iter, end_iter):
        print(f"\n{'='*60}")
        print(f"=== Iteration {i} | {i - start_iter + 1}/{MAX_ITERATIONS} | best: {best:.6f} | baseline: {baseline:.6f} ===")
        print(f"{'='*60}")

        try:
            # Ensure clean state
            git_revert()

            # Read history and recent context
            history = RESEARCH_LOG.read_text()
            recent_context = get_recent_context(max_recent=5)

            # Build list of things NOT to try (already failed)
            failed_ideas = []
            for r in all_results:
                if r.get('status') in ('discarded', 'fail'):
                    failed_ideas.append(f"- Iter {r['iter']}: {r['title']} (ao10={r['ao10']:.6f})")
            failed_summary = "\n".join(failed_ideas[-15:]) if failed_ideas else "(none yet)"

            # Ask subagent to propose AND implement a change
            idea_prompt = f"""You are an ML researcher improving a chess move-prediction transformer (the "oxi" model).

## Goal
Improve the model's top-1 move prediction accuracy (measured as "ao10" — average of last 10 accuracy values during training).

## Current State
- Current best ao10: {best:.6f}
- Baseline ao10: {baseline:.6f}
- We are on iteration {i} of the research loop

## Past Experiments (FULL LOG)
{history}

## FAILED IDEAS — DO NOT REPEAT THESE:
{failed_summary}

## Recent Experiment Details
{recent_context}

## Architecture Summary
- Transformer with FiLM-conditioned RMSNorm, SwiGLU MLPs, SmolGen attention
- Factorized policy head (source/target projections, POLICY_RANK constant)
- Separate value tower with attention pooling
- Auxiliary heads: mobility, material, from-square, to-square, side-info
- Training uses Muon + AdamW optimizers with μP scaling
- Residual scaling: 1/sqrt(2*num_layers) per block

## Key Files (read these before making changes)
- src/model.rs — forward pass, loss computation, head architectures
- src/config.rs — hyperparameter defaults
- src/relative_position_transformer.rs — TransformerBlock, FiLMRmsNorm, MLP (SwiGLU)
- src/factorized_policy.rs — factorized policy head
- src/custom_training.rs — training loop, optimizer setup
- src/smolgen.rs — SmolGen attention mechanism

## CRITICAL RULES
1. DO NOT repeat any failed idea from the list above
2. Try something GENUINELY DIFFERENT from all prior experiments
3. Read the actual code first to find unexplored opportunities
4. Make EXACTLY ONE focused code change
5. The change must compile: `cargo check --features "train,backend-tch"`
6. Don't change CLI argument handling or data loading

## Ideas to Explore (NOVEL — not tried before)
- Learning rate adjustments (muon_base_lr, adamw_base_lr, embedding_base_lr)
- Weight decay tuning
- Warmup schedule changes
- Residual scaling formula changes
- Attention head count changes
- Pre-norm vs post-norm changes
- Skip connection modifications
- Dropout rates
- Embedding initialization changes
- Value head architecture changes
- From/to square head improvements
- Gradient clipping adjustments
- Optimizer hyperparameters (beta1, beta2)
- Layer-specific learning rate scaling
- Temperature scaling for policy logits
- Auxiliary loss weight adjustments (value, mobility, material weights individually)
- Number of attention heads
- SmolGen architecture (activation, compression ratio)

## REQUIRED: Structured Output
After making your change, you MUST end your response with a JSON block in exactly this format:

```json
{{"title": "<=10 word title of the change", "description": "1-2 sentence description of what was changed and why", "changes": "list of files changed and what was done in each"}}
```

This JSON block is parsed programmatically. Do not omit it."""

            print(f"  Requesting experiment idea from subagent...")
            idea_output = call_tool("subagent.run", task=idea_prompt).get("output", "")

            # Save full output
            iter_dir = AUTORESEARCH_DIR / str(i)
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "idea.md").write_text(idea_output)

            # Extract structured output instead of spawning a summarization subagent
            title, description, changes = extract_structured_output(idea_output)
            print(f"  Title: {title}")
            if description:
                print(f"  Description: {description}")

            # Compile check
            print(f"  Checking compilation...")
            compiles, stderr = compile_check()
            if not compiles:
                print(f"  ❌ Compile failed!")
                print(f"  {stderr[-500:]}")
                git_revert()
                result = {'iter': i, 'title': f'[COMPILE FAIL] {title}', 'ao10': 0.0, 'status': 'fail'}
                all_results.append(result)
                with open(RESEARCH_LOG, 'a') as f:
                    f.write(f"| {i} | [COMPILE FAIL] {title} | 0.000000 | ❌ |\n")
                state["iteration"] = i + 1
                save_state(state)
                continue

            print(f"  ✅ Compilation succeeded")

            # Train
            ao10 = run_training(f"run_{i}")

            # Evaluate and keep/discard
            if ao10 > best:
                print(f"  ✅ IMPROVEMENT: {ao10:.6f} > {best:.6f} (+{ao10-best:.6f})")
                best = ao10
                git_commit(f"research iter {i}: {title} (ao10={ao10:.6f})")
                status = "kept"
                status_emoji = "✅"
            else:
                print(f"  ❌ No improvement: {ao10:.6f} <= {best:.6f}")
                git_revert()
                status = "discarded"
                status_emoji = "❌"

            result = {'iter': i, 'title': title, 'ao10': ao10, 'status': status}
            all_results.append(result)
            with open(RESEARCH_LOG, 'a') as f:
                f.write(f"| {i} | {title} | {ao10:.6f} | {status_emoji} |\n")

            # Update state
            state["iteration"] = i + 1
            state["best_ao10"] = best
            state["results"] = all_results
            save_state(state)

            # Generate chart every iteration
            generate_chart(all_results, baseline)

        except Exception as e:
            print(f"  ⚠️ Error in iteration {i}: {e}")
            traceback.print_exc()
            git_revert()
            result = {'iter': i, 'title': f'[ERROR] {str(e)[:50]}', 'ao10': 0.0, 'status': 'error'}
            all_results.append(result)
            with open(RESEARCH_LOG, 'a') as f:
                f.write(f"| {i} | [ERROR] {str(e)[:50]} | 0.000000 | ⚠️ |\n")
            state["iteration"] = i + 1
            save_state(state)

    # Final summary
    print(f"\n{'='*60}")
    print(f"=== RESEARCH COMPLETE ===")
    print(f"  Baseline ao10: {baseline:.6f}")
    print(f"  Best ao10:     {best:.6f}")
    print(f"  Improvement:   {best - baseline:+.6f}")
    kept = [r for r in all_results if r['status'] == 'kept']
    print(f"  Kept changes:  {len(kept)} / {len([r for r in all_results if r.get('status') != 'baseline'])}")
    print(f"  Log: {RESEARCH_LOG}")

    # Final chart
    generate_chart(all_results, baseline)

    # Upload final log
    try:
        call_tool("artifact_upload", path=str(RESEARCH_LOG), kind="file",
                  description="Final research log with all iterations")
    except Exception:
        print("  (artifact_upload not available)")


if __name__ == "__main__":
    main()