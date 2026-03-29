"""
Automated ML research loop for oxi chess model.
Runs 20 iterations: each proposes a code change via subagent, trains, evaluates,
keeps improvements and reverts failures.
"""
from shadesmar_tools import subagent, call_tool
import subprocess
import os
import time
import json
import traceback
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
WORKSPACE = Path("/Users/marcusbuffett/projects/chessbook/oxi")
RESEARCH_LOG = WORKSPACE / "research_log.md"
AUTORESEARCH_DIR = WORKSPACE / "autoresearch"
RESEARCH_RUNS = WORKSPACE / "research_runs"
PROGRESS_CHART = RESEARCH_RUNS / "progress.png"
MAX_ITERATIONS = 20
TRAIN_TIMEOUT = 300  # 5 minutes per training run
STATE_FILE = WORKSPACE / ".research_state.json"

# ── Helpers ────────────────────────────────────────────────────────────────

def run_git(args, check=False):
    """Run a git command in the workspace."""
    return subprocess.run(
        ["git"] + args,
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        timeout=30,
        check=check,
    )

def git_checkout_clean():
    """Revert all uncommitted changes."""
    run_git(["checkout", "."])
    run_git(["clean", "-fd", "--", "src/"])

def run_training(run_name):
    """Run training for TRAIN_TIMEOUT seconds and parse ao10 accuracy."""
    log_dir = RESEARCH_RUNS / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "cargo", "run", "--release",
        "--features", "backend-tch train",
        "--", "train",
        "--pretrain-samples=0",
        f"--data-path=../data",
        "--physical-batch-size=512",
        "--num-layers=6",
        "--embed-dim=192",
        "--disable-tui",
        f"--log-dir={log_dir}",
    ]
    
    print(f"  Training: {' '.join(cmd)}")
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=TRAIN_TIMEOUT,
        )
        elapsed = time.time() - start
        print(f"  Training finished in {elapsed:.0f}s (exit code {result.returncode})")
        if result.returncode != 0:
            # Print last few lines of stderr for debugging
            stderr_lines = result.stderr.strip().split('\n')[-10:]
            print(f"  STDERR (last lines): {'  '.join(stderr_lines)}")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  Training killed after {elapsed:.0f}s timeout (expected)")
    
    # Parse accuracy
    accuracy = parse_ao10(log_dir)
    return accuracy

def parse_ao10(log_dir):
    """Parse ao10 from top1_accuracy.log."""
    accuracy_log = log_dir / "metrics_logs" / "top1_accuracy.log"
    if not accuracy_log.exists():
        print(f"  WARNING: {accuracy_log} does not exist")
        return 0.0
    
    with open(accuracy_log, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print(f"  WARNING: {accuracy_log} is empty")
        return 0.0
    
    recent_lines = lines[-10:] if len(lines) >= 10 else lines
    accuracies = []
    for line in recent_lines:
        line = line.strip()
        if '\t' in line:
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    accuracies.append(float(parts[1]))
                except ValueError:
                    pass
    
    if not accuracies:
        print(f"  WARNING: No valid accuracy values found in {accuracy_log}")
        return 0.0
    
    ao10 = sum(accuracies) / len(accuracies)
    print(f"  ao10 = {ao10:.6f} (from {len(accuracies)} values, {len(lines)} total lines)")
    return ao10

def init_research_log():
    """Initialize research_log.md with header."""
    header = """# OXI Research Log

| Iter | Description | ao10 | Kept |
|------|-------------|------|------|
"""
    with open(RESEARCH_LOG, 'w') as f:
        f.write(header)

def append_to_log(iteration, title, accuracy, kept):
    """Append a row to research_log.md."""
    kept_str = "✅" if kept else "❌"
    row = f"| {iteration} | {title} | {accuracy:.6f} | {kept_str} |\n"
    with open(RESEARCH_LOG, 'a') as f:
        f.write(row)

def generate_chart(results):
    """Generate progress chart with matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if not results:
        return
    
    RESEARCH_RUNS.mkdir(parents=True, exist_ok=True)
    
    iters = [r['iter'] for r in results]
    accs = [r['accuracy'] for r in results]
    kept = [r['kept'] for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Baseline line
    baseline = results[0]['accuracy'] if results else 0
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.6f})')
    
    # Best-so-far line
    best_so_far = []
    current_best = 0
    for r in results:
        if r['kept']:
            current_best = max(current_best, r['accuracy'])
        best_so_far.append(current_best)
    ax.plot(iters, best_so_far, color='blue', alpha=0.3, linewidth=1, label='Best so far')
    
    # Scatter: green=kept, red=discarded
    for i, r in enumerate(results):
        color = 'green' if r['kept'] else 'red'
        marker = 'o' if r['kept'] else 'x'
        ax.scatter(r['iter'], r['accuracy'], color=color, marker=marker, s=80, zorder=5)
    
    # Legend proxies
    ax.scatter([], [], color='green', marker='o', s=80, label='Kept')
    ax.scatter([], [], color='red', marker='x', s=80, label='Discarded')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('ao10 Accuracy')
    ax.set_title('OXI Research Loop — ao10 Accuracy per Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(PROGRESS_CHART), dpi=150)
    plt.close()
    
    try:
        call_tool("artifact_upload", path=str(PROGRESS_CHART), kind="image", description="Research loop progress chart — ao10 accuracy")
    except Exception as e:
        print(f"  Warning: Failed to upload chart artifact: {e}")

def load_state():
    """Load checkpoint state if it exists."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.loads(f.read())
    return None

def save_state(state):
    """Save checkpoint state."""
    with open(STATE_FILE, 'w') as f:
        f.write(json.dumps(state, indent=2))

def compile_check():
    """Quick compile check to validate code changes."""
    result = subprocess.run(
        ["cargo", "check", "--features", "train,backend-tch"],
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        timeout=180,
    )
    return result.returncode == 0, result.stderr

# ── Main Loop ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OXI Automated ML Research Loop")
    print("=" * 60)
    
    # Setup directories
    AUTORESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_RUNS.mkdir(parents=True, exist_ok=True)
    
    # Try to resume from checkpoint
    state = load_state()
    if state and state.get("results"):
        results = state["results"]
        best = state["best"]
        start_iter = state["next_iter"]
        print(f"\nResuming from iteration {start_iter}, best so far: {best:.6f}")
        print(f"Previous results: {len(results)} iterations completed")
    else:
        results = []
        best = 0.0
        start_iter = 0
        
        # Initialize fresh log
        init_research_log()
        
        # ── Step 0: Baseline ──
        print("\n── Baseline Training ──")
        git_checkout_clean()
        baseline_acc = run_training("run_baseline")
        
        if baseline_acc <= 0.0:
            print("ERROR: Baseline training produced no accuracy. Checking if log exists...")
            # Try to see what happened
            log_dir = RESEARCH_RUNS / "run_baseline"
            for p in log_dir.rglob("*"):
                print(f"  Found: {p}")
            print("Continuing anyway with baseline=0...")
        
        best = baseline_acc
        results.append({
            'iter': 0,
            'title': 'Baseline (no changes)',
            'accuracy': baseline_acc,
            'kept': True,
        })
        append_to_log(0, "Baseline (no changes)", baseline_acc, True)
        run_git(["add", "-A"])
        run_git(["commit", "-m", "research: baseline run"])
        
        start_iter = 1
        save_state({"results": results, "best": best, "next_iter": start_iter})
        generate_chart(results)
        print(f"\nBaseline ao10: {baseline_acc:.6f}")
    
    # ── Iterations 1..20 ──
    for i in range(start_iter, MAX_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"Iteration {i}/{MAX_ITERATIONS}")
        print(f"Current best: {best:.6f}")
        print(f"{'=' * 60}")
        
        try:
            # ── Step A: Read history for context ──
            with open(RESEARCH_LOG, 'r') as f:
                history = f.read()
            
            # ── Step B: Ask subagent for an idea + implementation ──
            # Build context about recent autoresearch files
            recent_iters_hint = ""
            existing_iters = sorted([
                int(d.name) for d in AUTORESEARCH_DIR.iterdir()
                if d.is_dir() and d.name.isdigit()
            ])
            if existing_iters:
                recent = existing_iters[-5:]  # Last 5
                recent_iters_hint = f"""
Recent iteration directories to read for context: {', '.join(f'autoresearch/{x}/idea.md' for x in recent)}
Read these to understand what was tried, what worked, and what didn't.
"""
            
            idea_prompt = f"""You are improving the oxi chess model at {WORKSPACE}.
Your goal is to propose and implement ONE concrete code change to improve the ao10 (average of last 10) move prediction accuracy.

First, read ./research_log.md for a summary of all past iterations and their results:

{history}

{recent_iters_hint}
Then read the ./autoresearch/ directory — each subfolder (e.g. autoresearch/1/idea.md)
contains the full reasoning and implementation details of a past iteration. Read the
most recent 3-5 idea.md files to understand what was tried, what worked, and what didn't.
Use this context to avoid repeating failed approaches and to build on successful ones.

Look at relevant code before making changes. Read the files you want to modify first.
Key files:
- src/model.rs — model architecture (OXIModel)
- src/config.rs — training configuration and defaults
- src/relative_position_transformer.rs — transformer block
- src/smolgen.rs — Smolgen dynamic positional attention
- src/factorized_policy.rs — factorized policy head
- src/custom_training.rs — training loop
- src/encoding.rs — board encoding
- src/dataset.rs — data loading

Make exactly one focused, testable change. The change should be something that could plausibly
improve move prediction accuracy within 5 minutes of training. Good ideas include:
- Adjusting loss weights, learning rates, or regularization
- Modifying activation functions or normalization
- Changing the architecture (adding/removing layers, changing dimensions)
- Improving the policy head or attention mechanism
- Adjusting label smoothing, focal loss, or other training tricks

IMPORTANT CONSTRAINTS:
- Do NOT change the --embed-dim, --num-layers, or --physical-batch-size CLI arguments (those are fixed by the training command)
- Do NOT add new CLI arguments (the training command is fixed)
- Focus on code changes within the existing configuration framework
- Make sure your changes compile with `cargo check --features "train,backend-tch"`
- Do NOT run training yourself — just make the code changes
- Do NOT modify the research_log.md or autoresearch files

Current best ao10 accuracy: {best:.6f}
This is iteration {i} of {MAX_ITERATIONS}.
If your change improves ao10 accuracy, it will be committed. Otherwise reverted with git checkout.

Describe what you changed and why at the end of your response."""

            print("  Asking subagent for idea...")
            idea_output = subagent.run(task=idea_prompt, timeout_secs=1200)
            
            # ── Step C: Save full subagent output ──
            iter_dir = AUTORESEARCH_DIR / str(i)
            iter_dir.mkdir(parents=True, exist_ok=True)
            idea_file = iter_dir / "idea.md"
            with open(idea_file, 'w') as f:
                f.write(idea_output)
            
            # ── Step D: Generate title ──
            print("  Generating title...")
            title = subagent.run(
                task=f"Summarize this ML experiment idea in 10 words or fewer. Output only the summary, no quotes or formatting.\n\n{idea_output[:3000]}",
                model="fast",
                timeout_secs=30,
            )
            title = title.strip().replace('\n', ' ')[:80]
            print(f"  Title: {title}")
            
            # ── Step E: Compile check ──
            print("  Checking compilation...")
            compiles, stderr = compile_check()
            if not compiles:
                print(f"  COMPILE FAILED — reverting changes")
                error_lines = stderr.strip().split('\n')[-15:]
                print(f"  Error: {'  '.join(error_lines)}")
                git_checkout_clean()
                results.append({
                    'iter': i,
                    'title': f"[COMPILE FAIL] {title}",
                    'accuracy': 0.0,
                    'kept': False,
                })
                append_to_log(i, f"[COMPILE FAIL] {title}", 0.0, False)
                save_state({"results": results, "best": best, "next_iter": i + 1})
                generate_chart(results)
                continue
            
            # ── Step F: Train ──
            print("  Running training...")
            run_name = f"run_{i}"
            accuracy = run_training(run_name)
            
            if accuracy <= 0.0:
                print(f"  TRAIN FAIL: accuracy is 0, reverting")
                git_checkout_clean()
                results.append({
                    'iter': i,
                    'title': f"[TRAIN FAIL] {title}",
                    'accuracy': 0.0,
                    'kept': False,
                })
                append_to_log(i, f"[TRAIN FAIL] {title}", 0.0, False)
                save_state({"results": results, "best": best, "next_iter": i + 1})
                generate_chart(results)
                continue
            
            # ── Step G: Keep or discard ──
            kept = accuracy > best
            if kept:
                print(f"  ✅ IMPROVEMENT: {accuracy:.6f} > {best:.6f} — keeping!")
                best = accuracy
                run_git(["add", "-A"])
                run_git(["commit", "-m", f"research iter {i}: {title} (ao10={accuracy:.6f})"])
            else:
                print(f"  ❌ No improvement: {accuracy:.6f} <= {best:.6f} — reverting")
                git_checkout_clean()
            
            results.append({
                'iter': i,
                'title': title,
                'accuracy': accuracy,
                'kept': kept,
            })
            append_to_log(i, title, accuracy, kept)
            
            # ── Step H: Chart + checkpoint ──
            save_state({"results": results, "best": best, "next_iter": i + 1})
            generate_chart(results)
            
        except Exception as e:
            print(f"  ERROR in iteration {i}: {e}")
            traceback.print_exc()
            git_checkout_clean()
            
            error_msg = str(e)[:60].replace('\n', ' ')
            results.append({
                'iter': i,
                'title': f"[ERROR] {error_msg}",
                'accuracy': 0.0,
                'kept': False,
            })
            append_to_log(i, f"[ERROR] {error_msg}", 0.0, False)
            save_state({"results": results, "best": best, "next_iter": i + 1})
            generate_chart(results)
            continue
    
    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\nTotal iterations: {len(results)}")
    print(f"Best ao10: {best:.6f}")
    
    kept_results = [r for r in results if r['kept'] and r['iter'] > 0]
    print(f"Improvements found: {len(kept_results)}")
    
    if kept_results:
        print("\nKept iterations:")
        for r in kept_results:
            print(f"  Iter {r['iter']}: {r['title']} (ao10={r['accuracy']:.6f})")
    
    print(f"\nResearch log: {RESEARCH_LOG}")
    print(f"Progress chart: {PROGRESS_CHART}")
    
    # Final chart upload
    generate_chart(results)

if __name__ == "__main__":
    main()
