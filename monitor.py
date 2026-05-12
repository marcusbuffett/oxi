#!/usr/bin/env python
"""
Live TUI chart monitor for oxi training runs.
Usage: python monitor.py [run_dir] [--window N] [--interval S]

Install deps: pip install plotext
"""

import os
import re
import sys
import time
import select
import termios
import argparse
import numpy as np
from pathlib import Path

try:
    import plotext as plt
except ImportError:
    print("Missing dependency: pip install plotext")
    sys.exit(1)


METRICS = [
    ("top1_accuracy",           "Top-1 Accuracy"),
    ("wdl_accuracy",            "WDL Accuracy"),
    ("aux_from_square_accuracy","From-Sq Accuracy"),
    ("aux_to_square_accuracy",  "To-Sq Accuracy"),
    ("cp_loss_calibration_overall", "CPL Calibration"),
]

WEIGHT_TOP1 = 1.0
WEIGHT_WDL  = 0.5
WEIGHT_AUX  = 0.2
WEIGHT_CALIBRATION = 0.4

TRAINING_TIMEOUT = 3600  # seconds — assumed duration of a completed comparison run

# Must match research_loop.ts scoring / acceptance criterion
CALIBRATION_MIN_LABELED_FRACTION = 0.01
MIN_COHEN_D = 0.3


def steps_to_times(steps, total_steps, duration_secs):
    """Convert a list of step values to elapsed times in seconds via linear interpolation.

    Uses the index within *steps* (not the step value itself) so sparse or
    non-zero-based step sequences are handled correctly.
    """
    if total_steps <= 1:
        return [0.0] * len(steps)
    return [(i / (total_steps - 1)) * duration_secs for i in range(len(steps))]


def best_score_from_log(base: Path) -> dict:
    """Parse research_log.md and return the highest value seen for each tracked metric."""
    log_path = base.parent / "research_log.md"
    bests = {"composite": 0.0, "top1": 0.0, "wdl": 0.0, "aux": 0.0, "calibration": 0.0}
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"\|\s*([\d.]+)\s*\(top1=([\d.]+),\s*wdl=([\d.]+),\s*aux=([\d.]+)(?:,\s*cal=([\d.]+))?", line)
                if m:
                    try:
                        bests["composite"] = max(bests["composite"], float(m.group(1)))
                        bests["top1"]      = max(bests["top1"],      float(m.group(2)))
                        bests["wdl"]       = max(bests["wdl"],       float(m.group(3)))
                        bests["aux"]       = max(bests["aux"],        float(m.group(4)))
                        if m.group(5) is not None:
                            bests["calibration"] = max(bests["calibration"], float(m.group(5)))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return bests


def ao100_composite(metrics_dir: Path) -> float:
    """Compute ao100 composite score for a run directory."""
    def ao100(name):
        pts = read_log(metrics_dir / f"{name}.log")
        vals = [v for _, v in pts[-100:]]
        return sum(vals) / len(vals) if vals else 0.0
    top1 = ao100("top1_accuracy")
    wdl  = normalize_wdl(ao100("wdl_accuracy"))
    aux  = (ao100("aux_from_square_accuracy") + ao100("aux_to_square_accuracy")) / 2.0
    calibration = ao100("cp_loss_calibration_overall")
    labeled_fraction = ao100("cp_loss_labeled_fraction")
    if labeled_fraction < CALIBRATION_MIN_LABELED_FRACTION:
        calibration = 0.0
    return (
        WEIGHT_TOP1 * top1
        + WEIGHT_WDL * wdl
        + WEIGHT_AUX * aux
        + WEIGHT_CALIBRATION * calibration
    )


def find_best_comparison_run(base: Path, current_run_dir: Path):
    """Return the run dir (excluding current) with the highest ao100 composite score."""
    runs = [p for p in base.iterdir() if p.is_dir() and (p / "metrics_logs").is_dir()
            and p.resolve() != current_run_dir.resolve()]
    if not runs:
        return None
    return max(runs, key=lambda p: ao100_composite(p / "metrics_logs"))


def find_latest_run(base: Path):
    runs = [p for p in base.iterdir() if p.is_dir() and (p / "metrics_logs").is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def read_log(path: Path) -> list[tuple[int, float]]:
    points = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 2:
                    try:
                        points.append((int(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return points


def smooth(values: list[float], k: int = 5) -> list[float]:
    if k <= 1 or not values:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - k + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def downsample_avg(times: list[float], values: list[float], max_points: int) -> tuple[list[float], list[float]]:
    """Downsample by averaging into fixed-width bins. Stable across renders."""
    if len(values) <= max_points or max_points < 2:
        return times, values
    t_min, t_max = times[0], times[-1]
    if t_max == t_min:
        return times, values
    bin_width = (t_max - t_min) / max_points
    out_t, out_v = [], []
    bin_start = t_min
    bin_sum, bin_tsum, bin_count = 0.0, 0.0, 0
    for t, v in zip(times, values):
        if t >= bin_start + bin_width and bin_count > 0:
            out_t.append(bin_tsum / bin_count)
            out_v.append(bin_sum / bin_count)
            bin_sum, bin_tsum, bin_count = 0.0, 0.0, 0
            bin_start = t
        bin_sum += v
        bin_tsum += t
        bin_count += 1
    if bin_count > 0:
        out_t.append(bin_tsum / bin_count)
        out_v.append(bin_sum / bin_count)
    return out_t, out_v


def normalize_wdl(value: float) -> float:
    """Normalize WDL values to 0-1 range, handling old percentage-scale logs."""
    return value / 100.0 if value > 1.0 else value


def compute_composite(metrics_dir: Path):
    """Return (steps, composite_values) aligned to the shortest common step range."""
    def load(name):
        pts = read_log(metrics_dir / f"{name}.log")
        return {s: v for s, v in pts}

    top1_map  = load("top1_accuracy")
    wdl_map   = load("wdl_accuracy")
    from_map  = load("aux_from_square_accuracy")
    to_map    = load("aux_to_square_accuracy")
    calibration_map = load("cp_loss_calibration_overall")
    labeled_fraction_map = load("cp_loss_labeled_fraction")

    common = sorted(set(top1_map) & set(wdl_map) & set(from_map) & set(to_map))
    if not common:
        return [], []

    steps, values = [], []
    for s in common:
        aux = (from_map[s] + to_map[s]) / 2.0
        calibration = calibration_map.get(s, 0.0)
        labeled_fraction = labeled_fraction_map.get(s, 0.0)
        if labeled_fraction < CALIBRATION_MIN_LABELED_FRACTION:
            calibration = 0.0
        score = (
            WEIGHT_TOP1 * top1_map[s]
            + WEIGHT_WDL * normalize_wdl(wdl_map[s])
            + WEIGHT_AUX * aux
            + WEIGHT_CALIBRATION * calibration
        )
        steps.append(s)
        values.append(score)
    return steps, values


def composite_summary(metrics_dir: Path, limit_points: int | None = None) -> dict:
    """Return research-loop-compatible ao100 composite summary."""
    steps, values = compute_composite(metrics_dir)
    if limit_points is not None:
        steps = steps[:limit_points]
        values = values[:limit_points]
    if not values:
        return {"score": 0.0, "n": 0, "last_step": None}
    window = values[-100:]
    return {
        "score": sum(window) / len(window),
        "n": len(values),
        "last_step": steps[-1] if steps else None,
    }


def render(run_dir: Path, window: int, smooth_k: int, best_ever: dict) -> None:
    metrics_dir = run_dir / "metrics_logs"

    # ------------------------------------------------------------------
    # Compute current run duration (wall-clock seconds since run started)
    # ------------------------------------------------------------------
    current_duration = TRAINING_TIMEOUT  # fallback
    elapsed = ""
    run_start = None
    try:
        first_log = next(metrics_dir.glob("*.log"), None)
        if first_log:
            stat = first_log.stat()
            run_start = getattr(stat, "st_birthtime", stat.st_ctime)
            current_duration = time.time() - run_start
            secs = int(current_duration)
            h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
            elapsed = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
    except OSError:
        pass

    # ------------------------------------------------------------------
    # Load current-run datasets; convert steps → times; apply window
    # ------------------------------------------------------------------
    datasets = []           # (metric_name, times, values, last_step)
    for metric_name, _ in METRICS:
        points = read_log(metrics_dir / f"{metric_name}.log")
        if points:
            all_steps  = [p[0] for p in points]
            all_values = [p[1] for p in points]
            if metric_name == "wdl_accuracy":
                all_values = [normalize_wdl(v) for v in all_values]
            all_times  = steps_to_times(all_steps, len(all_steps), current_duration)
            if window > 0:
                all_times  = all_times[-window:]
                all_values = all_values[-window:]
                all_steps  = all_steps[-window:]
            datasets.append((metric_name, all_times, all_values, all_steps[-1] if all_steps else 0))

    comp_steps_full, comp_values_full = compute_composite(metrics_dir)
    has_composite = bool(comp_steps_full)

    comp_times_full = steps_to_times(comp_steps_full, len(comp_steps_full), current_duration)
    if window > 0:
        comp_times  = comp_times_full[-window:]
        comp_values = comp_values_full[-window:]
        comp_steps  = comp_steps_full[-window:]
    else:
        comp_times  = comp_times_full
        comp_values = comp_values_full
        comp_steps  = comp_steps_full

    # aux = average of from/to square accuracy (current run)
    aux_times, aux_values = [], []
    if has_composite:
        from_map = {s: v for s, v in read_log(metrics_dir / "aux_from_square_accuracy.log")}
        to_map   = {s: v for s, v in read_log(metrics_dir / "aux_to_square_accuracy.log")}
        common_aux = sorted(set(from_map) & set(to_map))
        all_aux_values = [(from_map[s] + to_map[s]) / 2.0 for s in common_aux]
        all_aux_times  = steps_to_times(common_aux, len(common_aux), current_duration)
        if window > 0:
            aux_times  = all_aux_times[-window:]
            aux_values = all_aux_values[-window:]
        else:
            aux_times  = all_aux_times
            aux_values = all_aux_values
    has_aux = bool(aux_times)

    if not datasets and not has_composite:
        print(f"No data found in {metrics_dir}")
        return

    # ------------------------------------------------------------------
    # Determine time window bounds for comparison-run clipping
    # ------------------------------------------------------------------
    if datasets:
        min_time = datasets[0][1][0] if datasets[0][1] else 0.0
        max_time = datasets[0][1][-1] if datasets[0][1] else current_duration
    elif comp_times:
        min_time = comp_times[0]
        max_time = comp_times[-1]
    else:
        min_time, max_time = 0.0, current_duration

    # aux + composite occupy the last two slots
    n = len(datasets) + (1 if has_aux else 0) + (1 if has_composite else 0)
    cols = 2
    rows = (n + cols - 1) // cols

    try:
        term_w = os.get_terminal_size().columns
        term_h = os.get_terminal_size().lines - 4
    except OSError:
        term_w, term_h = 200, 50

    plt.clf()
    plt.plotsize(term_w, term_h)
    plt.subplots(rows, cols)
    plt.theme("dark")

    label_map = {m: label for m, label in METRICS}

    # ------------------------------------------------------------------
    # Load comparison run data (x-axis = times, clipped to [min_time, max_time])
    # ------------------------------------------------------------------
    cmp_base = Path(__file__).parent / "research_runs"
    cmp_dir = find_best_comparison_run(cmp_base, run_dir)
    cmp_metrics = cmp_dir / "metrics_logs" if cmp_dir else None

    def load_cmp(name, min_t, max_t):
        """Return (times, values) for comparison run, clipped to [min_t, max_t]."""
        if not cmp_metrics:
            return [], []
        pts = read_log(cmp_metrics / f"{name}.log")
        if not pts:
            return [], []
        all_steps  = [p[0] for p in pts]
        all_values = [p[1] for p in pts]
        if name == "wdl_accuracy":
            all_values = [normalize_wdl(v) for v in all_values]
        all_times  = steps_to_times(all_steps, len(all_steps), TRAINING_TIMEOUT)
        paired = [(t, v) for t, v in zip(all_times, all_values) if min_t <= t <= max_t]
        if not paired:
            return [], []
        return [p[0] for p in paired], [p[1] for p in paired]

    # Approximate plot width per subplot (terminal chars minus axis/padding)
    plot_width = max(40, term_w // cols - 10)

    def setup_subplot(times, values, title, line_color, cmp_times=None, cmp_values=None,
                      threshold=0.0, improvement_symbol="▲"):
        plt.ticks_color("white")
        plt.xfrequency(0)
        ao100 = sum(values[-100:]) / min(len(values), 100) if values else 0
        cmp_ao100 = sum(cmp_values[-100:]) / min(len(cmp_values), 100) if cmp_values else 0
        beat_val = threshold if threshold > 0 else cmp_ao100
        improvement = beat_val > 0 and ao100 > beat_val
        marker_str = f" {improvement_symbol}" if improvement else ""
        plt.title(f"{title}  ao100={ao100:.4f}{marker_str}")
        smoothed = smooth(values, smooth_k)
        times, smoothed = downsample_avg(times, smoothed, plot_width)
        cmp_smoothed = smooth(cmp_values, smooth_k) if cmp_values else []
        if cmp_times:
            cmp_times, cmp_smoothed = downsample_avg(cmp_times, cmp_smoothed, plot_width)
        all_vals = smoothed + cmp_smoothed
        lo, hi = min(all_vals), max(all_vals)
        if lo != hi:
            step_size = (hi - lo) / 4
            ticks = [lo + step_size * i for i in range(5)]
            plt.yticks(ticks, [f"{v:.2f}" for v in ticks])
        if cmp_times:
            plt.plot(cmp_times, cmp_smoothed, color="red+", marker="braille")
        plt.plot(times, smoothed, color=line_color, marker="braille")

    for i, (metric_name, times, values, _last_step) in enumerate(datasets):
        row = i // cols + 1
        col = i % cols + 1
        plt.subplot(row, col)
        ct, cv = load_cmp(metric_name, min_time, max_time)
        setup_subplot(times, values, label_map.get(metric_name, metric_name), "cyan+",
                      cmp_times=ct, cmp_values=cv)

    extra = len(datasets)
    if has_aux:
        row = extra // cols + 1
        col = extra % cols + 1
        plt.subplot(row, col)
        cf_t, cf_v = load_cmp("aux_from_square_accuracy", min_time, max_time)
        ct_t, ct_v = load_cmp("aux_to_square_accuracy", min_time, max_time)
        # Intersect comparison aux on time (same times after independent clip)
        cf_map_t = dict(zip(cf_t, cf_v))
        ct_map_t = dict(zip(ct_t, ct_v))
        cmp_aux_times  = sorted(set(cf_t) & set(ct_t))
        cmp_aux_values = [(cf_map_t[t] + ct_map_t[t]) / 2.0 for t in cmp_aux_times]
        setup_subplot(aux_times, aux_values, "Aux Accuracy", "cyan+",
                      cmp_times=cmp_aux_times, cmp_values=cmp_aux_values)
        extra += 1

    if has_composite:
        row = extra // cols + 1
        col = extra % cols + 1
        plt.subplot(row, col)
        cmp_cs_full, cmp_cv_full = compute_composite(cmp_metrics) if cmp_metrics else ([], [])
        if cmp_cs_full:
            cmp_ct_full = steps_to_times(cmp_cs_full, len(cmp_cs_full), TRAINING_TIMEOUT)
            paired = [(t, v) for t, v in zip(cmp_ct_full, cmp_cv_full) if min_time <= t <= max_time]
            cmp_ct = [p[0] for p in paired]
            cmp_cv = [p[1] for p in paired]
        else:
            cmp_ct, cmp_cv = [], []
        # Target: score the current run needs to exceed to achieve Cohen's d >= MIN_COHEN_D.
        # Uses the last-100 values of the comparison run (same window as acceptance criterion).
        if len(cmp_cv_full) >= 10:
            cmp_arr = np.array(cmp_cv_full[-100:])
            threshold = float(cmp_arr.mean() + MIN_COHEN_D * cmp_arr.std(ddof=1))
        else:
            threshold = best_ever["composite"] * (1 + MIN_COHEN_D * 0.01)
        setup_subplot(comp_times, comp_values, "Composite Score", "green+",
                      cmp_times=cmp_ct, cmp_values=cmp_cv,
                      threshold=threshold, improvement_symbol="✓")

    plt.show()

    all_steps = comp_steps or (datasets[0][3:4] if datasets else [])
    # Keep step number in status line for reference
    last_step = comp_steps[-1] if comp_steps else (datasets[0][3] if datasets else None)
    step_label = f"step {last_step}" if last_step is not None else ""
    window_label = f"last {window}" if window > 0 else "all"
    print(f"  run: {run_dir.name}  |  {step_label}  |  elapsed: {elapsed}  |  [{window_label}] tab to toggle  |  ctrl-c to quit", end="", flush=True)


def report(run_dir: Path, cmp_dir: Path | None) -> None:
    """Print a one-shot text comparison of current run vs comparison run at matched wall-clock times."""
    metrics_dir = run_dir / "metrics_logs"

    # Current run duration
    first_log = next(metrics_dir.glob("*.log"), None)
    if not first_log:
        print(f"No metrics in {metrics_dir}")
        return
    stat = first_log.stat()
    run_start = getattr(stat, "st_birthtime", stat.st_ctime)
    current_duration = time.time() - run_start
    secs = int(current_duration)
    elapsed = f"{secs // 3600}h{(secs % 3600) // 60:02d}m{secs % 60:02d}s" if secs >= 3600 else f"{secs // 60}m{secs % 60:02d}s"

    cmp_metrics = cmp_dir / "metrics_logs" if cmp_dir else None

    print(f"\n{'='*70}")
    print(f"  Run: {run_dir.name}  |  Elapsed: {elapsed}  |  Compare: {cmp_dir.name if cmp_dir else 'none'}")
    print(f"{'='*70}\n")

    # For each metric + composite, show current ao100 vs comparison ao100 at same wall-clock time
    all_metrics = [("top1_accuracy", "Top-1 Accuracy"), ("wdl_accuracy", "WDL Accuracy"),
                   ("aux_from_square_accuracy", "From-Sq Acc"), ("aux_to_square_accuracy", "To-Sq Acc"),
                   ("cp_loss_calibration_overall", "CPL Calibration")]

    for metric_name, label in all_metrics:
        # Current run
        pts = read_log(metrics_dir / f"{metric_name}.log")
        if not pts:
            print(f"  {label:20s}  no data")
            continue
        vals = [v for _, v in pts]
        if metric_name == "wdl_accuracy":
            vals = [normalize_wdl(v) for v in vals]
        ao100 = sum(vals[-100:]) / min(len(vals), 100)

        # Comparison at same elapsed time
        cmp_ao100 = None
        if cmp_metrics:
            cmp_pts = read_log(cmp_metrics / f"{metric_name}.log")
            if cmp_pts:
                cmp_times = steps_to_times([s for s, _ in cmp_pts], len(cmp_pts), TRAINING_TIMEOUT)
                # Find values up to current_duration
                cmp_vals_at_time = [v for t, (_, v) in zip(cmp_times, cmp_pts) if t <= current_duration]
                if metric_name == "wdl_accuracy":
                    cmp_vals_at_time = [normalize_wdl(v) for v in cmp_vals_at_time]
                if cmp_vals_at_time:
                    cmp_ao100 = sum(cmp_vals_at_time[-100:]) / min(len(cmp_vals_at_time), 100)

        if cmp_ao100 is not None:
            delta = ao100 - cmp_ao100
            arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
            print(f"  {label:20s}  {ao100:8.4f}  vs  {cmp_ao100:8.4f}  ({delta:+.4f} {arrow})")
        else:
            print(f"  {label:20s}  {ao100:8.4f}")

    # Composite
    comp_summary = composite_summary(metrics_dir)
    comp_ao100 = comp_summary["score"]
    cmp_comp_ao100 = None
    cmp_same_n = None
    if cmp_metrics:
        cmp_same_n = composite_summary(cmp_metrics, comp_summary["n"])
        cmp_cs, cmp_cv = compute_composite(cmp_metrics)
        if cmp_cv:
            cmp_times = steps_to_times(cmp_cs, len(cmp_cs), TRAINING_TIMEOUT)
            cmp_cv_at_time = [v for t, v in zip(cmp_times, cmp_cv) if t <= current_duration]
            if cmp_cv_at_time:
                cmp_comp_ao100 = sum(cmp_cv_at_time[-100:]) / min(len(cmp_cv_at_time), 100)

    if cmp_comp_ao100 is not None:
        delta = comp_ao100 - cmp_comp_ao100
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
        print(f"\n  {'COMPOSITE':20s}  {comp_ao100:8.4f}  vs  {cmp_comp_ao100:8.4f}  ({delta:+.4f} {arrow})")
    else:
        print(f"\n  {'COMPOSITE':20s}  {comp_ao100:8.4f}")

    if cmp_same_n and cmp_same_n["n"] > 0:
        delta = comp_ao100 - cmp_same_n["score"]
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
        print(
            f"  {'same-N composite':20s}  {comp_ao100:8.4f}  vs  {cmp_same_n['score']:8.4f}  "
            f"({delta:+.4f} {arrow}, n={comp_summary['n']})"
        )

    # Throughput
    pts = read_log(metrics_dir / "top1_accuracy.log")
    if pts:
        total_steps = pts[-1][0]
        # Infer batch size from train.log
        train_log = run_dir / "train.log"
        batch_size = 1024  # fallback
        try:
            with open(train_log) as f:
                first_line = f.readline()
                m = re.search(r'physical_batch_size: (\d+)', first_line)
                if m:
                    batch_size = int(m.group(1))
        except (FileNotFoundError, ValueError):
            pass
        samples = total_steps * batch_size
        samples_per_sec = samples / current_duration if current_duration > 0 else 0
        print(f"\n  Steps: {total_steps}  |  Samples: {samples:,}  |  {samples_per_sec:.0f} samples/sec  |  Batch: {batch_size}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live oxi training monitor")
    parser.add_argument("run_dir", nargs="?", help="Path to run dir (default: latest)")
    parser.add_argument("--window", type=int, default=0, metavar="N",
                        help="Show last N steps (0 = all, default 0)")
    parser.add_argument("--interval", type=float, default=2.0, metavar="S",
                        help="Refresh interval in seconds (default 2)")
    parser.add_argument("--smooth", type=int, default=100, metavar="K",
                        help="Smoothing window size (default 100)")
    parser.add_argument("--report", action="store_true",
                        help="Print a one-shot text comparison vs best run and exit")
    parser.add_argument("--compare-run", metavar="DIR",
                        help="Comparison run dir for --report (default: best research run)")
    args = parser.parse_args()

    base = Path(__file__).parent / "research_runs"

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = Path(__file__).parent / run_dir
    else:
        run_dir = find_latest_run(base)
        if run_dir is None:
            print(f"No runs found under {base}")
            sys.exit(1)

    if args.report:
        if args.compare_run:
            cmp_dir = Path(args.compare_run)
            if not cmp_dir.is_absolute():
                candidate = Path(__file__).parent / cmp_dir
                cmp_dir = candidate if candidate.exists() else base / cmp_dir
        else:
            cmp_dir = find_best_comparison_run(base, run_dir)
        report(run_dir, cmp_dir)
        sys.exit(0)

    window = args.window
    interactive = sys.stdin.isatty()
    fd = sys.stdin.fileno() if interactive else None
    old_settings = termios.tcgetattr(fd) if interactive else None

    def wait_for_input(timeout):
        nonlocal window
        if not interactive:
            time.sleep(timeout)
            return False
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            ready, _, _ = select.select([sys.stdin], [], [], min(remaining, 0.1))
            if ready:
                ch = sys.stdin.read(1)
                if ch == "\t":
                    window = 50 if window == 0 else 0
                    return True
        return False

    try:
        if interactive:
            new_settings = termios.tcgetattr(fd)
            new_settings[3] &= ~(termios.ECHO | termios.ICANON)  # no echo, no line buffer
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        while True:
            if not args.run_dir:
                latest = find_latest_run(base)
                if latest:
                    run_dir = latest

            best_ever = best_score_from_log(base)
            os.system("clear")
            effective_smooth = 10 if window > 0 else args.smooth
            render(run_dir, window, effective_smooth, best_ever)
            wait_for_input(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if interactive:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()


if __name__ == "__main__":
    main()
