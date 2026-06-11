#!/usr/bin/env python3
"""Summarize training throughput from a train.log with perf_full_iteration lines.

Usage: python3 scripts/bench_stats.py <log_dir>... [--batch N]
"""
import re
import statistics as st
import sys


def load(path):
    rows = {}
    for line in open(path):
        m = re.search(r"perf_full_iteration: iter=(\d+) total=([\d.]+)(ms|s)", line)
        if m:
            v = float(m.group(2))
            if m.group(3) == "ms":
                v /= 1000
            rows[int(m.group(1))] = v
    return rows


def main():
    argv = sys.argv[1:]
    batch = 512
    if "--batch" in argv:
        i = argv.index("--batch")
        batch = int(argv[i + 1])
        del argv[i : i + 2]
    args = argv
    for d in args:
        r = load(f"{d}/train.log")
        vals = [v for i, v in r.items() if i >= 5]
        if not vals:
            print(f"{d}: no data")
            continue
        normal = [v for i, v in r.items() if i >= 5 and i % 20 and i % 50]
        probe = [v for i, v in r.items() if i >= 5 and i % 20 == 0]
        print(
            f"{d}: iters={len(vals)} wall={sum(vals):.0f}s "
            f"mean={st.mean(vals) * 1000:.0f}ms "
            f"median_normal={st.median(normal) * 1000:.0f}ms "
            f"median_probe={st.median(probe) * 1000:.0f}ms "
            f"samples/s={batch * len(vals) / sum(vals):.0f}"
        )


if __name__ == "__main__":
    main()
