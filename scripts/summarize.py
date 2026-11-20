"""Summarize HTTP JSONL runs and compare probabilities on identical payloads."""

import argparse
import json
import math
import statistics
from pathlib import Path


def load_run(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in records if row["phase"] == "measured"]
    if not rows:
        raise ValueError(f"No measured samples in {path}")
    for row in records:
        if row["phase"] in ("health_before", "health_after"):
            health = row["health"]
            if health["checkpoint_devices"].get("multilingual") != "cuda":
                raise ValueError(f"Non-CUDA health in {path}")
            if health["cpu_fallbacks"]["multilingual"]["count"]:
                raise ValueError(f"CPU fallback in {path}")
    if any(row["usage"].get("truncated") for row in rows):
        raise ValueError(f"Truncated input in {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    runs = [load_run(path) for path in args.runs]
    for path, rows in zip(args.runs, runs):
        for count in sorted({row["questions"] for row in rows}):
            values = sorted(row["http_ms"] for row in rows if row["questions"] == count)
            p95 = values[math.ceil(0.95 * len(values)) - 1]
            print(f"{path.name} Q={count} n={len(values)} "
                  f"HTTP p50={statistics.median(values):.2f} p95={p95:.2f} ms")
    reference = {
        (row["request_sha256"], row["iteration"]): row for row in runs[0]
    }
    for path, rows in zip(args.runs[1:], runs[1:]):
        candidates = {(row["request_sha256"], row["iteration"]): row for row in rows}
        if reference.keys() != candidates.keys():
            raise ValueError(f"Payloads or iterations differ between {args.runs[0]} and {path}")
        differences = []
        flips = 0
        unique = set()
        for key, row in candidates.items():
            before = reference[key]
            if before["probabilities"].keys() != row["probabilities"].keys():
                raise ValueError(f"Answer IDs differ in {path}")
            for question, probability in row["probabilities"].items():
                original = before["probabilities"][question]
                differences.append(abs(probability - original))
                flips += (probability >= args.threshold) != (original >= args.threshold)
                unique.add((row["request_sha256"], question))
        print(f"{path.name} vs {args.runs[0].name} max|delta p|={max(differences):.4f} "
              f"threshold={args.threshold:g} flips={flips}/{len(differences)} "
              f"unique_decisions={len(unique)}")


if __name__ == "__main__":
    main()
