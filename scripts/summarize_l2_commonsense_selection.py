#!/usr/bin/env python3
import argparse
import csv
import glob
import re
import statistics
from pathlib import Path


DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
]

METHODS = [
    "classic",
    "relative_scores",
    "relative_scores_mix_a0.5",
    "classic_mix_rerank_a0.5",
]


def method_from_tag(tag: str) -> str:
    if tag.startswith("l2_crr"):
        return "classic_mix_rerank_a0.5"
    if tag.startswith("l2_mix"):
        return "relative_scores_mix_a0.5"
    if tag.startswith("l2_relative"):
        return "relative_scores"
    if tag.startswith("l2_classic"):
        return "classic"
    raise ValueError(f"Cannot infer method from tag: {tag}")


def read_summary(path: Path) -> dict:
    result = {}
    total = 0
    correct = 0
    with path.open() as f:
        for row in csv.DictReader(f):
            dataset = row["dataset"]
            result[dataset] = float(row["accuracy"])
            total += int(row["n"])
            correct += int(row["correct"])
    missing = sorted(set(DATASETS) - set(result))
    if missing:
        raise ValueError(f"{path} missing datasets: {missing}")
    result["avg"] = sum(result[d] for d in DATASETS) / len(DATASETS)
    result["weighted"] = correct / total
    return result


def collect(run_root: Path) -> list[dict]:
    rows = []
    pattern = str(run_root / "l2_commonsense_seed*" / "eval_commonsense_*" / "*" / "summary.csv")
    for path_str in sorted(glob.glob(pattern)):
        path = Path(path_str)
        match = re.search(r"l2_commonsense_seed(\d+).*?/(l2_[^/]+)_seed", path_str)
        if not match:
            continue
        seed = int(match.group(1))
        tag = match.group(2)
        row = {
            "seed": seed,
            "method": method_from_tag(tag),
            "summary_path": str(path),
        }
        row.update(read_summary(path))
        rows.append(row)
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def write_csv(rows: list[dict], output: Path) -> None:
    fields = ["method", "seed", "avg", "weighted", *DATASETS, "summary_path"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: (METHODS.index(x["method"]), x["seed"])):
            writer.writerow({k: row[k] for k in fields})


def print_report(rows: list[dict]) -> None:
    print("Per-run")
    print("method,seed,avg,weighted")
    for method in METHODS:
        for row in sorted([r for r in rows if r["method"] == method], key=lambda x: x["seed"]):
            print(f"{method},{row['seed']},{row['avg']:.6f},{row['weighted']:.6f}")

    summaries = {}
    print("\nBy method")
    print("method,avg_mean,avg_std,weighted_mean,weighted_std")
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        avg_mean, avg_std = mean_std([r["avg"] for r in method_rows])
        weighted_mean, weighted_std = mean_std([r["weighted"] for r in method_rows])
        summaries[method] = {
            "avg": avg_mean,
            "weighted": weighted_mean,
            **{dataset: statistics.mean([r[dataset] for r in method_rows]) for dataset in DATASETS},
        }
        print(f"{method},{avg_mean:.6f},{avg_std:.6f},{weighted_mean:.6f},{weighted_std:.6f}")

    if "classic" in summaries:
        print("\nDelta vs classic")
        print("method,avg_delta,weighted_delta")
        base = summaries["classic"]
        for method in METHODS:
            if method == "classic" or method not in summaries:
                continue
            print(
                f"{method},"
                f"{summaries[method]['avg'] - base['avg']:+.6f},"
                f"{summaries[method]['weighted'] - base['weighted']:+.6f}"
            )

    print("\nBest per metric")
    for metric in ["avg", "weighted", *DATASETS]:
        available = [method for method in METHODS if method in summaries]
        best = max(available, key=lambda method: summaries[method][metric])
        print(f"{metric},{best},{summaries[best][metric]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = collect(args.run_root)
    if not rows:
        raise SystemExit(f"No summaries found under {args.run_root}")
    print_report(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, args.output)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
