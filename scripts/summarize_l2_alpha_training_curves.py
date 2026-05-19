#!/usr/bin/env python3
import csv
import json
import statistics
from pathlib import Path


RUNS = {
    "alpha0_old_classic": Path(
        "/nas_data/xueyue.yang/ohora_runs/requested_l2_l3_commonsense_bs2_lr7e4_20260425_113527/"
        "llama2_7b_lr7e-4_bs2_wrr0.01_e3"
    ),
    "alpha0p3_mix": Path(
        "/nas_data/xueyue.yang/ohora_runs/l2_relative_mix_alpha_sweep_20260428_queued/"
        "llama2_7b_lr7e-4_bs2_ga32_relative_scores_mix_alpha0p3_wrr0.01_e3"
    ),
    "alpha0p5_mix": Path(
        "/nas_data/xueyue.yang/ohora_runs/init_ablation_bs2_lr7e4_20260426_173600/"
        "llama2_7b_lr7e-4_bs2_ga32_relative_scores_mix_wrr0.01_e3"
    ),
    "alpha0p7_mix": Path(
        "/nas_data/xueyue.yang/ohora_runs/l2_relative_mix_alpha_sweep_20260428_queued/"
        "llama2_7b_lr7e-4_bs2_ga32_relative_scores_mix_alpha0p7_wrr0.01_e3"
    ),
    "alpha1_relative": Path(
        "/nas_data/xueyue.yang/ohora_runs/init_ablation_bs2_lr7e4_20260426_173600/"
        "llama2_7b_lr7e-4_bs2_ga32_relative_scores_wrr0.01_e3"
    ),
}


def iter_loss_rows(run_dir: Path):
    state_path = run_dir / "trainer_state.json"
    with state_path.open() as f:
        state = json.load(f)
    for row in state.get("log_history", []):
        if "loss" not in row:
            continue
        yield {
            "step": int(row["step"]),
            "epoch": float(row.get("epoch", 0.0)),
            "loss": float(row["loss"]),
            "grad_norm": float(row["grad_norm"]) if "grad_norm" in row else None,
            "learning_rate": float(row["learning_rate"]) if "learning_rate" in row else None,
        }


def window_mean(rows, lo=None, hi=None):
    vals = []
    for row in rows:
        if lo is not None and row["step"] < lo:
            continue
        if hi is not None and row["step"] > hi:
            continue
        vals.append(row["loss"])
    return statistics.fmean(vals) if vals else None


def main():
    out_dir = Path("/nas_data/xueyue.yang/ohora_runs/ohora_numeric_analysis/llama2_alpha_training_curves_20260502")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name, run_dir in RUNS.items():
        rows = list(iter_loss_rows(run_dir))
        with (out_dir / f"{name}_loss_curve.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "epoch", "loss", "grad_norm", "learning_rate"])
            writer.writeheader()
            writer.writerows(rows)

        losses = [row["loss"] for row in rows]
        final_window = rows[-50:] if len(rows) >= 50 else rows
        first_window = rows[:50] if len(rows) >= 50 else rows
        summary_rows.append(
            {
                "name": name,
                "n_logs": len(rows),
                "first_step": rows[0]["step"],
                "last_step": rows[-1]["step"],
                "min_loss": min(losses),
                "mean_loss_first50": statistics.fmean(row["loss"] for row in first_window),
                "mean_loss_last50": statistics.fmean(row["loss"] for row in final_window),
                "mean_loss_step_0_1000": window_mean(rows, hi=1000),
                "mean_loss_step_1000_3000": window_mean(rows, lo=1000, hi=3000),
                "mean_loss_step_3000_6000": window_mean(rows, lo=3000, hi=6000),
                "mean_loss_step_6000_end": window_mean(rows, lo=6000),
            }
        )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    for row in summary_rows:
        print(
            f"{row['name']}: last50={row['mean_loss_last50']:.6f} "
            f"min={row['min_loss']:.6f} logs={row['n_logs']}"
        )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
