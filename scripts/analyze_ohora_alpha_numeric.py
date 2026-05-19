#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

TRAINED_STRATEGIES = (
    "classic",
    "mix_alpha0p3",
    "mix_alpha0p5",
    "mix_alpha0p7",
    "relative",
)

ALL_STRATEGIES = TRAINED_STRATEGIES + (
    "mix_alpha0p0_norm_only",
    "classic_abs_diag",
)


def constrained_gcd(m: int, n: int) -> int:
    t = min(m, n)
    k = math.isqrt(t)
    for i in range(k, 0, -1):
        if m % i == 0 and n % i == 0:
            return i
    return 1


def layer_and_module(name: str):
    m = re.search(r"model\.layers\.(\d+)\.", name)
    if not m:
        return None, None
    layer = int(m.group(1))
    for module in TARGET_MODULES:
        if f".{module}.weight" in name:
            return layer, module
    return layer, None


def load_weight_map(model_dir: Path):
    with open(model_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    return index["weight_map"]


def iter_target_weights(model_dir: Path, layers: set[int] | None):
    weight_map = load_weight_map(model_dir)
    by_file = defaultdict(list)
    for name, filename in weight_map.items():
        layer, module = layer_and_module(name)
        if module is None:
            continue
        if layers is not None and layer not in layers:
            continue
        by_file[filename].append(name)

    for filename in sorted(by_file):
        path = model_dir / filename
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in sorted(by_file[filename]):
                yield name, f.get_tensor(name)


def tensor_quantiles(x: torch.Tensor):
    qs = torch.quantile(x.float(), torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=x.device))
    return {
        "min": float(qs[0].item()),
        "p10": float(qs[1].item()),
        "p25": float(qs[2].item()),
        "p50": float(qs[3].item()),
        "p75": float(qs[4].item()),
        "p90": float(qs[5].item()),
        "max": float(qs[6].item()),
        "mean": float(x.float().mean().item()),
        "std": float(x.float().std(unbiased=False).item()),
    }


def select_indices(scores: torch.Tensor, r: int):
    return torch.topk(scores, 2 * r).indices


def strategy_scores(r_diag_signed, r_diag_abs, w_norms):
    relative = r_diag_abs / (w_norms + 1e-8)
    return {
        "classic": r_diag_signed,
        "classic_abs_diag": r_diag_abs,
        "relative": relative,
        "mix_alpha0p0_norm_only": w_norms,
        "mix_alpha0p3": (w_norms ** 0.7) * (relative ** 0.3),
        "mix_alpha0p5": (w_norms ** 0.5) * (relative ** 0.5),
        "mix_alpha0p7": (w_norms ** 0.3) * (relative ** 0.7),
    }


def overlap_stats(a: torch.Tensor, b: torch.Tensor):
    a_set = set(int(x) for x in a.detach().cpu().tolist())
    b_set = set(int(x) for x in b.detach().cpu().tolist())
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return {
        "intersection": inter,
        "frac_of_a": inter / len(a_set) if a_set else 0.0,
        "jaccard": inter / union if union else 0.0,
    }


def selected_metrics(indices, r, q, r_mat, weight, r_diag_abs, w_norms, relative, scaling=2.0, compute_residual=True):
    out_dim, in_dim = q.shape[0], r_mat.shape[1]
    b_index = indices[:r]
    a_index = indices[r:]
    a_row = out_dim // r
    b_col = in_dim // r

    q_a = q[:a_row, a_index]
    r_a = r_mat[a_index, :r]
    q_b = q[:b_col, b_index]
    r_b = r_mat[b_index, :r]
    lora_a = q_a @ r_a
    lora_b = (q_b @ r_b).T

    weight_norm = torch.linalg.norm(weight).item()
    delta_norm = (scaling * torch.linalg.norm(lora_a) * torch.linalg.norm(lora_b)).item()
    out = {
        "r_diag_abs_mean": float(r_diag_abs[indices].mean().item()),
        "r_diag_abs_median": float(r_diag_abs[indices].median().item()),
        "w_norm_mean": float(w_norms[indices].mean().item()),
        "w_norm_median": float(w_norms[indices].median().item()),
        "relative_mean": float(relative[indices].mean().item()),
        "relative_median": float(relative[indices].median().item()),
        "selected_r_diag_energy_frac": float((r_diag_abs[indices].pow(2).sum() / (r_diag_abs.pow(2).sum() + 1e-12)).item()),
        "selected_w_norm_energy_frac": float((w_norms[indices].pow(2).sum() / (w_norms.pow(2).sum() + 1e-12)).item()),
        "delta_norm_ratio": delta_norm / weight_norm if weight_norm else 0.0,
    }

    if compute_residual:
        delta = torch.kron(lora_a.contiguous(), lora_b.contiguous())
        residual = weight - scaling * delta
        out["residual_norm_ratio"] = float(torch.linalg.norm(residual).item() / weight_norm) if weight_norm else 0.0
        out["delta_weight_cosine"] = float((torch.sum(delta * weight) / ((torch.linalg.norm(delta) * torch.linalg.norm(weight)) + 1e-12)).item())
        del delta, residual

    return out


def existing_done(jsonl_path: Path):
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add(row["name"])
    return done


def analyze(args):
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "module_stats.jsonl"
    done = existing_done(jsonl_path) if args.resume else set()
    layers = None if args.layers == "all" else {int(x) for x in args.layers.split(",") if x}
    device = torch.device(args.device)

    with open(jsonl_path, "a") as out_f:
        for name, weight_cpu in iter_target_weights(model_dir, layers):
            if name in done:
                print(f"skip existing {name}", flush=True)
                continue
            started = time.time()
            layer, module = layer_and_module(name)
            weight = weight_cpu.to(device=device, dtype=torch.float32)
            out_dim, in_dim = weight.shape
            r = constrained_gcd(out_dim, in_dim)
            k = min(out_dim, in_dim)

            print(f"analyze {name} shape={tuple(weight.shape)} r={r}", flush=True)
            q, r_mat = torch.linalg.qr(weight, mode="reduced")
            r_diag_signed = torch.diag(r_mat)
            r_diag_abs = torch.abs(r_diag_signed)
            w_norms = torch.linalg.norm(weight[:, :k], dim=0)
            relative = r_diag_abs / (w_norms + 1e-8)
            scores = strategy_scores(r_diag_signed, r_diag_abs, w_norms)
            indices = {strategy: select_indices(score, r) for strategy, score in scores.items()}

            strategy_rows = {}
            for strategy in ALL_STRATEGIES:
                strategy_rows[strategy] = selected_metrics(
                    indices[strategy],
                    r,
                    q,
                    r_mat,
                    weight,
                    r_diag_abs,
                    w_norms,
                    relative,
                    compute_residual=not args.no_residual,
                )

            overlap_rows = {}
            for a in ALL_STRATEGIES:
                for b in ALL_STRATEGIES:
                    if a >= b:
                        continue
                    key = f"{a}__{b}"
                    overlap_rows[key] = overlap_stats(indices[a], indices[b])
                    overlap_rows[f"{key}__b_block"] = overlap_stats(indices[a][:r], indices[b][:r])
                    overlap_rows[f"{key}__a_block"] = overlap_stats(indices[a][r:], indices[b][r:])

            row = {
                "name": name,
                "layer": layer,
                "module": module,
                "shape": [int(out_dim), int(in_dim)],
                "r": r,
                "elapsed_sec": time.time() - started,
                "distributions": {
                    "r_diag_signed": tensor_quantiles(r_diag_signed),
                    "r_diag_abs": tensor_quantiles(r_diag_abs),
                    "w_norms": tensor_quantiles(w_norms),
                    "relative": tensor_quantiles(relative),
                },
                "strategies": strategy_rows,
                "overlaps": overlap_rows,
            }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

            del weight, q, r_mat, r_diag_signed, r_diag_abs, w_norms, relative, scores, indices
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate(out_dir)


def aggregate(out_dir: Path):
    rows = []
    with open(out_dir / "module_stats.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return

    trained = list(TRAINED_STRATEGIES)
    metric_names = [
        "r_diag_abs_mean",
        "w_norm_mean",
        "relative_mean",
        "selected_r_diag_energy_frac",
        "selected_w_norm_energy_frac",
        "delta_norm_ratio",
        "residual_norm_ratio",
        "delta_weight_cosine",
    ]

    with open(out_dir / "strategy_summary_by_module.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "strategy", "n_modules"] + metric_names)
        groups = {"all": rows}
        for module in TARGET_MODULES:
            groups[module] = [r for r in rows if r["module"] == module]
        groups["attention"] = [r for r in rows if r["module"] in {"q_proj", "k_proj", "v_proj", "o_proj"}]
        groups["mlp"] = [r for r in rows if r["module"] in {"gate_proj", "up_proj", "down_proj"}]
        for group, group_rows in groups.items():
            if not group_rows:
                continue
            for strategy in trained:
                values = []
                for metric in metric_names:
                    vals = [
                        r["strategies"][strategy][metric]
                        for r in group_rows
                        if metric in r["strategies"][strategy]
                    ]
                    values.append(sum(vals) / len(vals) if vals else "")
                writer.writerow([group, strategy, len(group_rows)] + values)

    pairs = [
        ("classic", "mix_alpha0p3"),
        ("classic", "mix_alpha0p5"),
        ("classic", "mix_alpha0p7"),
        ("classic", "relative"),
        ("mix_alpha0p3", "mix_alpha0p5"),
        ("mix_alpha0p5", "mix_alpha0p7"),
        ("mix_alpha0p7", "relative"),
        ("classic", "mix_alpha0p0_norm_only"),
    ]
    with open(out_dir / "overlap_summary_by_module.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "pair", "block", "n_modules", "frac_overlap", "jaccard"])
        groups = {"all": rows}
        for module in TARGET_MODULES:
            groups[module] = [r for r in rows if r["module"] == module]
        groups["attention"] = [r for r in rows if r["module"] in {"q_proj", "k_proj", "v_proj", "o_proj"}]
        groups["mlp"] = [r for r in rows if r["module"] in {"gate_proj", "up_proj", "down_proj"}]
        for group, group_rows in groups.items():
            if not group_rows:
                continue
            for a, b in pairs:
                pair_key = f"{a}__{b}" if a < b else f"{b}__{a}"
                for suffix, block in [("", "all"), ("__b_block", "b"), ("__a_block", "a")]:
                    vals = []
                    jacs = []
                    for row in group_rows:
                        key = pair_key + suffix
                        if key in row["overlaps"]:
                            vals.append(row["overlaps"][key]["frac_of_a"])
                            jacs.append(row["overlaps"][key]["jaccard"])
                    writer.writerow([
                        group,
                        f"{a}__{b}",
                        block,
                        len(vals),
                        sum(vals) / len(vals) if vals else "",
                        sum(jacs) / len(jacs) if jacs else "",
                    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/nas_data/xueyue.yang/hf_models/Llama-2-7b-hf")
    parser.add_argument("--out_dir", default="/nas_data/xueyue.yang/ohora_runs/ohora_numeric_analysis/llama2_alpha_selection")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer ids, e.g. 0,1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_residual", action="store_true")
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
