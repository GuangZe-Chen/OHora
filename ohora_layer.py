from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rhigh(rows: int, cols: int) -> int:
    limit = min(math.isqrt(rows), math.isqrt(cols))
    for candidate in range(limit, 0, -1):
        if rows % candidate == 0 and cols % candidate == 0:
            return candidate
    return 1


@dataclass(frozen=True)
class OHoRAInitResult:
    rhigh: int
    Ahigh: torch.Tensor
    Bhigh: torch.Tensor
    Wres: torch.Tensor
    selected_indices: torch.Tensor


def build_ohora_factors(weight: torch.Tensor) -> OHoRAInitResult:
    rows, cols = weight.shape
    if rows != cols:
        raise ValueError(
            "The minimal OHoRA layer currently supports square nn.Linear weights only. "
            f"Received weight shape {(rows, cols)}."
        )

    weight_fp32 = weight.detach().to(torch.float32)
    rhigh = compute_rhigh(rows, cols)
    q_matrix, r_matrix = torch.linalg.qr(weight_fp32, mode="reduced")

    non_redundant_scores = r_matrix.diagonal().abs()
    selected_indices = torch.topk(non_redundant_scores, k=2 * rhigh, largest=True).indices
    first_indices = selected_indices[:rhigh]
    second_indices = selected_indices[rhigh:]

    q_first = q_matrix[:, first_indices]
    r_first = r_matrix[first_indices, :]
    q_second = q_matrix[:, second_indices]
    r_second = r_matrix[second_indices, :]

    a_rows = rows // rhigh
    b_rows = cols // rhigh
    ahigh = q_first[:a_rows, :] @ r_first[:, :rhigh]
    bhigh = q_second[:b_rows, :] @ r_second[:, :rhigh]

    wres = weight_fp32 - torch.kron(ahigh.contiguous(), bhigh.transpose(0, 1).contiguous())
    return OHoRAInitResult(
        rhigh=rhigh,
        Ahigh=ahigh,
        Bhigh=bhigh,
        Wres=wres,
        selected_indices=selected_indices,
    )


class OHoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"OHoRALinear expects nn.Linear, received {type(base_linear)!r}.")

        init_result = build_ohora_factors(base_linear.weight)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rhigh = init_result.rhigh
        self.a_rows = init_result.Ahigh.shape[0]
        self.b_rows = init_result.Bhigh.shape[0]

        self.register_buffer("w0", base_linear.weight.detach().clone())
        self.register_buffer("wres", init_result.Wres.to(base_linear.weight.dtype))
        self.register_buffer("selected_indices", init_result.selected_indices.detach().clone())

        if base_linear.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", base_linear.bias.detach().clone())

        self.Ahigh = nn.Parameter(init_result.Ahigh.to(base_linear.weight.dtype))
        self.Bhigh = nn.Parameter(init_result.Bhigh.to(base_linear.weight.dtype))

    @classmethod
    def from_linear(cls, base_linear: nn.Linear) -> "OHoRALinear":
        return cls(base_linear)

    def delta_weight(self) -> torch.Tensor:
        return torch.kron(self.Ahigh.contiguous(), self.Bhigh.transpose(0, 1).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        base_output = F.linear(x_flat, self.wres, self.bias)

        x_blocks = x_flat.reshape(-1, self.rhigh, self.b_rows).transpose(1, 2)
        b_transposed = self.Bhigh.transpose(0, 1).unsqueeze(0)
        intermediate = torch.matmul(b_transposed, x_blocks)
        delta_blocks = torch.matmul(intermediate, self.Ahigh.transpose(0, 1))
        delta_output = delta_blocks.transpose(1, 2).reshape(-1, self.out_features)

        output = base_output + delta_output
        return output.reshape(*original_shape[:-1], self.out_features)