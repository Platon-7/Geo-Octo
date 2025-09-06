# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Thin wrappers around VGGT to expose per-layer token features for ONNX export.

from typing import List, Sequence

import torch
import torch.nn as nn


class VGGTAllLayersFeatures(nn.Module):
    """
    Wrapper that returns concatenated per-layer token features of shape [D, P, 2C].

    Notes:
    - Uses only the first batch index and first sequence index to match dataset expectations.
    - D equals aggregator.depth (typically 24).
    - P equals number of tokens per frame (camera + register + patch tokens).
    - The channel dimension is 2C due to frame/global concatenation inside the aggregator.
    """

    def __init__(self, backbone: nn.Module, num_patch_tokens: int = 261):
        super().__init__()
        self.backbone = backbone
        self.num_patch_tokens = num_patch_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Accept [S, 3, H, W] or [B, S, 3, H, W]
        if images.dim() == 4:
            images = images.unsqueeze(0)
        assert images.dim() == 5, "Expected input of shape [B, S, 3, H, W] or [S, 3, H, W]"

        aggregated_tokens_list, patch_start_idx = self.backbone.aggregator(images)
        # aggregated_tokens_list: list of tensors [B, S, P, 2C], length D

        # Select first batch and first sequence item, slice patch tokens only; shape -> [261, 2C]
        start = int(patch_start_idx)
        end = start + int(self.num_patch_tokens)
        per_layer: List[torch.Tensor] = [t[0, 0, start:end, :] for t in aggregated_tokens_list]
        # Stack into [D, 261, 2C]
        return torch.stack(per_layer, dim=0)


class VGGTSelectedLayersFeatures(nn.Module):
    """
    Wrapper that returns selected per-layer token features of shape [K, P, 2C].

    layer_indices are 0-based indices into the aggregator depth.
    """

    def __init__(self, backbone: nn.Module, layer_indices: Sequence[int], num_patch_tokens: int = 261):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("layer_indices_tensor", torch.tensor(list(layer_indices), dtype=torch.long))
        self.num_patch_tokens = num_patch_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Accept [S, 3, H, W] or [B, S, 3, H, W]
        if images.dim() == 4:
            images = images.unsqueeze(0)
        assert images.dim() == 5, "Expected input of shape [B, S, 3, H, W] or [S, 3, H, W]"

        aggregated_tokens_list, patch_start_idx = self.backbone.aggregator(images)
        # aggregated_tokens_list: list of tensors [B, S, P, 2C], length D

        selected = []
        start = int(patch_start_idx)
        end = start + int(self.num_patch_tokens)
        for idx in self.layer_indices_tensor.tolist():
            t = aggregated_tokens_list[idx]
            selected.append(t[0, 0, start:end, :])  # [261, 2C]
        return torch.stack(selected, dim=0)  # [K, 261, 2C]

