# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
import torch.nn.functional as F

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, return_dense_tokens: bool = False):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            return_dense_tokens (bool): If True, return only the stacked per-layer patch tokens tensor
                with shape [B, S, L, Hp*Wp, 2048]. Default: False.

        Returns:
            dict or torch.Tensor:
                If return_dense_tokens is False:
                    dict with:
                    - pose_enc (torch.Tensor): [B, S, 9]
                    - depth (torch.Tensor): [B, S, H, W, 1]
                    - depth_conf (torch.Tensor): [B, S, H, W]
                    - world_points (torch.Tensor): [B, S, H, W, 3]
                    - world_points_conf (torch.Tensor): [B, S, H, W]
                    - layer_patch_tokens (torch.Tensor): [B, S, L, Hp*Wp, 2048]
                    - aggregated_tokens (torch.Tensor): last-layer tokens [B, S, P, 2048]
                    - images (torch.Tensor): input images
                    Optionally (if query_points provided):
                    - track (torch.Tensor): [B, S, N, 2]
                    - vis (torch.Tensor): [B, S, N]
                    - conf (torch.Tensor): [B, S, N]
                If return_dense_tokens is True:
                    torch.Tensor with shape [B, S, L, Hp*Wp, 2048].
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        # aggregated_tokens_list: list length=L (e.g., 24), each [B, S, P, 2048]
        # slice patch tokens (exclude camera/register) and stack across layers -> [B, S, L, Hp*Wp, 2048]
        layer_patch_tokens = torch.stack(
            [x[:, :, patch_start_idx:, :] for x in aggregated_tokens_list], dim=2
        )

        # If requested, return only the dense features (useful for ONNX export)
        if return_dense_tokens:
            return layer_patch_tokens

        predictions = {}

        # Keep last layer tokens for backward compatibility
        predictions["aggregated_tokens"] = aggregated_tokens_list[-1]

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        # Expose stacked per-layer patch tokens in predictions as well
        predictions["layer_patch_tokens"] = layer_patch_tokens  # [B, S, L, Hp*Wp, 2048]
        predictions["images"] = images

        return predictions