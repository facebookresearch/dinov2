# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
from typing import Callable

import torch
from torch import nn, Tensor

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


logger = logging.getLogger("dinov2")


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    def _attn_residual(self, x: Tensor) -> Tensor:
        return self.ls1(self.attn(self.norm1(x)))

    def _ffn_residual(self, x: Tensor) -> Tensor:
        return self.ls2(self.mlp(self.norm2(x)))

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = self.drop_add_attn_residual_stochastic_depth(
                x,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = self.drop_add_ffn_residual_stochastic_depth(
                x,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(self._attn_residual(x))
            x = x + self.drop_path1(self._ffn_residual(x))
        else:
            x = x + self._attn_residual(x)
            x = x + self._ffn_residual(x)
        return x

    def drop_add_attn_residual_stochastic_depth(
        self,
        x: Tensor,
        sample_drop_ratio: float = 0.0,
    ) -> Tensor:
        # 1) extract subset using permutation
        b, n, d = x.shape
        sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
        brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
        x_subset = x[brange]

        # 2) apply residual_func to get residual
        residual = self._attn_residual(x_subset)

        x_flat = x.flatten(1)
        residual = residual.flatten(1)

        residual_scale_factor = b / sample_subset_size

        # 3) add the residual
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
        return x_plus_residual.view_as(x)

    def drop_add_ffn_residual_stochastic_depth(
        self,
        x: Tensor,
        sample_drop_ratio: float = 0.0,
    ) -> Tensor:
        # 1) extract subset using permutation
        b, n, d = x.shape
        sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
        brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
        x_subset = x[brange]

        # 2) apply residual_func to get residual
        residual = self._ffn_residual(x_subset)

        x_flat = x.flatten(1)
        residual = residual.flatten(1)

        residual_scale_factor = b / sample_subset_size

        # 3) add the residual
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
        return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor):
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual
