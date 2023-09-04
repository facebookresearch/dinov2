# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
from functools import partial

import torch

from .backbones import  _make_dinov2_model
from .depth import BNHead, DepthEncoderDecoder
from .utils import _DINOV2_BASE_URL, _make_dinov2_model_name, CenterPadding


class Weights(Enum):
    NYU = "NYU"
    KITTI = "KITTI"


def _make_dinov2_linear_depth_head(
    *,
    embed_dim: int = 1024,
    layers: int = 4,
    **kwargs,
):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")

    if layers == 1:
        in_index = [0]
    else:
        assert layers == 4
        in_index = [0, 1, 2, 3]

    return BNHead(
        classify=True,
        n_bins=256,
        bins_strategy="UD",
        norm_strategy="linear",
        upsample=4,
        in_channels=[embed_dim] * len(in_index),
        in_index=in_index,
        input_transform="resize_concat",
        channels=embed_dim * len(in_index) * 2,
        align_corners=False,
        min_depth=0.001,
        max_depth=10,
        loss_decode=(),
    )


def _make_dinov2_linear_depther(
    *,
    arch_name: str = "vit_large",
    layers: int = 4,
    pretrained: bool = True,
    weights: str = Weights.NYU.value,
    **kwargs,
):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")
    if weights not in (weights.value for weights in Weights):
        raise AssertionError(f"Unsupported weights: {weights}")

    backbone = _make_dinov2_model(arch_name=arch_name,
                                  pretrained=pretrained,
                                  **kwargs)

    embed_dim = backbone.embed_dim
    patch_size = backbone.patch_size
    model_name = _make_dinov2_model_name(arch_name, patch_size)
    linear_depth_head = _make_dinov2_linear_depth_head(
        arch_name=arch_name,
        embed_dim=embed_dim,
        layers=layers,
    )

    layer_counts = {
        "vit_small": 12,
        "vit_base": 12,
        "vit_large": 24,
        "vit_giant2": 40,
    }
    layer_count = layer_counts[arch_name]

    out_indices = {
        "vit_small": [2, 5, 8, 11],
        "vit_base": [2, 5, 8, 11],
        "vit_large": [4, 11, 17, 23],
        "vit_giant2": [9, 19, 29, 39],
    }

    if layers == 4:
        out_index = out_indices[arch_name]
    else:
        assert layers == 1
        out_index = [layer_count - 1]

    model = DepthEncoderDecoder(backbone=backbone, decode_head=linear_depth_head)
    model.backbone.forward = partial(
        backbone.get_intermediate_layers,
        n=out_index,
        reshape=True,
        return_class_token=True,
        norm=False,
    )
    model.backbone.register_forward_pre_hook(
        lambda _, x: CenterPadding(patch_size)(x[0])
    )

    if pretrained:
        layers_str = str(layers) if layers == 4 else ""
        weights_str = weights.lower()
        url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_{weights_str}_linear{layers_str}_head.pth"
        print(url)
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    return model


def dinov2_vits14_ld(*,
                     layers: int = 4,
                     pretrained: bool = True,
                     weights: str = Weights.NYU.value,
                     **kwargs):
    return _make_dinov2_linear_depther(arch_name="vit_small",
                                       layers=layers,
                                       pretrained=pretrained,
                                       weights=weights,
                                       **kwargs)


def dinov2_vitb14_ld(*,
                     layers: int = 4,
                     pretrained: bool = True,
                     weights: str = Weights.NYU.value,
                     **kwargs):
    return _make_dinov2_linear_depther(arch_name="vit_base",
                                       layers=layers,
                                       pretrained=pretrained,
                                       weights=weights,
                                       **kwargs)


def dinov2_vitl14_ld(*,
                     layers: int = 4,
                     pretrained: bool = True,
                     weights: str = Weights.NYU.value,
                     **kwargs):
    return _make_dinov2_linear_depther(arch_name="vit_large",
                                       layers=layers,
                                       pretrained=pretrained,
                                       weights=weights,
                                       **kwargs)


def dinov2_vitg14_ld(*,
                     layers: int = 4,
                     pretrained: bool = True,
                     weights: str = Weights.NYU.value,
                     **kwargs):
    return _make_dinov2_linear_depther(arch_name="vit_giant2",
                                       layers=layers,
                                       pretrained=pretrained,
                                       weights=weights,
                                       **kwargs)
