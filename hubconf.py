# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


dependencies = ["torch"]


_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov2_{compact_arch_name}{patch_size}"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    pretrained: bool = True,
    **kwargs,
):
    from dinov2.models import vision_transformer as vits

    model_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model


def dinov2_vits14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_small", pretrained=pretrained, **kwargs)


def dinov2_vitb14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-B/14 model pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained, **kwargs)


def dinov2_vitl14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_large", pretrained=pretrained, **kwargs)


def dinov2_vitg14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_giant2", ffn_layer="swiglufused", pretrained=pretrained, **kwargs)


def _make_dinov2_linear_head(
    *,
    model_name: str = "dinov2_vitl14",
    embed_dim: int = 1024,
    layers: int = 4,
    pretrained: bool = True,
    **kwargs,
):
    assert layers in (1, 4), f"Unsupported number of layers: {layers}"
    linear_head = nn.Linear((1 + layers) * embed_dim, 1_000)

    if pretrained:
        layers_str = str(layers) if layers == 4 else ""
        url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_linear{layers_str}_head.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        linear_head.load_state_dict(state_dict, strict=False)

    return linear_head


class _LinearClassifierWrapper(nn.Module):
    def __init__(self, *, backbone: nn.Module, linear_head: nn.Module, layers: int = 4):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head
        self.layers = layers

    def forward(self, x):
        if self.layers == 1:
            x = self.backbone.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            # fmt: off
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
            # fmt: on
        elif self.layers == 4:
            x = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
            # fmt: off
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
            # fmt: on
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        return self.linear_head(linear_input)


def _make_dinov2_linear_classifier(
    *,
    arch_name: str = "vit_large",
    layers: int = 4,
    pretrained: bool = True,
    **kwargs,
):
    backbone = _make_dinov2_model(arch_name=arch_name, pretrained=pretrained, **kwargs)

    embed_dim = backbone.embed_dim
    patch_size = backbone.patch_size
    model_name = _make_dinov2_model_name(arch_name, patch_size)
    linear_head = _make_dinov2_linear_head(
        model_name=model_name, embed_dim=embed_dim, layers=layers, pretrained=pretrained
    )

    return _LinearClassifierWrapper(backbone=backbone, linear_head=linear_head, layers=layers)


def dinov2_vits14_lc(*, layers: int = 4, pretrained: bool = True, **kwargs):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-S/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(arch_name="vit_small", layers=layers, pretrained=pretrained, **kwargs)


def dinov2_vitb14_lc(*, pretrained: bool = True, **kwargs):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-B/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(arch_name="vit_base", pretrained=pretrained, **kwargs)


def dinov2_vitl14_lc(*, pretrained: bool = True, **kwargs):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-L/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(arch_name="vit_large", pretrained=pretrained, **kwargs)


def dinov2_vitg14_lc(*, pretrained: bool = True, **kwargs):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-g/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name="vit_giant2", ffn_layer="swiglufused", pretrained=pretrained, **kwargs
    )
