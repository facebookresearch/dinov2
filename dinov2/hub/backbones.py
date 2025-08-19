# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Union
from torch.nn.functional import pad

import torch

from .utils import _DINOV2_BASE_URL, _make_dinov2_model_name


class Weights(Enum):
    LVD142M = "LVD142M"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    from ..models import vision_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model


def dinov2_vits14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_small", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitb14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitl14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_large", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitg14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        **kwargs,
    )


def dinov2_vits14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-S/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_small",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitb14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-B/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_base",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitl14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-L/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_large",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitg14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-g/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vits16_hipt(*, pretrained: bool = True, img_size: int = 224, use_teacher_weights: bool = False, **kwargs):
    """
    DINOv2 ViT-s/16 model (optionally) pretrained on TCGA.

    Pretrained weights are copied from the HIPT model:
    https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth
    """
    if img_size < 224:
        raise NotImplementedError('Shrinking position embeddings is not currently supported')

    model = _make_dinov2_model(
        arch_name='vit_small',
        img_size=img_size,
        patch_size=16,
        pretrained=False,
    )

    if pretrained:
        if use_teacher_weights:
            state_dict_key = 'teacher'
            backbone_prefix = 'backbone.'
        else:
            state_dict_key = 'student'
            backbone_prefix = 'module.backbone.'

        hipt_state_dict = torch.hub.load_state_dict_from_url(
            'https://github.com/mahmoodlab/HIPT/raw/a9b5bb8d159684fc4c2c497d68950ab915caeb7e/HIPT_4K/Checkpoints/vit256_small_dino.pth',
            map_location="cpu",
        )
        hipt_backbone_weights = {
            name[len(backbone_prefix):]: params
            for name, params in hipt_state_dict[state_dict_key].items()
            if name.startswith(backbone_prefix)
        }

        hipt_backbone_weights['mask_token'] = model.mask_token

        # Initialise layer scale (gamma) to 1
        for i, block in enumerate(model.blocks):
            hipt_backbone_weights[f'blocks.{i}.ls1.gamma'] = torch.ones_like(block.ls1.gamma)
            hipt_backbone_weights[f'blocks.{i}.ls2.gamma'] = torch.ones_like(block.ls2.gamma)

        # Changing the input image size for a vision transformer model is tricky.
        #
        # The established approach is to interpolate the position embeddings, but this is only really
        # appropriate when change in input image size implies scale change (in terms of pathology,
        # this corresponds to a change in magnification).
        #
        # For us, changing the input size doesn't mean changing scale---rather, we're adding additional
        # surrounding context at the same scale.
        #
        # I don't think there's a perfect solution to this problem, so I'll just use reflection padding and
        # accept that some amount of further training will be required later on. A potential risk of this
        # approach is that we are now presenting the model with multiple patches that have the same position
        # embedding. The flipped ordering caused by reflection is a bit strange, too.
        #
        # Another options would be random initialisation, but that seems risky given that the model has
        # never seen those.
        pos_embed_hipt = hipt_backbone_weights.pop('pos_embed')
        num_patches = model.patch_embed.num_patches
        class_pos_embed = pos_embed_hipt[:, 0:1]
        patch_pos_embed = pos_embed_hipt[:, 1:]
        old_num_patches = pos_embed_hipt.shape[1]
        old_sz = round(old_num_patches ** 0.5)
        sz = round(num_patches ** 0.5)
        sz_diff = sz - old_sz
        patch_pos_embed = patch_pos_embed.view(14, 14, 384)
        pad_before = sz_diff // 2
        pad_after = sz_diff - pad_before
        new_patch_pos_embed = pad(patch_pos_embed.permute(2, 0, 1), (pad_before, pad_after, pad_before, pad_after), mode='reflect').permute(1, 2, 0)
        hipt_backbone_weights['pos_embed'] = torch.cat([
            class_pos_embed,
            new_patch_pos_embed.view(1, num_patches, model.num_features),
        ], axis=1)

        model.load_state_dict(hipt_backbone_weights)

    return model
