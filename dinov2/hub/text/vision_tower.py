from functools import partial
from typing import Callable, Tuple

import torch
from torch import nn, Tensor

from dinov2.layers import (
    LayerScale,
    NestedTensorBlock as AttentionBlock,
    SwiGLUFFNAligned as SwiGLUFFN,
)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, nn.Conv2d):
        module.reset_parameters()


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class VisionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        blocks_drop_path: float,
        use_class_token: bool,
        use_patch_tokens: bool,
        use_linear_projection: bool,
    ):
        super().__init__()
        block_list = [nn.Identity()]
        self.ln_final = nn.Identity()
        if num_blocks > 0:
            block_list = [
                AttentionBlock(
                    input_dim,
                    num_heads,
                    ffn_layer=partial(SwiGLUFFN, align_to=64),
                    init_values=1e-5,
                    drop_path=blocks_drop_path,
                )
                for _ in range(num_blocks)
            ]
            self.ln_final = nn.LayerNorm(input_dim)
        self.block_list = nn.ModuleList(block_list)
        self.num_blocks = num_blocks
        multiplier = 2 if use_class_token and use_patch_tokens else 1
        self.linear_projection = nn.Identity()
        if multiplier * input_dim != embed_dim or use_linear_projection:
            assert embed_dim % multiplier == 0, f"Expects {embed_dim} to be divisible by {multiplier}"
            self.linear_projection = nn.Linear(input_dim, embed_dim // multiplier, bias=False)

    def init_weights(self):
        if self.num_blocks > 0:
            for i in range(self.num_blocks):
                block = self.block_list[i]
                named_apply(init_weights_vit_timm, block)
            self.ln_final.reset_parameters()
        if isinstance(self.linear_projection, nn.Linear):
            nn.init.normal_(self.linear_projection.weight, std=self.linear_projection.in_features**-0.5)

    def forward(self, image_tokens: Tensor) -> Tensor:
        for block in self.block_list:
            image_tokens = block(image_tokens)
        image_tokens = self.ln_final(image_tokens)
        return self.linear_projection(image_tokens)


class VisionTower(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        freeze_backbone: bool,
        embed_dim: int,
        num_head_blocks: int,
        head_blocks_block_drop_path: float,
        use_class_token: bool,
        use_patch_tokens: bool,
        patch_token_layer: int,
        patch_tokens_pooler_type: str,
        use_linear_projection: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.use_class_token = use_class_token
        self.use_patch_tokens = use_patch_tokens
        self.patch_token_layer = patch_token_layer
        self.patch_tokens_pooler_type = patch_tokens_pooler_type
        self.num_register_tokens = 0
        if hasattr(self.backbone, "num_register_tokens"):
            self.num_register_tokens = self.backbone.num_register_tokens
        elif hasattr(self.backbone, "n_storage_tokens"):
            self.num_register_tokens = self.backbone.n_storage_tokens
        backbone_out_dim = self.backbone.embed_dim
        self.head = VisionHead(
            backbone_out_dim,
            embed_dim,
            self.backbone.num_heads,
            num_head_blocks,
            head_blocks_block_drop_path,
            use_class_token,
            use_patch_tokens,
            use_linear_projection,
        )

    def init_weights(self):
        if not self.freeze_backbone:
            self.backbone.init_weights()
        self.head.init_weights()

    def get_backbone_features(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        tokens = self.backbone.get_intermediate_layers(
            images,
            n=self.patch_token_layer,
            return_class_token=True,
            return_register_tokens=True,
        )
        class_token = tokens[-1][1]
        patch_tokens = tokens[0][0]
        register_tokens = tokens[0][2]
        return class_token, patch_tokens, register_tokens

    def get_class_and_patch_tokens(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        class_token, patch_tokens, register_tokens = self.get_backbone_features(images)
        image_tokens = self.head(torch.cat([class_token.unsqueeze(1), register_tokens, patch_tokens], dim=1))
        class_token, patch_tokens = image_tokens[:, 0], image_tokens[:, self.num_register_tokens + 1 :]
        return class_token, patch_tokens

    def forward(self, images: Tensor) -> Tensor:
        class_token, patch_tokens = self.get_class_and_patch_tokens(images)
        features = []
        if self.use_class_token:
            features.append(class_token)
        if self.use_patch_tokens:
            if self.patch_tokens_pooler_type == "mean":
                features.append(torch.mean(patch_tokens, dim=1))
            elif self.patch_tokens_pooler_type == "max":
                features.append(torch.max(patch_tokens, dim=1).values)
            elif self.patch_tokens_pooler_type == "gem":
                power = 3
                eps = 1e-6
                patch_tokens_power = patch_tokens.clamp(min=eps).pow(power)
                features.append(torch.mean(patch_tokens_power, dim=1).pow(1 / power))
            else:
                raise ValueError(f"Unknown patch tokens pooler type: {self.patch_tokens_pooler_type}")
        return torch.cat(features, dim=-1)
