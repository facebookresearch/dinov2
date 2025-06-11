import torch
from torch import nn, Tensor

from dinov2.layers import (
    CausalAttentionBlock,
)


class TextHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        block_drop_prob: float,
        is_causal: bool,
        use_linear_projection: bool,
    ):
        super().__init__()
        block_list = [nn.Identity()]
        self.ln_final = nn.Identity()
        if num_blocks > 0:
            block_list = [
                CausalAttentionBlock(
                    dim=input_dim,
                    num_heads=num_heads,
                    is_causal=is_causal,
                    dropout_prob=block_drop_prob,
                )
                for _ in range(num_blocks)
            ]
            self.ln_final = nn.LayerNorm(input_dim)
        self.block_list = nn.ModuleList(block_list)
        self.num_blocks = num_blocks
        self.linear_projection = nn.Identity()
        if input_dim != embed_dim or use_linear_projection:
            self.linear_projection = nn.Linear(input_dim, embed_dim, bias=False)

    def init_weights(self):
        if self.num_blocks > 0:
            for i in range(self.num_blocks):
                self.block_list[i].init_weights()
            self.ln_final.reset_parameters()
        if isinstance(self.linear_projection, nn.Linear):
            nn.init.normal_(self.linear_projection.weight, std=self.linear_projection.in_features**-0.5)

    def forward(self, text_tokens: Tensor) -> Tensor:
        for block in self.block_list:
            text_tokens = block(text_tokens)
        text_tokens = self.ln_final(text_tokens)
        return self.linear_projection(text_tokens)


class TextTower(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        freeze_backbone: bool,
        embed_dim: int,
        num_head_blocks: int,
        head_blocks_is_causal: bool,
        head_blocks_block_drop_prob: float,
        tokens_pooler_type: str,
        use_linear_projection: bool,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        backbone_out_dim = backbone.embed_dim
        self.backbone = backbone
        self.head = TextHead(
            backbone_out_dim,
            embed_dim,
            self.backbone.num_heads,
            num_head_blocks,
            head_blocks_block_drop_prob,
            head_blocks_is_causal,
            use_linear_projection,
        )
        self.tokens_pooler_type = tokens_pooler_type

    def init_weights(self):
        self.backbone.init_weights()
        self.head.init_weights()

    def forward(self, token_indices: Tensor) -> Tensor:
        text_tokens = self.backbone(token_indices)
        text_tokens = self.head(text_tokens)
        if self.tokens_pooler_type == "first":
            features = text_tokens[:, 0]
        elif self.tokens_pooler_type == "last":
            features = text_tokens[:, -1]
        elif self.tokens_pooler_type == "argmax":
            assert token_indices is not None
            features = text_tokens[torch.arange(text_tokens.shape[0]), token_indices.argmax(dim=-1)]
        else:
            raise ValueError(f"Unknown text tokens pooler type: {self.pooler_type}")
        return features
