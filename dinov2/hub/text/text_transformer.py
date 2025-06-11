from typing import Callable, Optional, Tuple

import torch
from torch import nn, Tensor


from dinov2.layers import CausalAttentionBlock


class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int,
        vocab_size: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_ratio: float,
        is_causal: bool,
        ls_init_value: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = dim
        self.num_heads = num_heads

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, dim))
        self.dropout = nn.Dropout(dropout_prob)
        self.num_layers = num_layers
        block_list = [
            CausalAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                ls_init_value=ls_init_value,
                is_causal=is_causal,
                act_layer=act_layer,
                norm_layer=norm_layer,
                dropout_prob=dropout_prob,
            )
            for _ in range(num_layers)
        ]
        self.blocks = nn.ModuleList(block_list)
        self.ln_final = norm_layer(dim)

    def init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        init_attn_std = self.embed_dim**-0.5
        init_proj_std = (self.embed_dim**-0.5) * ((2 * self.num_layers) ** -0.5)
        init_fc_std = (2 * self.embed_dim) ** -0.5
        for block in self.blocks:
            block.init_weights(init_attn_std, init_proj_std, init_fc_std)
        self.ln_final.reset_parameters()

    def forward(self, token_indices: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, N = token_indices.size()
        x = self.token_embedding(token_indices) + self.positional_embedding[:N]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return x
