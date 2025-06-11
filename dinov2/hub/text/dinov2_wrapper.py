from typing import Sequence

import torch


class DINOv2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.num_register_tokens = model.num_register_tokens

    # Same as the original forward, but assert is_training and rename x_norm_regtokens -> x_storage_tokens
    def forward(self, img, is_training: bool):
        assert is_training
        H, W = img.shape[-2:]
        P = self.model.patch_size
        x_dict = self.model(img, is_training=True)
        x_dict["h"] = h = H // P
        x_dict["w"] = w = W // P
        assert x_dict["x_norm_patchtokens"].shape[-2] == h * w
        return x_dict

    # Same as the original get_intermediate_layers, but allow returining extra tokens (registers)
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | Sequence[int] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_register_tokens: bool = False,
        norm=True,
    ) -> tuple[torch.Tensor] | tuple[tuple[torch.Tensor, ...], ...]:
        if self.model.chunked_blocks:
            outputs = self.model._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self.model._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.model.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        register_tokens = [out[:, 1 : 1 + self.model.num_register_tokens] for out in outputs]
        outputs = [out[:, 1 + self.model.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.model.patch_size, w // self.model.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if not return_class_token and not return_register_tokens:
            return tuple(outputs)
        if return_class_token and not return_register_tokens:
            return tuple(zip(outputs, class_tokens))
        if not return_class_token and return_register_tokens:
            return tuple(zip(outputs, register_tokens))
        return tuple(zip(outputs, class_tokens, register_tokens))
