import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .vision_tower import VisionTower
from .text_tower import TextTower


@dataclass
class DinoTxtConfig:
    embed_dim: int
    vision_model_freeze_backbone: bool = True
    vision_model_train_img_size: int = 224
    vision_model_use_class_token: bool = True
    vision_model_use_patch_tokens: bool = False
    vision_model_num_head_blocks: int = 0
    vision_model_head_blocks_drop_path: float = 0.3
    vision_model_use_linear_projection: bool = False
    vision_model_patch_tokens_pooler_type: str = "mean"
    vision_model_patch_token_layer: int = 1  # which layer to take patch tokens from
    # 1 - last layer, 2 - second last layer, etc.
    text_model_freeze_backbone: bool = False
    text_model_num_head_blocks: int = 0
    text_model_head_blocks_is_causal: bool = False
    text_model_head_blocks_drop_prob: float = 0.0
    text_model_tokens_pooler_type: str = "first"
    text_model_use_linear_projection: bool = False
    init_logit_scale: float = math.log(1 / 0.07)
    init_logit_bias: Optional[float] = None
    freeze_logit_scale: bool = False


class DinoTxt(nn.Module):
    def __init__(
        self,
        model_config: DinoTxtConfig,
        vision_backbone: nn.Module,
        text_backbone: nn.Module,
    ):
        super().__init__()
        self.model_config = model_config
        self.visual_model = VisionTower(
            vision_backbone,
            model_config.vision_model_freeze_backbone,
            model_config.embed_dim,
            model_config.vision_model_num_head_blocks,
            model_config.vision_model_head_blocks_drop_path,
            model_config.vision_model_use_class_token,
            model_config.vision_model_use_patch_tokens,
            model_config.vision_model_patch_token_layer,
            model_config.vision_model_patch_tokens_pooler_type,
            model_config.vision_model_use_linear_projection,
        )
        self.text_model = TextTower(
            text_backbone,
            model_config.text_model_freeze_backbone,
            model_config.embed_dim,
            model_config.text_model_num_head_blocks,
            model_config.text_model_head_blocks_is_causal,
            model_config.text_model_head_blocks_drop_prob,
            model_config.text_model_tokens_pooler_type,
            model_config.text_model_use_linear_projection,
        )
        self.logit_scale = nn.Parameter(torch.ones(1) * model_config.init_logit_scale)
        if model_config.freeze_logit_scale:
            self.logit_scale.requires_grad = False

    def init_weights(self):
        self.visual_model.init_weights()
        self.text_model.init_weights()

    def get_visual_class_and_patch_tokens(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        return self.visual_model.get_class_and_patch_tokens(image)

    def encode_image(
        self,
        image: Tensor,
        normalize: bool = False,
    ) -> Tensor:
        """
        Encode an image into a vector descriptor containing both global and local features.

        Args:
            image (Tensor): Tensor of shape `(batch_size, rgb, height, width)`, normalized using ImageNet mean and std.
            normalize (bool, optional): Whether to normalize the output vectors. Default is False.
                Image features should always be normalized before comparing them with text features:
        Returns:
            Tensor: Tensor of shape `(batch_size, embed_dim)` containing the image features.
                The first half of the vector corresponds to the global features (class token),
                and the second half corresponds to the pooled patch features.
        """
        features = self.visual_model(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text: Tensor, normalize: bool = False) -> Tensor:
        """
        Encode a text input into a vector descriptor.

        Args:
            text (Tensor): Tensor of shape `(batch_size, seq_len)` containing token indices.
            normalize (bool, optional): Whether to normalize the output vectors. Default is False.
                Text features should be normalized before comparing them with image features:
        Returns:
            Tensor: Tensor of shape `(batch_size, embed_dim)` containing the text features.
                As a consequence of the training procedure, assume that the first half of the tensor corresponds
                to global image features and the second half to pooled patch features.
        """
        features = self.text_model(text)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image: Tensor, text: Tensor) -> Tuple[Tensor, Tensor]:
        text_features = self.encode_text(text, normalize=True)
        image_features = self.encode_image(image, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
        self,
        image: Tensor,
        text: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        text_features = self.encode_text(text, normalize=True)
        image_features = self.encode_image(image, normalize=True)
        return image_features, text_features, self.logit_scale.exp()
