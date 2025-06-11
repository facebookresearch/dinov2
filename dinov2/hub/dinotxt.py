import torch
import math

from .backbones import dinov2_vitl14_reg
from .utils import _DINOV2_BASE_URL


def dinov2_vitl14_reg4_dinotxt_tet1280d20h24l():
    from .text.dinotxt_model import DinoTxtConfig, DinoTxt
    from .text.dinov2_wrapper import DINOv2Wrapper
    from .text.text_transformer import TextTransformer

    dinotxt_config = DinoTxtConfig(
        embed_dim=2048,
        vision_model_freeze_backbone=True,
        vision_model_train_img_size=224,
        vision_model_use_class_token=True,
        vision_model_use_patch_tokens=True,
        vision_model_num_head_blocks=2,
        vision_model_head_blocks_drop_path=0.3,
        vision_model_use_linear_projection=False,
        vision_model_patch_tokens_pooler_type="mean",
        vision_model_patch_token_layer=1,  # which layer to take patch tokens from
        # 1 - last layer, 2 - second last layer, etc.
        text_model_freeze_backbone=False,
        text_model_num_head_blocks=0,
        text_model_head_blocks_is_causal=False,
        text_model_head_blocks_drop_prob=0.0,
        text_model_tokens_pooler_type="argmax",
        text_model_use_linear_projection=True,
        init_logit_scale=math.log(1 / 0.07),
        init_logit_bias=None,
        freeze_logit_scale=False,
    )
    vision_backbone = DINOv2Wrapper(dinov2_vitl14_reg())
    text_backbone = TextTransformer(
        context_length=77,
        vocab_size=49408,
        dim=1280,
        num_heads=20,
        num_layers=24,
        ffn_ratio=4,
        is_causal=True,
        ls_init_value=None,
        dropout_prob=0.0,
    )
    model = DinoTxt(dinotxt_config, vision_backbone, text_backbone)
    model.init_weights()
    model.visual_model.backbone = vision_backbone
    model.eval()

    visual_model_head_state_dict = torch.hub.load_state_dict_from_url(
        _DINOV2_BASE_URL + "/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
        map_location="cpu",
    )
    text_model_state_dict = torch.hub.load_state_dict_from_url(
        _DINOV2_BASE_URL + "/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth",
        map_location="cpu",
    )
    model.visual_model.head.load_state_dict(visual_model_head_state_dict, strict=True)
    model.text_model.load_state_dict(text_model_state_dict, strict=True)
    return model


def get_tokenizer():
    from .text.tokenizer import Tokenizer
    import requests
    from io import BytesIO

    url = _DINOV2_BASE_URL + "/thirdparty/bpe_simple_vocab_16e6.txt.gz"
    try:
        response = requests.get(url)
        response.raise_for_status()
        file_buf = BytesIO(response.content)
        return Tokenizer(vocab_path=file_buf)
    except Exception as e:
        raise FileNotFoundError(f"Failed to download file from url {url} with error last: {e}")
