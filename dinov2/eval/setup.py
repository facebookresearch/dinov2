# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

from dinov2.models import build_model_from_cfg
import dinov2.utils.utils as dinov2_utils


def get_autocast_dtype(cfg):
    teacher_dtype_str = cfg.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(cfg, pretrained_weights):
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model


def setup_and_build_model(cfg) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    model = build_model_for_eval(cfg, cfg.pretrained_weights)
    autocast_dtype = get_autocast_dtype(cfg)
    return model, autocast_dtype
