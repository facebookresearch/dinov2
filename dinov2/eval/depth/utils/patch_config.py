# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


def patch_config(cfg, head, in_index, out_index, embed_dim):
    if type(in_index) == int:
        in_index = [in_index]
    if type(out_index) == int:
        out_index = [out_index]
    if head == "linear":
        decode_head = dict(
            type="BNHead",
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
        )

    else:
        decode_head = dict(
            type="DPTHead",
            in_channels=[embed_dim] * 4,
            channels=256,
            embed_dims=embed_dim,
            post_process_channels=[embed_dim // 2 ** (3 - i) for i in range(4)],
            readout_type="project",
        )
    for k in decode_head:
        cfg.model.decode_head[k] = decode_head[k]
    # cfg.model.decode_head = decode_head
    cfg.model.backbone.out_indices = out_index
    return cfg
