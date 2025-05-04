# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random
import math


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None, grad_accum_steps: int = 1):
    # dtype = torch.half  # TODO: Remove

    def _collate_impl(sub_samples_list):
        n_global_crops = len(sub_samples_list[0][0]["global_crops"])
        n_local_crops = len(sub_samples_list[0][0]["local_crops"])

        collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in sub_samples_list])

        collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in sub_samples_list])

        B = len(collated_global_crops)
        N = n_tokens
        n_samples_masked = int(B * mask_probability)
        probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
        upperbound = 0
        masks_list = []
        for i in range(0, n_samples_masked):
            prob_min = probs[i]
            prob_max = probs[i + 1]
            masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
            upperbound += int(N * prob_max)
        for i in range(n_samples_masked, B):
            masks_list.append(torch.BoolTensor(mask_generator(0)))

        random.shuffle(masks_list)

        collated_masks = torch.stack(masks_list).flatten(1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

        masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

        return {
            "collated_global_crops": collated_global_crops.to(dtype),
            "collated_local_crops": collated_local_crops.to(dtype),
            "collated_masks": collated_masks,
            "mask_indices_list": mask_indices_list,
            "masks_weight": masks_weight,
            "upperbound": upperbound,
            "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        }

    if grad_accum_steps <= 1:
        return [_collate_impl(samples_list)]

    split_size = math.ceil(len(samples_list) / grad_accum_steps)
    shards = []
    for i in range(grad_accum_steps):
        s = i * split_size
        e = min((i + 1) * split_size, len(samples_list))
        if s >= e:
            break
        shards.append(_collate_impl(samples_list[s:e]))
    return shards
