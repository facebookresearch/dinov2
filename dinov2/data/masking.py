# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np


class MaskingGenerator:
    def __init__(
            self,
            input_size,
            num_masking_patches=None,
            min_num_patches=4,
            max_num_patches=None,
            min_aspect=0.3,
            max_aspect=None,
    ):
        # If input_size is a single integer, create a square input size
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        # Calculate the total number of patches
        self.num_patches = self.height * self.width

        # Number of patches to mask out
        self.num_masking_patches = num_masking_patches

        # Minimum and maximum number of patches to mask out
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        # Calculate the minimum and maximum aspect ratios of the masked patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        # String representation of the generator instance
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        # Returns the input shape
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        # Masks out patches in the input mask until the specified number of patches has been masked
        delta = 0
        for _ in range(10):
            # Select a random area to mask
            target_area = random.uniform(self.min_num_patches, max_mask_patches)

            # Select a random aspect ratio for the masked area
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            # Check if the masked area is smaller than the input size
            if w < self.width and h < self.height:
                # Select a random location to mask
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                # Count the number of already masked patches in the selected area
                num_masked = mask[top: top + h, left: left + w].sum()

                # If there is overlap between the masked area and already masked patches,
                # mask out the non-masked patches in the selected area
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                # Stop masking if the specified number of patches has been masked
                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        # Returns a binary mask where 1 indicates a masked patch
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            # call _mask method to add masked patches to the mask
            delta = self._mask(mask, max_mask_patches)

            # if delta is 0, meaning no more patches can be masked, break the loop
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask
