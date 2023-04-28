# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

# initialize a logger
logger = logging.getLogger("dinov2")


# define a class for data augmentation
class DataAugmentationDINO(object):
    def __init__(
            self,
            global_crops_scale,  # scaling factor for global crops
            local_crops_scale,  # scaling factor for local crops
            local_crops_number,  # number of local crops
            global_crops_size=224,  # size of global crops
            local_crops_size=96,  # size of local crops
    ):
        # store augmentation parameters
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        # log the augmentation parameters
        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # define geometric augmentation for global crops, consisting of a random resized crop and horizontal flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # define geometric augmentation for local crops, consisting of a random resized crop and horizontal flip
        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # define color distortions and blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        # define additional transformations for global crops
        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        # define additional transformation for local crops
        local_transfo_extra = GaussianBlur(p=0.5)

        # define normalization transformation
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        # define the complete transformations for global crops, including color distortions, blurring and normalization
        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])

        # define the complete transformations for local crops, including color distortions, blurring and normalization
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        # Add the two global crops to the output dictionary under the key 'global_crops'
        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        # Add the same two global crops to the output dictionary under the key 'global_crops_teacher'
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops: Apply local geometric transformations and color distortions and add the transformed crops to
        # the 'local_crops' list
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]

        # Add the list of local crops to the output dictionary under the key 'local_crops'
        output["local_crops"] = local_crops

        # Add an empty tuple to the output dictionary under the key 'offsets'
        output["offsets"] = ()

        # Return the output dictionary
        return output
