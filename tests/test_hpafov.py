# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC licence,
# found in the LICENSE_CELL_DINO_CODE file in the root directory of this source tree.

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from dinov2.data import make_dataset
from dinov2.data.datasets import HPAFoV


class HPAFoVSeparateChannelsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = [
            np.broadcast_to(np.array(values, dtype=np.uint8), (3, 2, 4)).copy()
            for values in ((10, 20, 30, 40), (50, 60, 70, 80))
        ]
        for directory in ("normalized_data", "512_whole_images"):
            image_directory = self.root / directory
            image_directory.mkdir()
            for index, image in enumerate(self.images):
                Image.fromarray(image).save(image_directory / f"image{index}.png")
        with (self.root / "whole_images_names.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["ID"])
            writer.writerows([[f"image{index}"] for index in range(2)])
        for filename in ("whole_images_512_train.csv", "whole_images_512_test.csv"):
            with (self.root / filename).open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["cell_type", "file"])
                writer.writerows([["BJ", "image0.png"], ["LHCN-M2", "image1.png"]])

    def make_separate_dataset(self, split="SSL"):
        return make_dataset(
            dataset_str=f"HPAFoV:root={self.root}:split={split}:mode=CELL_TYPE:wildcard=SEPARATE_CHANNELS"
        )

    def assert_expanded(self, dataset):
        self.assertEqual(len(dataset), 8)
        self.assertEqual(dataset.get_targets().tolist(), [0, 1] * 4)
        for index in range(8):
            self.assertEqual(Path(dataset.get_image_relpath(index)).name, f"image{index % 2}.png")
            self.assertEqual(dataset.get_target(index), index % 2)

    def test_string_wildcard_expands_each_split(self):
        for split in ("SSL", "TRAIN", "VAL"):
            with self.subTest(split=split):
                self.assert_expanded(self.make_separate_dataset(split))

    def test_enum_wildcard_expands_samples(self):
        dataset = HPAFoV(root=str(self.root), split=HPAFoV.Split.SSL, wildcard=HPAFoV.WildCard.SEPARATECHANNELS)
        self.assert_expanded(dataset)

    def test_every_channel_decodes_in_sample_order(self):
        dataset = self.make_separate_dataset()
        self.assertEqual(len(dataset), 8)
        for index in range(8):
            with self.subTest(index=index):
                image, target = dataset[index]
                expected = torch.tensor(self.images[index % 2][:, :, index // 2]).unsqueeze(0).float()
                torch.testing.assert_close(image, expected)
                self.assertEqual(target, index % 2)

    def test_empty_dataset_does_not_crash(self):
        with (self.root / "whole_images_names.csv").open("w", newline="") as output:
            csv.writer(output).writerow(["ID"])
        for wildcard in (HPAFoV.WildCard.SEPARATECHANNELS, "SEPARATE_CHANNELS"):
            with self.subTest(wildcard=wildcard):
                dataset = HPAFoV(root=str(self.root), split="SSL", wildcard=wildcard)
                self.assertEqual(len(dataset), 0)

    def test_full_image_mode_preserves_all_channels(self):
        for wildcard in (HPAFoV.WildCard.NONE, "NONE"):
            with self.subTest(wildcard=wildcard):
                dataset = HPAFoV(root=str(self.root), split="SSL", wildcard=wildcard)
                self.assertEqual(len(dataset), 2)
                for index in range(2):
                    image, target = dataset[index]
                    torch.testing.assert_close(image, torch.tensor(self.images[index]).permute(2, 0, 1).float())
                    self.assertEqual(target, index)


if __name__ == "__main__":
    unittest.main()
