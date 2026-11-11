# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC licence,
# found in the LICENSE_CELL_DINO_CODE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import numpy as np

from ..extended import ExtendedVisionDataset
from ..decoders import DecoderType

logger = logging.getLogger("dinov2")

CELL_TYPE = [
    "BJ",  # 1
    "LHCN-M2",
    "RH-30",
    "SH-SY5Y",
    "U-2 OS",  # 5
    "ASC TERT1",
    "HaCaT",
    "A-431",
    "U-251 MG",
    "HEK 293",  # 10
    "A549",
    "RT4",
    "HeLa",
    "MCF7",
    "PC-3",  # 15
    "hTERT-RPE1",
    "SK-MEL-30",
    "EFO-21",
    "AF22",
    "HEL",  # 20
    "Hep G2",
    "HUVEC TERT2",
    "THP-1",
    "CACO-2",
    "JURKAT",  # 25
    "RPTEC TERT1",
    "SuSa",
    "REH",
    "HDLM-2",
    "K-562",  # 30
    "hTCEpi",
    "NB-4",
    "HAP1",
    "OE19",
    "SiHa",  # 35
]

PROTEIN_LOCALIZATION = [  # matches https://www.kaggle.com/c/human-protein-atlas-image-classification/data
    "nucleoplasm",
    "nuclear membrane",
    "nucleoli",
    "nucleoli fibrillar center",
    "nuclear speckles",  # 5
    "nuclear bodies",
    "endoplasmic reticulum",
    "golgi apparatus",
    "peroxisomes",
    "endosomes",  # 10
    "lysosomes",
    "intermediate filaments",
    "actin filaments",
    "focal adhesion sites",
    "microtubules",  # 15
    "microtubule ends",
    "cytokinetic bridge",
    "mitotic spindle",
    "microtubule organizing center",
    "centrosome",  # 20
    "lipid droplets",
    "plasma membrane",
    "cell junctions",
    "mitochondria",
    "aggresome",  # 25
    "cytosol",
    "cytoplasmic bodies",
    "rods & rings",
]


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    SSL = "ssl"


def get_csv_fpath(split):
    """
    Path to data relative to root
    """
    if split == _Split.TRAIN.value.upper() or split == _Split.TRAIN or split == "TRAIN":
        return "whole_images_512_train.csv"
    elif split == _Split.VAL.value.upper() or split == _Split.VAL or split == "VAL":
        return "whole_images_512_test.csv"


class _WildCard(Enum):
    NONE = "none"
    SEPARATECHANNELS = "separate_channels"  # each channel from each image is treated as an independent sample, overrides chosen channel configuration


class _Mode(Enum):
    """
    Targets:
    - ALL: tuple, (one hot encoding of multilabel protein localization, categorical encoding of cell type)
    - PROTEIN_LOCALIZATION: one hot encoding of multilabel protein localization
    - CELL_TYPE: categorical encoding of cell type
    """

    ALL = "all"
    PROTEIN_LOCALIZATION = "protein_localization"
    CELL_TYPE = "cell_type"

    @property
    def nb_labels(self):
        if self == _Mode.CELL_TYPE:
            return len(CELL_TYPE)
        elif self == _Mode.PROTEIN_LOCALIZATION:
            return len(PROTEIN_LOCALIZATION)
        else:
            return None


def _list_images_from_csv(img_path, csv_path):
    L = []
    with open(csv_path) as filename:
        reader = csv.DictReader(filename)
        for row in reader:
            L.append(os.path.join(img_path, row["ID"] + ".png"))
    return L


def _load_file_names_and_labels_ssl(
    root: str,
) -> Tuple[List[str], List[Any]]:

    curr_img_path = os.path.join(root, "normalized_data")
    csv_train_ssl = os.path.join(root, "whole_images_names.csv")
    image_paths = _list_images_from_csv(curr_img_path, csv_train_ssl)
    labels = [i for i in range(len(image_paths))]
    return image_paths, labels


def _load_file_names_and_labels(
    root: str,
    split: _Split,
    mode: _Mode,
) -> Tuple[List[str], List[Any], np.ndarray]:

    data_path = os.path.join(root, "512_whole_images")
    csv_fpath = os.path.join(root, get_csv_fpath(split))

    image_paths = []
    labels = []

    with open(csv_fpath) as filename:
        reader = csv.DictReader(filename)
        for row in reader:

            add_sample = True
            if mode != _Mode.PROTEIN_LOCALIZATION.value.upper():
                # categorical
                if row["cell_type"] in CELL_TYPE:
                    cell_type = CELL_TYPE.index(row["cell_type"])
                else:
                    cell_type = np.nan

            if mode != _Mode.CELL_TYPE.value.upper():
                # one hot encoding
                prot_loc = np.zeros(len(PROTEIN_LOCALIZATION), dtype=np.int_)
                for k in range(len(PROTEIN_LOCALIZATION)):
                    if row[PROTEIN_LOCALIZATION[k]] == "True":
                        prot_loc[k] = 1
                if prot_loc.max() < 0.5:
                    add_sample = False

            if add_sample:
                if mode == _Mode.PROTEIN_LOCALIZATION.value.upper():
                    labels.append(prot_loc)
                elif mode == _Mode.CELL_TYPE.value.upper():
                    labels.append(cell_type)
                else:
                    labels.append({"prot_loc": prot_loc, "cell_type": cell_type})

                candidate_path = os.path.join(data_path, row["file"].split("/")[-1])
                if os.path.exists(candidate_path):
                    image_paths.append(candidate_path)
                else:
                    candidate_path = os.path.join(
                        data_path, row["file"].split("/")[-1].split(".")[0] + ".tiff"
                    )  # _blue.png") # some images on the normalized_data folder have a _blue suffix on their names
                    if os.path.exists(candidate_path):
                        image_paths.append(candidate_path)
                    else:
                        raise FileNotFoundError(f"File {candidate_path} not found.")

        return image_paths, labels


class HPAFoV(ExtendedVisionDataset):
    Split = Union[_Split]
    Mode = Union[_Mode]
    WildCard = Union[_WildCard]

    def __init__(
        self,
        *,
        split: "HPAFoV.Split" = _Split.TRAIN,
        mode: "HPAFoV.Mode" = _Mode.ALL,
        wildcard: "HPAFoV.WildCard" = _WildCard.NONE,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder_type: DecoderType = DecoderType.ChannelSelectDecoder,
        image_decoder_params: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        super().__init__(
            root,
            transforms,
            transform,
            target_transform,
            image_decoder_type=image_decoder_type,
            image_decoder_params={
                "select_channel": True
                if wildcard == _WildCard.SEPARATECHANNELS or wildcard == "SEPARATE_CHANNELS"
                else False
            },
            **kwargs,
        )
        self.mode = mode
        self.split = split
        self.root = root
        self.wildcard = wildcard
        self.channel_adaptive = True
        if split == _Split.SSL.value.upper() or split == _Split.SSL or split == "SSL":
            self._image_paths, self._labels = _load_file_names_and_labels_ssl(root)
        else:
            self._image_paths, self._labels = _load_file_names_and_labels(root, self.split, self.mode)

        self._channels = np.repeat(np.array([[0, 1, 2, 3]]), len(self._image_paths), axis=0).tolist()

        if self._channels and self.wildcard in (_WildCard.SEPARATECHANNELS, _WildCard.SEPARATECHANNELS.value.upper()):
            image_paths, labels, channels = self._image_paths, self._labels, self._channels
            channels = np.array(channels)
            # separate and stack the columns of the channels array
            C = channels.shape[1]
            channels = np.concatenate([channels[:, i] for i in range(C)])
            self._channels = np.expand_dims(channels, 1).tolist()
            self._image_paths = image_paths * C
            self._labels = labels * C

    def get_image_relpath(self, index: int) -> str:
        return self._image_paths[index]

    def get_image_data(self, index: int) -> bytes:
        image_relpath = self.get_image_relpath(index)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        if self.channel_adaptive:
            channels = self._channels[index]
            return image_data + bytes(channels)
        else:
            return image_data

    def get_target(self, index: int) -> Any:
        return self._labels[index]

    def get_targets(self) -> np.ndarray:
        return np.array(self._labels)

    def __len__(self) -> int:
        return len(self._image_paths)
