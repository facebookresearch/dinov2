# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar

import torch
from torch.utils.data import Sampler

from .datasets import ImageNet, ImageNet22k, HuggingFaceDataset
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler
from .semisup_wrapper import SemiSupervisedWrapper, SemiSupervisedSampler


logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4
    SEMI_SUPERVISED = 5


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(
    image_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")

    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        key, value = token.split("=")
        kwargs[key] = value

    if name == "ImageNet":
        class_ = ImageNet
        # Only allow specific keys for ImageNet
        for key in kwargs:
            assert key in ("root", "extra", "split")
        if "split" in kwargs:
            kwargs["split"] = ImageNet.Split[kwargs["split"]]
    elif name == "ImageNet22k":
        class_ = ImageNet22k
        # Only allow specific keys for ImageNet22k
        for key in kwargs:
            assert key in ("root", "extra", "split")
    elif name == "HuggingFace":
        class_ = HuggingFaceDataset
        # HuggingFace datasets require at least a dataset_name
        if "dataset_name" not in kwargs:
            raise ValueError("HuggingFace dataset requires 'dataset_name' parameter")
        # Convert string values to appropriate types
        if "streaming" in kwargs:
            kwargs["streaming"] = kwargs["streaming"].lower() == "true"
        if "trust_remote_code" in kwargs:
            kwargs["trust_remote_code"] = kwargs["trust_remote_code"].lower() == "true"
        if "add_index" in kwargs:
            kwargs["add_index"] = kwargs["add_index"].lower() == "true"
    else:
        raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs


def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    """
    Creates a dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.

    Returns:
        The created dataset.
    """
    logger.info(f'using dataset: "{dataset_str}"')

    class_, kwargs = _parse_dataset_str(dataset_str)
    dataset = class_(transform=transform, target_transform=target_transform, **kwargs)

    logger.info(f"# of dataset samples: {len(dataset):,d}")

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)

    return dataset


def make_semisupervised_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    main_label_col: str = "label",
    supervised_proportion: float = 0.1,
    seed: int = 813846,
    labels_per_class: Optional[int] = None,
):
    """
    Creates a semi-supervised dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.
        main_label_col: Name of the main label column.
        supervised_proportion: Proportion of data to use as supervised (0-1).
        seed: Random seed for reproducibility.
        labels_per_class: If set, use this many labels per class instead of proportion.

    Returns:
        The created semi-supervised dataset.
    """
    logger.info(f'creating semi-supervised dataset from: "{dataset_str}"')
    logger.info(f"supervised_proportion: {supervised_proportion}")
    if labels_per_class is not None:
        logger.info(f"labels_per_class: {labels_per_class}")

    # Create the base dataset
    dataset = make_dataset(
        dataset_str=dataset_str,
        transform=transform,
        target_transform=target_transform,
    )

    # Wrap with semi-supervised functionality
    semisup_dataset = SemiSupervisedWrapper(
        dataset=dataset,
        main_label_col=main_label_col,
        supervised_proportion=supervised_proportion,
        seed=seed,
        labels_per_class=labels_per_class,
    )

    logger.info(f"# of supervised samples: {len(semisup_dataset.supervised_split):,d}")
    logger.info(f"# of total samples: {len(semisup_dataset):,d}")

    return semisup_dataset


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
    # SEMI_SUPERVISED sampler is handled in make_data_loader function
    # and should never reach this point

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
    supervised_per_batch: int = 0,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED, SEMI_SUPERVISED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
        supervised_per_batch: Number of supervised samples per batch (for semi-supervised training)
    """

    # Handle semi-supervised sampler specially
    if sampler_type == SamplerType.SEMI_SUPERVISED:
        logger.info("sampler: semi-supervised")

        sampler = SemiSupervisedSampler(
            dataset=dataset,
            batch_size=batch_size,
            supervised_per_batch=supervised_per_batch,
            seed=seed,
        )
    else:
        sampler = _make_sampler(
            dataset=dataset,
            type=sampler_type,
            shuffle=shuffle,
            seed=seed,
            size=sampler_size,
            advance=sampler_advance,
        )

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader


def make_dataset_from_config(
    data_config: dict,
    split: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    """
    Creates a dataset from a configuration dictionary.

    Args:
        data_config: Configuration dictionary containing dataset parameters
        split: Which split to use ('train' or 'val')
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.

    Returns:
        The created dataset.
    """

    # Get the split name from config
    split_name = data_config.get(f"{split}_split", split)

    # Create HuggingFace dataset with config parameters
    dataset = HuggingFaceDataset(
        dataset_name=data_config["dataset"],
        config_name=data_config.get("config_name"),
        split=split_name,
        img_col_name=data_config.get("img_col_name", "image"),
        label_col_names=data_config.get("label_col_names", ["label"]),
        transform=transform,
        target_transform=target_transform,
        cache_dir=data_config.get("cache_dir"),
        data_dir=data_config.get("data_dir"),
        trust_remote_code=data_config.get("trust_remote_code", True),
        streaming=data_config.get("streaming", False),
    )

    logger.info(
        f"Created dataset for split '{split_name}' with {len(dataset):,d} samples"
    )

    # Set attributes for compatibility
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)

    return dataset


def make_semisupervised_dataset_from_config(
    data_config: dict,
    split: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    """
    Creates a semi-supervised dataset from a configuration dictionary.

    Args:
        data_config: Configuration dictionary containing dataset parameters
        split: Which split to use ('train' or 'val')
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.

    Returns:
        The created semi-supervised dataset.
    """

    # First create the base dataset
    dataset = make_dataset_from_config(
        data_config=data_config,
        split=split,
        transform=transform,
        target_transform=target_transform,
    )

    # Get semi-supervised parameters from config
    semisup_config = data_config.get("semisupervised", {})
    main_label_col = semisup_config.get("main_label_col", "label")
    supervised_proportion = semisup_config.get("supervised_proportion", 0.1)
    seed = semisup_config.get("seed", 813846)
    labels_per_class = semisup_config.get("labels_per_class", None)

    logger.info(f"Creating semi-supervised dataset with:")
    logger.info(f"  main_label_col: {main_label_col}")
    logger.info(f"  supervised_proportion: {supervised_proportion}")
    if labels_per_class is not None:
        logger.info(f"  labels_per_class: {labels_per_class}")

    # Wrap with semi-supervised functionality
    semisup_dataset = SemiSupervisedWrapper(
        dataset=dataset,
        main_label_col=main_label_col,
        supervised_proportion=supervised_proportion,
        seed=seed,
        labels_per_class=labels_per_class,
    )

    logger.info(
        f"Created semi-supervised dataset with {len(semisup_dataset.supervised_split):,d} supervised samples out of {len(semisup_dataset):,d} total"
    )

    return semisup_dataset
