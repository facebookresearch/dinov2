# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
from datasets import load_dataset, Dataset
from aiohttp import ClientTimeout
from typing import Union, Optional, Dict, Any, List, Callable
import logging
from PIL import Image
import io

logger = logging.getLogger("dinov2")


class HuggingFaceDataset(torch.utils.data.Dataset):
    """Load a HuggingFace dataset.

    Parameters
    ----------
    dataset_name: str
        Name of the HuggingFace dataset to load.
    config_name: str, optional
        Configuration name for the dataset.
    split: str, optional
        Split of the dataset to load (e.g., 'train', 'validation', 'test').
    img_col_name: str, optional
        Name of the image column in the dataset. Defaults to 'image'.
    label_col_names: List[str], optional
        Names of the label columns in the dataset. Defaults to ['label'].
    data_dir: str, optional
        Path to the local data directory.
    cache_dir: str, optional
        Path to the cache directory.
    rename_columns: dict, optional
        A mapping of names from the HF dataset to what the dict should contain in this dataset.
        For example `{"image":"image", "label":"target"}`.
    remove_columns: list, optional
        List of column names to remove from the dataset.
    transform: callable, optional
        A transform to apply to images.
    target_transform: callable, optional
        A transform to apply to targets.
    transforms: callable, optional
        A transform to apply to both images and targets.
    add_index: bool
        Whether to add a key "index" with the datum index.
    trust_remote_code: bool
        Whether to trust remote code when loading the dataset.
    streaming: bool
        Whether to use streaming mode.
    **kwargs: dict
        Additional keyword arguments to pass to `datasets.load_dataset`.
    """

    def __init__(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,
        split: Optional[str] = None,
        img_col_name: str = "image",
        label_col_names: Optional[List[str]] = None,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        rename_columns: Optional[Dict[str, str]] = None,
        remove_columns: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        add_index: bool = False,
        trust_remote_code: bool = True,
        streaming: bool = False,
        **kwargs: Dict[str, Any],
    ):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.img_col_name = img_col_name
        self.label_col_names = label_col_names or ["label"]
        self.add_index = add_index
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        # Set up timeout for downloading
        timeout_period = 500000  # intentionally long enough so timeout will not ever occur
        storage_options = {"client_kwargs": {"timeout": ClientTimeout(total=timeout_period)}}

        # Load the dataset
        logger.info(f'Loading HuggingFace dataset: "{dataset_name}"')
        if config_name:
            logger.info(f'Using config: "{config_name}"')
        if split:
            logger.info(f'Using split: "{split}"')

        dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=split,
            data_dir=data_dir,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            streaming=streaming,
            storage_options=storage_options,
        )

        # Apply column operations
        if rename_columns is not None:
            logger.info(f"Renaming columns: {rename_columns}")
            dataset = dataset.rename_columns(rename_columns)

        if remove_columns is not None:
            logger.info(f"Removing columns: {remove_columns}")
            dataset = dataset.remove_columns(remove_columns)

        self.dataset = dataset

        # Only log length if dataset supports it (non-streaming)
        if hasattr(dataset, "__len__"):
            logger.info(f"# of dataset samples: {len(dataset):,d}")
        else:
            logger.info("Using streaming dataset (length unknown)")

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)  # type: ignore
        else:
            raise TypeError("Dataset does not support len() (likely streaming dataset)")

    def get_image_data(self, index: int) -> bytes:
        """Get image data as bytes for compatibility with existing decoders."""
        if not hasattr(self.dataset, "__getitem__"):
            raise TypeError("Dataset does not support indexing (likely streaming dataset)")

        sample = self.dataset[index]  # type: ignore

        # Handle different image formats and column names
        image = None

        # First try the configured image column name
        if self.img_col_name in sample:
            image = sample[self.img_col_name]
        else:
            # Fall back to common image column names
            for key in ["image", "img", "picture", "photo"]:
                if key in sample:
                    image = sample[key]
                    logger.warning(f"Using '{key}' column instead of configured '{self.img_col_name}'")
                    break

        if image is None:
            # Try to get available keys for debugging
            available_keys = "unknown"
            try:
                if hasattr(sample, "keys"):
                    available_keys = list(sample.keys())  # type: ignore
                elif isinstance(sample, dict):
                    available_keys = list(sample.keys())
            except:
                pass
            raise ValueError(
                f"Could not find image column '{self.img_col_name}' in sample at index {index}. Available keys: {available_keys}"
            )

        if isinstance(image, Image.Image):
            # Convert PIL Image to bytes
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            return buffer.getvalue()
        elif isinstance(image, bytes):
            return image
        elif isinstance(image, str):
            # If it's a path, read the file
            with open(image, "rb") as f:
                return f.read()

        raise ValueError(f"Unsupported image format: {type(image)}")

    def get_target(self, index: int) -> Any:
        """Get target for compatibility with existing code."""
        if not hasattr(self.dataset, "__getitem__"):
            raise TypeError("Dataset does not support indexing (likely streaming dataset)")

        sample = self.dataset[index]  # type: ignore

        # Try configured label column names first
        for key in self.label_col_names:
            if key in sample:
                return sample[key]

        # Fall back to common label column names
        for key in ["target", "label", "labels", "class", "y"]:
            if key in sample:
                logger.warning(f"Using '{key}' column instead of configured label columns {self.label_col_names}")
                return sample[key]

        # Return None if no target/label found
        return None

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Any:
        """Get a sample from the dataset.

        Parameters
        ----------
        idx: int or torch.Tensor
            Index to sample from the dataset.

        Returns
        -------
        Any
            The processed sample. Format depends on transforms configuration.
        """
        if isinstance(idx, torch.Tensor) and idx.dim() == 0:
            idx = int(idx.item())
        idx = int(idx)

        if not hasattr(self.dataset, "__getitem__"):
            raise TypeError("Dataset does not support indexing (likely streaming dataset)")

        sample = self.dataset[idx]  # type: ignore

        # Extract image and target
        # Handle both dict-like and Dataset objects
        image = None

        # First try the configured image column name
        if hasattr(sample, "get"):
            image = sample.get(self.img_col_name)  # type: ignore
        elif isinstance(sample, dict):
            image = sample.get(self.img_col_name)
        else:
            # For datasets that return feature objects
            try:
                image = sample[self.img_col_name]  # type: ignore
            except (KeyError, AttributeError):
                pass

        # If configured column not found, try common image column names
        if image is None:
            for key in ["image", "img", "picture", "photo"]:
                if hasattr(sample, "get"):
                    image = sample.get(key)  # type: ignore
                elif isinstance(sample, dict):
                    image = sample.get(key)
                else:
                    # For datasets that return feature objects
                    try:
                        image = sample[key]  # type: ignore
                    except (KeyError, AttributeError):
                        continue
                if image is not None:
                    if key != self.img_col_name:
                        logger.warning(f"Using '{key}' column instead of configured '{self.img_col_name}'")
                    break

        if image is None:
            raise ValueError(f"Could not find image column '{self.img_col_name}' in sample at index {idx}")

        target = self.get_target(idx)

        # Handle PIL Images
        if isinstance(image, Image.Image):
            pass  # Keep as PIL Image for transforms
        elif isinstance(image, bytes):
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            # If it's a path, load the image
            image = Image.open(image)
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")

        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None and target is not None:
                target = self.target_transform(target)

        # Add index if requested
        if self.add_index:
            # Return as tuple for compatibility with existing code
            if isinstance(image, tuple):
                # If transforms already returned a tuple, extend it
                return image + (idx,)
            else:
                return image, target, idx

        return image, target
