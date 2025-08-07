import math
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist


class SemiSupervisedWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        main_label_col: str,
        supervised_proportion: float = 0.1,
        seed: int = 813846,
        labels_per_class: Optional[int] = None,
    ):
        assert 0 <= supervised_proportion <= 1, "supervised_proportion must be in [0, 1]"

        self.dataset = dataset
        self.supervised_proportion = supervised_proportion
        self.labels_per_class = labels_per_class

        g = torch.Generator()
        g.manual_seed(seed)

        # Resolve dataset object for HuggingFace compatibility
        if hasattr(self.dataset, "dataset"):
            dataset_obj = self.dataset.dataset
        else:
            dataset_obj = self.dataset

        labels = torch.tensor(dataset_obj[main_label_col])

        if labels_per_class is not None and labels_per_class > 0:
            print(f"Label per class has been set to {labels_per_class}, skipping supervised_proportion!")

            classes = dataset_obj.unique(main_label_col)
            classes = [c for c in classes if c != -1]  # remove unlabeled

            supervised_split = torch.empty((len(classes), labels_per_class), dtype=torch.long)

            for pt, c in enumerate(classes):
                idxs = torch.nonzero(labels == c, as_tuple=False).flatten()
                if len(idxs) < labels_per_class:
                    raise ValueError(
                        f"Not enough samples for class {c} got {len(idxs)} but need {labels_per_class}"
                    )
                selected = torch.randperm(len(idxs), generator=g)[:labels_per_class]
                supervised_split[pt] = idxs[selected]

            supervised_split = supervised_split.flatten()

        else:
            valid_idx = torch.nonzero(labels != -1, as_tuple=False).flatten()
            num_supervised_samples = int(len(valid_idx) * self.supervised_proportion)
            split_idx = torch.randperm(len(valid_idx), generator=g)[:num_supervised_samples]
            supervised_split = valid_idx[split_idx]

        self.supervised_split = supervised_split.tolist()

        # Create binary supervision flag list
        is_sup = torch.zeros(len(self.dataset), dtype=torch.bool)
        is_sup[supervised_split] = True

        # Try to add column if possible (HuggingFace-style)
        if hasattr(self.dataset, "dataset"):
            self.dataset.dataset = self.dataset.dataset.add_column("is_supervised", is_sup.tolist())

        # Always keep local reference
        self.is_supervised = is_sup.tolist()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index], self.is_supervised[index]


class SemiSupervisedSampler(torch.utils.data.DistributedSampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        supervised_per_batch: int = 0,
        seed: int = 0,
    ):
        self.dataset_len = len(dataset)

        indices = np.arange(len(dataset))
        self.num_batches = len(dataset) // batch_size
        self.sup_indices = np.array(dataset.supervised_split)
        self.unsup_indices = indices[~np.isin(indices, self.sup_indices)]
        self.batch_size = batch_size
        self.sup_ratio = dataset.supervised_proportion

        # Linearly scale supervision, with minimum threshold
        self.sup_per_batch = max(supervised_per_batch, int(batch_size * self.sup_ratio))
        print(f"👷🏼 Supervised samples per batch set to {self.sup_per_batch}")

        self.seed = seed
        self.epoch = 0

        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            if self.rank >= self.num_replicas or self.rank < 0:
                raise ValueError(f"Invalid rank {self.rank}, should be in [0, {self.num_replicas - 1}]")
        else:
            self.num_replicas = 1
            self.rank = 0

        if self.dataset_len % self.num_replicas != 0:
            self.num_samples = math.ceil((self.dataset_len - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = self.dataset_len // self.num_replicas

    def __len__(self):
        return self.num_batches * self.batch_size

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)

        num_sup = len(self.sup_indices)
        num_unsup = len(self.unsup_indices)

        total_sup = self.num_batches * self.sup_per_batch
        total_unsup = self.num_batches * self.batch_size - total_sup

        if num_sup == 0 and self.sup_per_batch > 0:
            raise ValueError("No supervised samples available, but supervised_per_batch > 0")

        choose = np.random.choice

        sup = choose(
            self.sup_indices,
            size=(self.num_batches, self.sup_per_batch),
            replace=num_sup < total_sup,
        )
        sup = torch.from_numpy(sup)

        if len(self.unsup_indices) > 0:
            unsup = choose(
                self.unsup_indices,
                size=(self.num_batches, self.batch_size - self.sup_per_batch),
                replace=num_unsup < total_unsup,
            )
            unsup = torch.from_numpy(unsup)
            overall_slice = torch.cat([sup, unsup], 1).flatten()
        else:
            overall_slice = sup.flatten()

        rank_slice = overall_slice[
            self.rank * self.num_samples: (self.rank + 1) * self.num_samples
        ]
        yield from rank_slice.tolist()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
