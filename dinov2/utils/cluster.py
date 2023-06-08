# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ClusterType(Enum):
    OL = "olab"
    UV = "ultraviolet"


def _guess_cluster_type() -> ClusterType:
    uname = os.uname()
    if uname.nodename.startswith("a100-40") or uname.nodename.startswith("gpu"):
            # Linux kernel versions on RSC instances are standard ones but hostnames start with "rsc"
            return ClusterType.UV

    return ClusterType.OL


def get_cluster_type(cluster_type: Optional[ClusterType] = None) -> Optional[ClusterType]:
    if cluster_type is None:
        return _guess_cluster_type()

    return cluster_type


def get_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[Path]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None

    CHECKPOINT_DIRNAMES = {
        ClusterType.OL: "checkpoints",
        ClusterType.UV: "checkpoint",
    }
    return Path("/") / CHECKPOINT_DIRNAMES[cluster_type]


def get_user_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[Path]:
    checkpoint_path = get_checkpoint_path(cluster_type)
    if checkpoint_path is None:
        return None

    username = os.environ.get("USER")
    assert username is not None
    return checkpoint_path / username


def get_slurm_partition(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None

    SLURM_PARTITIONS = {
        ClusterType.OL: "oermannlab",
        ClusterType.UV: "a100_long",
    }
    return SLURM_PARTITIONS[cluster_type]


def get_slurm_executor_parameters(
    nodes: int, num_gpus_per_node: int, cluster_type: Optional[ClusterType] = None, **kwargs
) -> Dict[str, Any]:
    # create default parameters
    params = {
        "mem_gb": 256,  # Requests all memory on a node, see https://slurm.schedmd.com/sbatch.html
        "gpus_per_node": num_gpus_per_node,
        "tasks_per_node": num_gpus_per_node,  # one task per GPU
        "cpus_per_task": 8,
        "nodes": nodes,
        "slurm_partition": get_slurm_partition(cluster_type),
    }
    # apply cluster-specific adjustments
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type == ClusterType.UV:
        params["cpus_per_task"] = 8
        params["mem_gb"] = 192
    elif cluster_type == ClusterType.OL:
        params["cpus_per_task"] = 8
    # set additional parameters / apply overrides
    params.update(kwargs)
    return params
