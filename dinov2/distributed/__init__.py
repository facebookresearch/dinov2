# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import random
import re
import socket
from typing import Dict, List

import torch
import torch.distributed as dist

global RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE

def is_enabled() -> bool:
    """
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()


def get_global_size() -> int:
    """
    Returns:
        The number of processes in the process group
    """
    return WORLD_SIZE if is_enabled() else 1


def get_global_rank() -> int:
    """
    Returns:
        The rank of the current process within the global process group.
    """
    #print('get_global_rank', RANK, dist.is_available() , dist.is_initialized())
    return RANK if is_enabled() else 0


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    return LOCAL_RANK if is_enabled() else 0
    

def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    return LOCAL_WORLD_SIZE if is_enabled() else 1


def is_main_process() -> bool:
    """
    Returns:
        True if the current process is the main one.
    """
    return get_global_rank() == 0


def _restrict_print_to_main_process() -> None:
    """
    This function disables printing when not in the main process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main_process() or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def _get_master_port(seed: int = 0) -> int:
    MIN_MASTER_PORT, MAX_MASTER_PORT = (20_000, 60_000)

    master_port_str = os.environ.get("MASTER_PORT")
    if master_port_str is None:
        rng = random.Random(seed)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)

    return int(master_port_str)


def _get_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # A "" host address means INADDR_ANY i.e. binding to all interfaces.
        # Note this is not compatible with IPv6.
        s.bind(("", 0))
        port = s.getsockname()[1]
        return port


_TORCH_DISTRIBUTED_ENV_VARS = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
)


def _parse_slurm_node_list(s: str) -> List[str]:
    nodes = []
    # Extract "hostname", "hostname[1-2,3,4-5]," substrings
    p = re.compile(r"(([^\[]+)(?:\[([^\]]+)\])?),?")
    for m in p.finditer(s):
        prefix, suffixes = s[m.start(2) : m.end(2)], s[m.start(3) : m.end(3)]
        for suffix in suffixes.split(","):
            span = suffix.split("-")
            if len(span) == 1:
                nodes.append(prefix + suffix)
            else:
                width = len(span[0])
                start, end = int(span[0]), int(span[1]) + 1
                nodes.extend([prefix + f"{i:0{width}}" for i in range(start, end)])
    return nodes


def _check_env_variable(key: str, new_value: str):
    # Only check for difference with preset environment variables
    if key in os.environ and os.environ[key] != new_value:
        raise RuntimeError(f"Cannot export environment variables as {key} is already set")


class _TorchDistributedEnvironment:
    def __init__(self):

        dist.init_process_group(backend="nccl")
        dist.barrier()

        global RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE
        print('torch.cuda.device_count()', torch.cuda.device_count())
        print('torch.distributed.get_rank(group=None)', dist.get_rank(group=None))
        print('torch.distributed.get_world_size(group=None)', dist.get_world_size(group=None))
        RANK = dist.get_rank(group=None)
        WORLD_SIZE = dist.get_world_size(group=None)
        LOCAL_WORLD_SIZE = torch.cuda.device_count()
        LOCAL_RANK = RANK % LOCAL_WORLD_SIZE
        # or if num nodes in env vars
        # LOCAL_WORLD_SIZE = WORLD_SIZE // node_count

        self.master_addr = "127.0.0.1"
        self.master_port = 0
        self.rank = RANK
        self.world_size = WORLD_SIZE
        self.local_rank = LOCAL_RANK
        self.local_world_size = LOCAL_WORLD_SIZE


    # Slurm job created with sbatch, submitit, etc...
    def _set_from_slurm_env(self):
        # logger.info("Initialization from Slurm environment")
        job_id = int(os.environ["SLURM_JOB_ID"])
        nodes = _parse_slurm_node_list(os.environ["SLURM_JOB_NODELIST"])

        self.master_addr = nodes[0]
        self.master_port = _get_master_port(seed=job_id)


    # Single node job with preset environment (i.e. torchrun)
    def _set_from_preset_env(self):
        # logger.info("Initialization from preset environment")
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]

    # Single node and GPU job (i.e. local script run)
    def _set_from_local(self):
        # logger.info("Initialization from local")
        self.master_addr = "127.0.0.1"
        self.master_port = _get_available_port()
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.local_world_size = 1

    def export(self, *, overwrite: bool) -> "_TorchDistributedEnvironment":
        # See the "Environment variable initialization" section from
        # https://pytorch.org/docs/stable/distributed.html for the complete list of
        # environment variables required for the env:// initialization method.
        env_vars = {
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
        }
        if not overwrite:
            for k, v in env_vars.items():
                _check_env_variable(k, v)

        os.environ.update(env_vars)
        return self


def enable(*, set_cuda_current_device: bool = True, overwrite: bool = False, allow_nccl_timeout: bool = False):
    """Enable distributed mode

    Args:
        set_cuda_current_device: If True, call torch.cuda.set_device() to set the
            current PyTorch CUDA device to the one matching the local rank.
        overwrite: If True, overwrites already set variables. Else fails.
    """

    torch_env = _TorchDistributedEnvironment()
    torch_env.export(overwrite=overwrite)

    if set_cuda_current_device:
        torch.cuda.set_device(torch_env.local_rank)

    if allow_nccl_timeout:
        # This allows to use torch distributed timeout in a NCCL backend
        key, value = "NCCL_ASYNC_ERROR_HANDLING", "1"
        if not overwrite:
            _check_env_variable(key, value)
        os.environ[key] = value

    print('torch_is_enabled', is_enabled())
    print('get_local_rank', get_local_rank())
    print('get_global_rank', get_global_rank())
    print('get_global_size', get_global_size())
    print('get_local_size', get_local_size())

    # Finalize setup
    _restrict_print_to_main_process()
