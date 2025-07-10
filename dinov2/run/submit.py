# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

import submitit

from dinov2.utils.cluster import (
    get_slurm_executor_parameters,
    get_slurm_partition,
    get_user_checkpoint_path,
)

logger = logging.getLogger("dinov2")


def get_shared_folder() -> Path:
    user_checkpoint_path = get_user_checkpoint_path()
    if user_checkpoint_path is None:
        raise RuntimeError("Path to user checkpoint cannot be determined")
    path = user_checkpoint_path / "experiments"
    path.mkdir(exist_ok=True)
    return path


def submit_jobs(task_class, cfg, name: str):
    if not cfg.output_dir:
        cfg.output_dir = str(get_shared_folder() / "%j")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=cfg.output_dir, slurm_max_num_timeout=30)

    kwargs = {}
    if getattr(cfg, "use_volta32", False):
        kwargs["slurm_constraint"] = "volta32gb"
    if getattr(cfg, "comment", ""):
        kwargs["slurm_comment"] = cfg.comment
    if getattr(cfg, "exclude", ""):
        kwargs["slurm_exclude"] = cfg.exclude

    executor_params = get_slurm_executor_parameters(
        nodes=cfg.nodes,
        num_gpus_per_node=cfg.ngpus,
        timeout_min=cfg.timeout,  # max is 60 * 72
        slurm_signal_delay_s=120,
        slurm_partition=cfg.partition,
        **kwargs,
    )
    executor.update_parameters(name=name, **executor_params)

    task = task_class(cfg)
    job = executor.submit(task)

    logger.info(f"Submitted job_id: {job.job_id}")
    str_output_dir = os.path.abspath(cfg.output_dir).replace("%j", str(job.job_id))
    logger.info(f"Logs and checkpoints will be saved at: {str_output_dir}")
