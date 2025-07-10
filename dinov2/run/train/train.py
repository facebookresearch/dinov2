# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import hydra
from omegaconf import DictConfig

from dinov2.logging import setup_logging
from dinov2.train import main as train_main


logger = logging.getLogger("dinov2")


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        self._setup_args()
        train_main(self.cfg)

    def checkpoint(self):
        import submitit

        logger.info(f"Requeuing {self.cfg}")
        empty = type(self)(self.cfg)
        return submitit.helpers.DelayedSubmission(empty)

    def _setup_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        # Example: update output_dir with job_id if needed
        if "%j" in self.cfg.train.output_dir:
            self.cfg.train.output_dir = self.cfg.train.output_dir.replace("%j", str(job_env.job_id))
        logger.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        logger.info(f"Config: {self.cfg}")


@hydra.main(config_path="../../configs", config_name="ssl_default_config")
def main(cfg: DictConfig):
    setup_logging()
    # Optionally check config file existence or other assertions here
    # Launch training job(s)
    trainer = Trainer(cfg)
    trainer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
