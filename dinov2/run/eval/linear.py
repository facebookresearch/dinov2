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


logger = logging.getLogger("dinov2")


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        from dinov2.eval.linear import main as linear_main

        self._setup_args()
        linear_main(self.cfg)

    def checkpoint(self):
        import submitit

        logger.info(f"Requeuing {self.cfg}")
        empty = type(self)(self.cfg)
        return submitit.helpers.DelayedSubmission(empty)

    def _setup_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        if "%j" in self.cfg.output_dir:
            self.cfg.output_dir = self.cfg.output_dir.replace("%j", str(job_env.job_id))
        logger.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        logger.info(f"Config: {self.cfg}")


@hydra.main(config_path="../../configs", config_name="ssl_default_config")
def main(cfg: DictConfig):
    setup_logging()
    evaluator = Evaluator(cfg)
    evaluator()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
