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
from dinov2.eval.log_regression import main as log_regression_main


logger = logging.getLogger("dinov2")


@hydra.main(config_path="../../configs", config_name="ssl_default_config")
def main(cfg: DictConfig):
    setup_logging()

    log_regression_main(cfg)
    return 0


if __name__ == "__main__":
    main()
