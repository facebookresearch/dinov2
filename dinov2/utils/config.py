# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import logging
import os

from omegaconf import OmegaConf

import dinov2.distributed as distributed
from dinov2.logging import setup_logging
from dinov2.utils import utils


logger = logging.getLogger("dinov2")


def apply_scaling_rules_to_cfg(cfg):
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        base_lr = cfg.optim.base_lr
        cfg.optim.lr = base_lr
        cfg.optim.lr *= math.sqrt(
            cfg.train.batch_size_per_gpu * distributed.get_global_size() / 1024.0
        )
        logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def default_setup(cfg):
    distributed.enable(overwrite=True)
    seed = getattr(cfg, "seed", 0)
    rank = distributed.get_global_rank()
    setup_logging(output=cfg.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")
    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(cfg.items())))
    return cfg


def setup(cfg):
    """
    Create configs and perform basic setups.
    """
    cfg = apply_scaling_rules_to_cfg(cfg)
    cfg = default_setup(cfg)
    return cfg
