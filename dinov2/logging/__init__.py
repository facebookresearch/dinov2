# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import functools
import logging
import os
import sys
from typing import Optional

import wandb

import dinov2.distributed as distributed
from .helpers import MetricLogger, SmoothedValue
import torch.distributed as tdist


# So that calling _configure_logger multiple times won't add many handlers
@functools.lru_cache()
def _configure_logger(
    name: Optional[str] = None,
    *,
    level: int = logging.DEBUG,
    output: Optional[str] = None,
):
    """
    Configure a logger.

    Adapted from Detectron2.

    Args:
        name: The name of the logger to configure.
        level: The logging level to use.
        output: A file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.

    Returns:
        The configured logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Loosely match Google glog format:
    #   [IWEF]yyyymmdd hh:mm:ss.uuuuuu threadid file:line] msg
    # but use a shorter timestamp and include the logger name:
    #   [IWEF]yyyymmdd hh:mm:ss logger threadid file:line] msg
    fmt_prefix = "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # stdout logging for main worker only
    if distributed.is_main_process():
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # file logging for all workers
    if output:
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs", "log.txt")

        if not distributed.is_main_process():
            filename = filename + ".rank{}".format(distributed.get_global_rank())

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_logging(
    output: Optional[str] = None,
    *,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
    capture_warnings: bool = True,
    args: Optional[dict] = None,
    do_eval: bool = False,
) -> None:
    """
    Setup logging.

    Args:
        output: A file name or a directory to save log files. If None, log
            files will not be saved. If output ends with ".txt" or ".log", it
            is assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name: The name of the logger to configure, by default the root logger.
        level: The logging level to use.
        capture_warnings: Whether warnings should be captured as logs.
    """
    print('distributed.is_main_process():', distributed.is_main_process())
    print('get_global_rank', distributed.get_global_rank())
    print('distributed.is_enabled()', distributed.is_enabled())
    if distributed.is_main_process():
        logging.captureWarnings(capture_warnings)
        _configure_logger(name, level=level, output=output)
        if args is not None:
            run_name = args.run_name
        else:
            run_name = ''

        args.output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(args.output_dir, exist_ok=True)
        print('Output dir: ', args.output_dir)

        if do_eval:
            project='dinov2_plankton_eval'
        else:
            project='dinov2_plankton'
        wandb.init(name=run_name, entity='kainmueller-lab', project=project, config=args, dir=output)
