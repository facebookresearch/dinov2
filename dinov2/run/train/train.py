# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import sys

from dinov2.train import get_args_parser as get_train_args_parser
from dinov2.run.submit import get_args_parser, submit_jobs
from dinov2.train import main as train_main



logger = logging.getLogger("dinov2")

def main():
    description = "Submitit launcher for DINOv2 training"
    train_args_parser = get_train_args_parser(add_help=False)
    parents = [train_args_parser]
    args_parser = get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()

    if 'JOB_ID' in os.environ.keys():
        job_id = os.environ['JOB_ID']
    else:
        job_id = 0
    args.output_dir = args.output_dir.replace("%j", str(job_id))

    assert os.path.exists(args.config_file), "Configuration file does not exist!"
    train_main(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
