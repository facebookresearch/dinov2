# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

from dinov2.eval.knn import get_args_parser as get_knn_args_parser
from dinov2.logging import setup_logging
from dinov2.run.submit import get_args_parser, submit_jobs


logger = logging.getLogger("dinov2")


class Evaluator:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        from dinov2.eval.knn import main as knn_main

        self._setup_args()
        knn_main(self.args)

    def checkpoint(self):
        import submitit

        logger.info(f"Requeuing {self.args}")
        empty = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty)

    def _setup_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = self.args.output_dir.replace("%j", str(job_env.job_id))
        logger.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        logger.info(f"Args: {self.args}")


def main():
    description = "Submitit launcher for DINOv2 k-NN evaluation"
    knn_args_parser = get_knn_args_parser(add_help=False)
    parents = [knn_args_parser]
    args_parser = get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()

    setup_logging()

    assert os.path.exists(args.config_file), "Configuration file does not exist!"
    submit_jobs(Evaluator, args, name="dinov2:knn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
