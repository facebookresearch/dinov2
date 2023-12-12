#!/bin/bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 dinov2/run/train/train.py --config-file dinov2/configs/train/whoi.yaml --output-dir /fast/AG_Kainmueller/plankton/test_experiment
