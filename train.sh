#!/bin/bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 dinov2/run/train/train.py \
    --config-file dinov2/configs/train/vitl16_short.yaml \
    --output-dir test_experiment \
    train.dataset_path=ImageNet:split=TRAIN:root=/fast/AG_Kainmueller/plankton/data/ImageNetSubset/imagenette2:extra=/fast/AG_Kainmueller/plankton/data/ImageNetSubset/imagenette2/EXTRA
