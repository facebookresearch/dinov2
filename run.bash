source .venv/bin/activate
PYTHONPATH=.
python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/train/vitl16_short.yaml \
    --output-dir output \
    train.dataset_path=ImageNet:split=TRAIN:root=datapath1:extra=datapath2 \