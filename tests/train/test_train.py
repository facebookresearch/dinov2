import logging
import torch
from omegaconf import OmegaConf
from pathlib import Path

from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.train.train import do_train

logger = logging.getLogger("dinov2")
cfg = OmegaConf.load(Path(__file__).parent / "config.yaml")


def test():
    if torch.cuda.is_available():
        model = SSLMetaArch(cfg).to(torch.device("cuda"))
        model.prepare_for_distributed_training()

        logger.info("Model:\n {}".format(model))
        do_train(cfg, model, resume=False)
    else:
        print("Unable to assess the training test, as no cuda devices were found")


if __name__ == "__main__":
    test()
