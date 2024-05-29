import logging
from omegaconf import OmegaConf
from pathlib import Path

from dinov2.train.ssl_meta_arch import SSLMetaArch

logger = logging.getLogger("dinov2")
cfg = OmegaConf.load(Path(__file__).parent / "config.yaml")

def test():

    model = SSLMetaArch(cfg)
    logger.info("Model: \n{}".format(model))


if __name__ == "__main__":
    test()
