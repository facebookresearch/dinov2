import logging
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.train.train import get_args_parser
from dinov2.utils.config import setup

logger = logging.getLogger("dinov2")

def test(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg)
    logger.info("Model: \n{}".format(model))

if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    test(args)
