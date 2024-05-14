import logging
import torch

from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.train.train import get_args_parser
from dinov2.utils.config import setup
from cell_similarity.training.train import do_train

logger = logging.getLogger("dinov2")

def test(args):

    cfg = setup(args)
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n {}".format(model))
    do_train(cfg, model, resume=False)

if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    test(args)
