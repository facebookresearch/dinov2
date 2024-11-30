import torch
from torch.nn import functional

import sys
sys.path.append('/home/li.yu/code/JupiterCVML/europa/base/src/europa')
from dl.network.nextvit_brt import _get_nextvit

class NextVitSmall(torch.nn.Module):
    """BRT Segmentation model with definition to make it a custom model supported."""
    def __init__(self, num_classes=197*1024) -> None:
        super().__init__()

        # define backbone
        self.backbone = _get_nextvit(
            model_size="small",
            frozen_stages=-1,
            norm_eval=False,
            with_extra_norm=True,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            in_channels=3,
        )

        # self.proj_head = torch.nn.Sequential(
        #     torch.nn.Linear(1024, num_classes),
        # )
        assert num_classes == 197 * 1024
        self.num_register_tokens = 1
        self.embed_dim = 1024
        self.proj_head = torch.nn.Linear(1024, num_classes)

    def forward_backbone(self, x, masks=None):
        y = self.backbone(x)
        y = functional.adaptive_avg_pool2d(y[-1], (1, 1))
        y = torch.flatten(y, 1)
        y = self.proj_head(y)

        n = y.shape[0]
        y_reshaped = y.reshape(n, 197, 1024)
        return {
            "x_norm_clstoken": y_reshaped[:, 0],  # teacher 128x1024
            "x_norm_regtokens": y_reshaped[:, 1 : self.num_register_tokens + 1],  # teacher 128x0x1024
            "x_norm_patchtokens": y_reshaped[:, self.num_register_tokens + 1 :],  # teacher 128x196x1024
            "x_prenorm": None,
            "masks": masks,
        }

    def forward(self, x, *args, masks=None, **kwargs):
        if isinstance(x, list):
            return [self.forward_backbone(_x, _masks) for _x, _masks in zip(x, masks)]
        return self.forward_backbone(x, masks)