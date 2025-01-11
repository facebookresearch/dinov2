from typing import List, Optional
import torch.nn as nn
from .base import BaseModel
from .resnet import ResNet, BasicStem, make_resnet_stages
from torchvision import models
class   ResNetWrapper(BaseModel):
    def __init__(
        self,
        depth: int = 50,
        out_features: Optional[List[str]] = None,
        freeze_at: int = 0,
        norm_type: str = 'BN',
    ):
        super().__init__()
        
        # Default output features if none specified
        if out_features is None:
            out_features = ['res2', 'res3', 'res4', 'res5']
        
        # Create ResNet model
        stem = BasicStem(in_channels=3, out_channels=64, norm=norm_type)
        stages = make_resnet_stages(
            depth=depth,
            dilation=(1, 1, 1, 1),
            norm=norm_type
        )
        
        self.model = ResNet(
            stem=stem,
            stages=stages,
            num_classes=None,
            out_features=out_features,
            freeze_at=freeze_at
        )
        self.depth = 50
        self._feature_channels = self.model._out_feature_channels

    def get_features(self, x):
        return self.model(x)
    
    @property
    def feature_channels(self):
        return self._feature_channels
    
    