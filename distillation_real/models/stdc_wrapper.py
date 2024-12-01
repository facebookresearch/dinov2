from typing import List, Optional
import torch.nn as nn
from .base import BaseModel
from .stdc import STDCNet

class STDCWrapper(BaseModel):
    def __init__(
        self,
        base_channels: int = 64,
        layers: List[int] = [4, 5, 3],
        block_num: int = 4,
        block_type: str = 'cat',
        use_conv_last: bool = False,
    ):
        super().__init__()
        
        self.model = STDCNet(
            base=base_channels,
            layers=layers,
            block_num=block_num,
            block_type=block_type,
            use_conv_last=use_conv_last
        )
        
        self.base = base_channels
        self.use_conv_last = use_conv_last
        
        # Initialize model
        self.model.init_params()
    
    def get_features(self, x):
        return self.model(x)
    
    @property
    def feature_channels(self):
        channels = {
            'res2': self.base * 1,
            'res3': self.base * 4,
            'res4': self.base * 8,
            'res5': self.base * 16 if not self.use_conv_last else max(1024, self.base * 16)
        }
        return channels