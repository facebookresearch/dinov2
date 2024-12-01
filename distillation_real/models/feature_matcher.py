import torch.nn as nn

class FeatureMatcher(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        activation: str = 'ReLU'
    ):
        super().__init__()
        
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
        )
        
        if activation:
            layers.append(getattr(nn, activation)())
        
        self.matcher = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.matcher(x)