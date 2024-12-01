import torch.nn as nn
from ...distillation_real.models.students.resnet import ResNet, BottleneckBlock, BasicStem, make_resnet_stages
# Import other student models when added
# from .efficientnet import EfficientNet, ...

class StudentModelFactory:
    """
    Factory class for creating student models.
    """
    @staticmethod
    def create_student(model_type: str, config: dict, device: torch.device) -> nn.Module:
        if model_type.lower() == 'resnet':
            stem = BasicStem(
                in_channels=config.get('in_channels', 3),
                out_channels=64,
                norm=config.get('norm_type', 'BN')
            )
            stages = make_resnet_stages(
                depth=config.get('model_depth', 50),
                block_class=config.get('block_class', BottleneckBlock),
                norm=config.get('norm_type', 'BN'),
                dilation=config.get('dilation', (1, 1, 1, 1))
            )
            student = ResNet(
                stem=stem,
                stages=stages,
                out_features=None,
                freeze_at=config.get('freeze_at', 0)
            ).to(device)
            return student
        # elif model_type.lower() == 'efficientnet':
        #     return EfficientNet(config).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")