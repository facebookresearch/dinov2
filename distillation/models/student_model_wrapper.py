import torch
import torch.nn as nn
from ...distillation_real.models.students.resnet import ResNet, BasicStem, BottleneckBlock, make_resnet_stages
import numpy as np
class ModelWrapper(nn.Module):
    """
    A generalizable and parametric wrapper for student models and feature matching.

    Args:
        model_type (str): Type of the model to initialize (e.g., 'resnet').
        model_depth (int, optional): Depth of the ResNet model (default: 50).
        block_class (nn.Module, optional): Block class for ResNet (default: BottleneckBlock).
        norm_type (str, optional): Normalization type (default: 'BN').
        in_channels (int, optional): Number of input channels (default: 3).
        feature_matcher_config (dict, optional): Configuration for the feature matcher.
            Should include 'in_channels', 'out_channels', 'kernel_size', and other relevant parameters.
        device (torch.device, optional): Device to load the model on (default: 'cpu').

    Attributes:
        student (nn.Module): The student model.
        feature_matcher (nn.Module): The feature matcher module.
    """
    def __init__(
        self,
        model_type='resnet',
        model_depth=50,
        block_class=BottleneckBlock,
        norm_type='BN',
        in_channels=3,
        n_patches=256,
        feature_matcher_config=None,
        device=torch.device('cpu')
    ):
        super(ModelWrapper, self).__init__()
        self.device = device
        self.n_patches = n_patches
        # Initialize the student model based on the specified type
        if model_type.lower() == 'resnet':
            stem = BasicStem(in_channels=in_channels, out_channels=64, norm=norm_type)
            stages = make_resnet_stages(
                depth=model_depth,
                block_class=block_class,
                norm=norm_type,
                dilation=(1, 1, 1, 1)
            )
            self.student = ResNet(
                stem=stem,
                stages=stages,
                out_features=None,
                freeze_at=0
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Initialize the feature matcher if configuration is provided
        if feature_matcher_config:
            self.feature_matcher = self._initialize_feature_matcher(feature_matcher_config)
        else:
            self.feature_matcher = None

    def _initialize_feature_matcher(self, config):
        """
        Initializes the feature matcher based on the provided configuration.

        Args:
            config (dict): Configuration dictionary for the feature matcher.

        Returns:
            nn.Module: Initialized feature matcher module.
        """
        layers = []
        in_channels = config.get('in_channels', 2048)
        out_channels = config.get('out_channels', 1536)
        kernel_size = config.get('kernel_size', 1)
        stride = config.get('stride', 1)
        padding = config.get('padding', 0)
        activation = config.get('activation', None)

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        if activation:
            layers.append(getattr(nn, activation)())

        return nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        """
        Forward pass through the student model and feature matcher.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (matched_features, student_output)
        """
        # Forward pass through the student model
        student_output = self.student(x)

        # If feature matcher is defined, process the feature maps
        if self.feature_matcher and 'res5' in student_output:
            res5_feature = student_output['res5']
            interpolated_feature = torch.nn.functional.interpolate(
                res5_feature, 
                size=(int(np.sqrt(self.n_patches)), int(np.sqrt(self.n_patches))), 
                mode='bilinear', 
                align_corners=False
            )
            matched_features = self.feature_matcher(interpolated_feature)
            return matched_features, student_output
        else:
            return None, student_output

    def get_feature_map_shape(self, x, feature_key='res5'):
        """
        Utility method to get the shape of a specified feature map.

        Args:
            x (torch.Tensor): Input tensor.
            feature_key (str, optional): Key of the feature map in student output (default: 'res5').

        Returns:
            torch.Size: Shape of the specified feature map.
        """
        output = self.student(x)
        if feature_key in output:
            return output[feature_key].shape
        else:
            raise KeyError(f"Feature key '{feature_key}' not found in student output.")