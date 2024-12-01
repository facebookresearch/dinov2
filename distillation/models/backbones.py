from utils import match_vit_features

import torch
from torch import nn
import torchvision.models as models
import math
from .aggregators import MixVPR

class CustomResNet(nn.Module):
    def __init__(self, model_name='resnet50', patch_size=14, pretrained=True):
        super().__init__()
        """
        Custom ResNet model for distillation.

        Args:
            model_name (str): The name of the ResNet model to use.
            patch_size (int): The patch size of the ViT model.
            pretrained (bool): Whether to use pretrained weights.

        Returns:
            dict: A dictionary containing:
                feature_map: torch.Tensor (B, C, H, W): The feature map after the last layer of ResNet.
                embeddings: torch.Tensor (B, D): The embeddings after global average pooling.
                patches: torch.Tensor (B, N, C): The patches which match the ViT patch grid.
        """

        if pretrained:
            base_model = getattr(models, model_name)(weights='IMAGENET1K_V1')
        else:
            base_model = getattr(models, model_name)(weights=None)
        self.patch_size = patch_size

        # Split the model into layers
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        self.avgpool = base_model.avgpool
        in_channels = 2048 if model_name in ['resnet50', 'resnet101', 'resnet152'] else 512
        self.feature_matcher = nn.Conv2d(in_channels, 1536, kernel_size=1)


        # self.mix = MixVPR(in_channels=int(in_channels), in_h=7, in_w=7, out_channels=512, mix_depth=4, mlp_ratio=1, out_rows=4)
        self.attention = nn.MultiheadAttention(embed_dim=1536, num_heads=16, batch_first=True)


    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        layer_3 = self.layer3(x)
        layer4 = self.layer4(layer_3)  # Final feature map before pooling
        feature_map = torch.nn.functional.interpolate(
            layer4, 
            size=(16, 16), 
            mode='bilinear', 
            align_corners=False
        )
        feature_map = self.feature_matcher(feature_map)

        
        

        pooled = self.avgpool(feature_map)  # Global average pooling
        embeddings = torch.flatten(pooled, 1)  # Flatten to get embeddings


        # print(f"layer_3e shape: {layer_3.shape}")
        # contrastive_embeddings = self.mix(layer4)
        B, C, H, W = feature_map.shape
        tokens = feature_map.view(B, C, H * W).permute(0, 2, 1)  # [B, T, C]
        attn_output, attn_weights = self.attention(tokens, tokens, tokens)  # attn_weights: [B, T, T]
        return {
            'feature_map': layer4,  # Final feature map after layer4
            'embedding': embeddings,     # Embeddings after pooling
            # 'patch_embeddings': patches,
            'dinov2_feature_map': feature_map,
            # 'contrastive_embeddings': contrastive_embeddings,
            'attn_weights': attn_weights,
        }
    
class DINOv2ViT(nn.Module):
    def __init__(self, model_name='dinov2_vitg14'):
        super().__init__()
        """
        DINOv2 ViT model for distillation.

        Args:
            model_name (str): The name of the DINOv2 ViT model to use.

        Returns:
            dict: A dictionary containing:
                patch_embeddings: torch.Tensor (B, N, D): The patch embeddings excluding the CLS token.
                cls_embedding: torch.Tensor (B, D): The CLS token embedding.
        """
        # Load model from torch hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
    def forward(self, x):
        # Get features from the model's last layer
        patch_embeddings, cls_token = self.model.get_intermediate_layers(x, n=1, return_class_token=True)[0]  # [B, N+1, D]
        # print(f" patch_embeddings shape: {patch_embeddings.shape}")
        # print(f" cls_token shape: {cls_token.shape}")
        # Convert patch embeddings to feature map format
        B, N, D = patch_embeddings.shape
        P = int(math.sqrt(N))  # -1 for cls token
        feature_map = patch_embeddings.reshape(B, P, P, D).permute(0, 3, 1, 2)  # [B, D, P, P]

        return {
            'patch_embeddings': patch_embeddings,  # Per-patch embeddings excluding CLS
            'embedding': cls_token,        # CLS token embedding
            'feature_map': feature_map,
        }





