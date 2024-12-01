import torch
import torch.nn as nn

def match_vit_features(feature_map, patch_size, height, width):
    """
    Match the feature map to the ViT patch grid.
    
    Args:
        feature_map (torch.Tensor) (B, C, H, W): The feature map to match.
        patch_size (int): The patch size of the ViT model.
        height (int): The height of the image.
        width (int): The width of the image.
    
    Returns:
        patches: torch.Tensor (B, N, C): The feature map matched to the ViT patch grid.
    """
    B, C, H, W = feature_map.shape
    H_patch = height // patch_size
    W_patch = width // patch_size
    
    # Interpolate to match ViT patch grid
    feature_map = nn.functional.interpolate(
            feature_map,
            size=(H_patch, W_patch),
            mode='bilinear',
            align_corners=False
        )
        
        # Reshape into patches and project to match ViT dimension
    patches = feature_map.permute(0, 2, 3, 1)  # [B, H, W, C]
    patches = patches.reshape(B, H_patch * W_patch, C)  # [B, N, C]

    
    return patches

def match_feature_map(feature_map, patch_size, height, width):
    B, C, H, W = feature_map.shape
    H_patch = height // patch_size
    W_patch = width // patch_size
    return feature_map.reshape(B, C, H_patch, W_patch)