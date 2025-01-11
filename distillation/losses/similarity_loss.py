import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureSimilarityDistillation(nn.Module):
    """
    Implements feature-level similarity distillation using MSE loss on similarity matrices.
    """
    def __init__(self, temperature=1.0, weight=1.0):
        """
        Args:
            temperature (float): Temperature scaling for similarity computation.
            weight (float): Weight for the similarity loss component.
        """
        super(FeatureSimilarityDistillation, self).__init__()
        self.temperature = temperature
        self.weight = weight
        self.mse_loss = nn.MSELoss()

    def compute_similarity_matrix(self, features):
        """
        Computes the similarity matrix for a batch of feature maps.

        Args:
            features (Tensor): Feature maps of shape [B, C, H, W]

        Returns:
            Tensor: Similarity matrices of shape [B, H*W, H*W]
        """
        B, C, H, W = features.shape
        # Reshape to [B, C, N]
        N = H * W
        features = features.view(B, C, N)
        # Normalize feature vectors
        features = F.normalize(features, p=2, dim=1)  # [B, C, N]
        # Compute similarity matrices: [B, N, N]
        similarity = torch.bmm(features.transpose(1, 2), features)  # [B, N, N]
        # Scale similarity by temperature
        similarity = similarity / self.temperature
        return similarity

    def forward(self, student_features, teacher_features):
        """
        Forward pass to compute feature similarity loss.

        Args:
            student_features (Tensor): Student feature maps [B, C, H, W]
            teacher_features (Tensor): Teacher feature maps [B, C, H, W]

        Returns:
            Dict: Contains 'feature_similarity_loss'
        """
        # Compute similarity matrices
        student_similarity = self.compute_similarity_matrix(student_features)  # [B, N, N]
        teacher_similarity = self.compute_similarity_matrix(teacher_features)  # [B, N, N]

        # Compute MSE loss between student and teacher similarity matrices
        loss = self.mse_loss(student_similarity, teacher_similarity)

        return loss