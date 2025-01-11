import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss:
    """Handles all loss computations for distillation."""
    
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Args:
            alpha: Weight for MSE loss
            beta: Weight for cosine loss
        """
        self.alpha = alpha
        self.beta = beta
        # Option 1: Normalize by sum of coefficients
        self.normalizer = alpha + beta if (alpha + beta) > 0 else 1.0

    def __call__(self, student_features, teacher_features):
        """Compute all losses and return as dictionary."""

        student_norm = F.normalize(student_features, dim=1)
        teacher_norm = F.normalize(teacher_features, dim=1)
        N,C,H,W = student_norm.shape
        # MSE on normalized features
        mse = nn.MSELoss(reduction='sum')
        mse_loss = mse(student_norm, teacher_norm)/N

        cosine_sim = F.cosine_similarity(student_norm, teacher_norm, dim=1)
        cosine_loss = 1 - cosine_sim


        # Option 1: Normalize by sum of coefficients (recommended)
        total_loss = (self.alpha * mse_loss + self.beta * cosine_loss) / self.normalizer


        return {
            'loss': total_loss.mean(),
            'mse': mse_loss.mean(),
            'cosine_similarity': cosine_sim.mean()
        }

