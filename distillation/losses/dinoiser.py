import torch
import torch.nn as nn
import wandb
class DinoiserLoss(nn.Module):
    def __init__(self,student_dims, teacher_dims):
        super().__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.feature_matcher = nn.Conv2d(student_dims, teacher_dims, kernel_size=3, stride=1, padding=1)
    def forward(self, student_features, teacher_features):
        # Match student spatial dimensions to teacher
        if student_features.shape[-2:] != teacher_features.shape[-2:]:
            student_features = nn.functional.interpolate(
                student_features,
                size=teacher_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        student_features = self.feature_matcher(student_features)
        B, C, H, W = student_features.shape
        teacher_features = teacher_features.flatten(-2,-1)
        student_features = student_features.flatten(-2,-1)
        teacher_features = teacher_features/teacher_features.norm(dim=1, keepdim=True)
        student_features = student_features/student_features.norm(dim=1, keepdim=True)
        teacher_corrs = torch.matmul(teacher_features.permute(0, 2, 1), teacher_features).reshape(B, H, W, H*W).permute(0, 3, 1, 2)
        student_corrs = torch.matmul(student_features.permute(0, 2, 1), student_features).reshape(B, H, W, H*W).permute(0, 3, 1, 2)
        total_loss = self.criterion(student_corrs.float().flatten(-2,-1), teacher_corrs.float().flatten(-2,-1))
        return {'loss': total_loss}
    
