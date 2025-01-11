import lightning as L
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import os
import tempfile
import torch.nn as nn
import logging
from losses import ScaleKD
# Create a temporary directory in your home or storage
USER_TMP = '/storage/disk0/arda/tmp'
os.makedirs(USER_TMP, exist_ok=True)

# Set multiple environment variables to ensure temp files go to the right place
os.environ['TMPDIR'] = USER_TMP
os.environ['TEMP'] = USER_TMP
os.environ['TMP'] = USER_TMP
tempfile.tempdir = USER_TMP

class DistillationModule(L.LightningModule):
    def __init__(
        self,
        student,
        teacher,
        cfg
    ):
        super().__init__()
        
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self._initialize_models(student, teacher)
        self.loss_fn = DistillationLoss(alpha=self.cfg.loss.alpha, beta=self.cfg.loss.beta)
        # Initialize ScaleKD here
        scalekd = ScaleKD(
            name='scalekd',
            use_this=True,
            alpha=[0.6, 0.4],
            student_dims=2048,
            teacher_dims=1536,
            query_hw=(16, 16),
            pos_hw=(16, 16),
            pos_dims=1536,
            window_shapes=(1, 1),
            self_query=True,
            softmax_scale=[5.0, 5.0],
            dis_freq='high',
            num_heads=16
        ).to(self.device)
        # Register it properly as a module
        self.register_module('scalekd', scalekd)
        self.loss_fn.scalekd = scalekd
    def _initialize_models(self, student, teacher):
        """Initialize and setup student and teacher models."""
        self.student = student
        self.teacher = teacher
        self._freeze_teacher()
        if self.cfg.student.get('checkpoint_path', None):
            self._load_student_checkpoint(self.cfg.student.checkpoint_path)
        

    def _freeze_teacher(self):
        """Freeze teacher model parameters."""
        for param in self.teacher.parameters():
            param.requires_grad = False
        

    def training_step(self, batch, batch_idx):
        features = self._extract_features(batch)
        losses = self.loss_fn(features['student'], features['teacher'], features['teacher_class_token'])
        
        # Log metrics
        self._log_training_metrics(losses, features)
        return losses['total']

    def validation_step(self, batch, batch_idx):
        features = self._extract_features(batch)
        losses = self.loss_fn(features['student'], features['teacher'], features['teacher_class_token'])
        
        self._log_validation_metrics(losses, features)

    def _extract_features(self, batch):
        """Extract features from both models."""
        global_crops = batch["collated_global_crops"]
        
        with torch.no_grad():
            teacher_output = self.teacher(global_crops)
            teacher_features = teacher_output[self.cfg.teacher.teacher_key]
            teacher_cls_token = teacher_output['embedding']


        student_output = self.student(global_crops)
        student_features = student_output[self.cfg.student.student_key]
        return {
            'student': student_features,
            'teacher': teacher_features,
            'teacher_class_token':teacher_cls_token,
        }

    def _log_training_metrics(self, losses, features):
        """Log training metrics."""
        # Add sync_dist=True for proper distributed logging
        self.log('train_loss', losses['total'], sync_dist=True)
        self.log('train_mse_loss', losses['mse'], sync_dist=True)
        self.log('train_cosine_loss', losses['cosine'], sync_dist=True)
        self.log('train_similarity', 
                1,
                sync_dist=True)
    def _log_validation_metrics(self, losses, features):
        """Log validation metrics."""
        self.log('val_loss', losses['mse'], sync_dist=True)
        self.log('val_similarity', 
                1,
                sync_dist=True)

    @staticmethod
    def _compute_feature_similarity(feat1, feat2):
        """Compute cosine similarity between feature vectors."""
        feat1 = feat1 / feat1.norm(dim=1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=1, keepdim=True)
        similarity = F.cosine_similarity(feat1, feat2, dim=1)
        return similarity.mean()

    def _load_student_checkpoint(self, checkpoint_path):
        """Load student checkpoint and return state dict."""
        checkpoint = torch.load(checkpoint_path)
        # Filter checkpoint to only include keys that exist in student model
        # state_dict = {k: v for k, v in checkpoint.items() if k in self.student.state_dict()}
        # self.student.load_state_dict(state_dict, strict=False)
        checkpoint = {f"model.model.{k}": v for k, v in checkpoint.items()}

        self.student.load_state_dict(checkpoint, strict=False)
        
        return checkpoint

    def configure_optimizers(self):
        """Configure optimizers with flexible optimizer and scheduler options."""
        # Configure optimizer
        params = list(self.student.parameters()) + list(self.loss_fn.scalekd.parameters())

        optimizer = getattr(torch.optim, self.cfg['optimizer']['type'])(
            params,
            **self.cfg['optimizer'].get('kwargs', {})
        )
        
        # cfgure scheduler if specified
        if 'scheduler' in self.cfg['optimizer']:
            scheduler = getattr(torch.optim.lr_scheduler, 
                              self.cfg['optimizer']['scheduler']['type'])(
                optimizer,
                **self.cfg['optimizer']['scheduler'].get('kwargs', {})
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.cfg['optimizer']['scheduler'].get('monitor', 'val_loss'),
                    "interval": self.cfg['optimizer']['scheduler'].get('interval', 'epoch'),
                    "frequency": self.cfg['optimizer']['scheduler'].get('frequency', 1)
                }
            }
        
        return optimizer


class DistillationLoss:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.scalekd = None  # Initialize as None
    @staticmethod
    def _compute_feature_similarity(feat1, feat2):
        """Compute cosine similarity between feature vectors."""
        feat1 = feat1 / feat1.norm(dim=1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=1, keepdim=True)
        similarity = F.cosine_similarity(feat1, feat2, dim=1)
        return similarity.mean()  
    def __call__(self, student_features, teacher_features, teacher_cls_token):
            
        spatial_loss, freq_loss, sim = self.scalekd(student_features, teacher_features,teacher_cls_token)
        scalekd_loss = spatial_loss + freq_loss
        cosine_loss = 1 - sim
        return {
            'total': scalekd_loss,
            'mse': 1,
            'cosine': cosine_loss
        }

class SimilarityDistillationLoss:
    """Handles all loss computations for distillation."""
    
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Args:
            alpha: Weight for MSE loss
            beta: Weight for cosine loss
        """
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, student_features, teacher_features):
        """
        Compute all losses and return as dictionary.
        
        Args:
            student_features: Features from student model [B, C, H, W]
            teacher_features: Features from teacher model [B, C, H, W]
        """
        B, C, H, W = student_features.shape
        
        teacher_features = teacher_features.flatten(-2,-1)
        student_features = student_features.flatten(-2,-1)
        teacher_features = teacher_features/teacher_features.norm(dim=1, keepdim=True)
        student_features = student_features/student_features.norm(dim=1, keepdim=True)
        cosine_similarity = torch.matmul(teacher_features, student_features.transpose(1,2))
        cosine_loss = 1-torch.mean(cosine_similarity)
        teacher_corrs = torch.matmul(teacher_features.permute(0, 2, 1), teacher_features).reshape(B, H, W, H*W).permute(0, 3, 1, 2)
        student_corrs = torch.matmul(student_features.permute(0, 2, 1), student_features).reshape(B, H, W, H*W).permute(0, 3, 1, 2)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        total_loss = bce_loss(student_corrs.float().flatten(-2,-1), teacher_corrs.float().flatten(-2,-1))

        return {
            'total': total_loss,
            'mse': 1,
            'cosine': cosine_loss
        }


