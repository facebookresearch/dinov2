import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses
from torch.cuda.amp import autocast

class DistillationModule(pl.LightningModule):
    def __init__(
        self,
        student,
        teacher,
        cfg
    ):
        super().__init__()
        self.cfg = cfg
        self._initialize_models(student, teacher)
        self.loss_fn = DistillationLoss(alpha=self.cfg.loss.alpha, beta=self.cfg.loss.beta)

    def _initialize_models(self, student, teacher):
        """Initialize and setup student and teacher models."""
        self.student = student
        self.teacher = teacher
        self._freeze_teacher()

    def _freeze_teacher(self):
        """Freeze teacher model parameters."""
        for param in self.teacher.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        features = self._extract_features(batch)
        losses = self.loss_fn(features['student'], features['teacher'])
        
        # Log metrics
        self._log_training_metrics(losses, features)
        return losses['total']

    def validation_step(self, batch, batch_idx):
        features = self._extract_features(batch)
        losses = self.loss_fn(features['student'], features['teacher'])
        
        self._log_validation_metrics(losses, features)

    def _extract_features(self, batch):
        """Extract features from both models."""
        global_crops = batch["collated_global_crops"]
        
        with torch.no_grad():
            teacher_output = self.teacher(global_crops)
            teacher_features = teacher_output[self.cfg.teacher.teacher_key]

        student_output = self.student(global_crops)
        student_features = student_output[self.cfg.student.student_key]

        return {
            'student': student_features,
            'teacher': teacher_features
        }

    def _log_training_metrics(self, losses, features):
        """Log training metrics."""
        self.log('train_loss', losses['total'])
        self.log('train_mse_loss', losses['mse'])
        self.log('train_cosine_loss', losses['cosine'])
        self.log('train_similarity', 
                self._compute_feature_similarity(features['student'], 
                                              features['teacher']))

    def _log_validation_metrics(self, losses, features):
        """Log validation metrics."""
        self.log('val_loss', losses['mse'])
        self.log('val_similarity', 
                self._compute_feature_similarity(features['student'], 
                                              features['teacher']))

    @staticmethod
    def _compute_feature_similarity(feat1, feat2):
        """Compute cosine similarity between feature vectors."""
        similarity = F.cosine_similarity(feat1, feat2, dim=1)
        return similarity.mean()

    def configure_optimizers(self):
        """Configure optimizers with flexible optimizer and scheduler options."""
        # Configure optimizer
        optimizer = getattr(torch.optim, self.cfg['optimizer']['type'])(
            self.student.parameters(),
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
        mse_loss = F.mse_loss(student_features, teacher_features)
        
        # Cosine similarity loss
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        cosine_sim = F.cosine_similarity(student_norm, teacher_norm, dim=1)
        cosine_loss = 1 - cosine_sim.mean()


        # Option 1: Normalize by sum of coefficients (recommended)
        total_loss = (self.alpha * mse_loss + self.beta * cosine_loss) / self.normalizer


        return {
            'total': total_loss,
            'mse': mse_loss,
            'cosine': cosine_loss
        }