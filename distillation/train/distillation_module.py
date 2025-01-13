import lightning as L
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import os
import tempfile
import torch.nn as nn
import logging
from losses import ScaleKD, DinoiserLoss
# Create a temporary directory in your home or storage
USER_TMP = '/storage/disk0/arda/tmp'
os.makedirs(USER_TMP, exist_ok=True)

# Set multiple environment variables to ensure temp files go to the right place
os.environ['TMPDIR'] = USER_TMP
os.environ['TEMP'] = USER_TMP
os.environ['TMP'] = USER_TMP
tempfile.tempdir = USER_TMP

LOSS_REGISTRY = {
    'scalekd': ScaleKD,
    'dinoiser': DinoiserLoss
}

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
        self._initialize_loss()

    def _initialize_models(self, student, teacher):
        """Initialize models with gradient verification."""
        self.student = student
        self.teacher = teacher
        self.register_module('student', self.student)



        
        # Freeze teacher
        self._freeze_teacher()
        
        # Load checkpoint if specified
        if self.cfg.student.get('checkpoint_path', None):
            self._load_student_checkpoint(self.cfg.student.checkpoint_path)
                # Assert all student parameters are trainable
        for name, param in self.student.named_parameters():
            assert param.requires_grad, f"Parameter {name} in student model must be trainable"
            

    def _freeze_teacher(self):
        """Freeze teacher model parameters."""
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        

    def _initialize_loss(self):
        """Initialize compound loss function."""
        self.losses = nn.ModuleDict()  # ModuleDict automatically registers modules
        self.loss_weights = {}
        
        for loss_spec in self.cfg.loss['losses']:
            loss_type = loss_spec['type']
            weight = loss_spec['weight']
            kwargs = loss_spec['kwargs']
            
            # Get loss name based on type or kwargs
            name = kwargs.get('name', loss_type)
            
            # Get loss class from registry and add it to ModuleDict
            loss_fn = LOSS_REGISTRY[loss_type](**kwargs)
            self.losses[name] = loss_fn  # This automatically registers the module
            self.loss_weights[name] = weight

        # Debug: Print all registered modules and their parameters
        for name, module in self.named_modules():
            if isinstance(module, nn.Module):
                params = [p for p in module.parameters() if p.requires_grad]
                if params:
                    print(f"Module {name} has {len(params)} trainable parameters")

    def _forward_specific_stage(self, feat):
        """Forward through specific stages of teacher model."""
        B = feat.shape[0]
        cls_tokens = self.teacher.model.cls_token.expand(B, -1, -1)
        feat = torch.cat((cls_tokens, feat), dim=1)

        n_total_blocks = len(self.teacher.model.blocks)
        target_block = int(n_total_blocks/4*3)

        # Use torch.inference_mode() for better performance
            # Forward through remaining blocks
        for i in range(target_block, n_total_blocks):
            feat = self.teacher.model.blocks[i](feat)


        return feat[:,1:,:]

    def _compute_losses(self, student_features_s3, student_features, teacher_features, *args, **kwargs):
        """Compute compound loss."""
        total_loss = 0
        loss_dict = {}
        
        # Handle ScaleKD losses
        scalekd_n_loss = self.losses['scalekd_n']
        scalekd_last_loss = self.losses['scalekd_last']
        scalekd_n_weight = self.loss_weights['scalekd_n']
        scalekd_last_weight = self.loss_weights['scalekd_last']
        
        N,C,H,W = teacher_features.shape
        feat_S_s3_spat = scalekd_n_loss.project_feat_spat(student_features_s3, query=None)
        feat_S_s3_freq = scalekd_n_loss.project_feat_freq(student_features_s3, query=None)

        feat_S_s3_spat = self._forward_specific_stage(feat_S_s3_spat)
        feat_S_s3_freq = self._forward_specific_stage(feat_S_s3_freq)
        feat_S_s3_spat_query, feat_S_s3_freq_query = feat_S_s3_spat, feat_S_s3_freq
        
        scalekd_n_spat = scalekd_n_loss.get_spat_loss(feat_S_s3_spat, teacher_features)
        scalekd_n_freq = scalekd_n_loss.get_freq_loss(feat_S_s3_freq, teacher_features)
        scalekd_last_dict = scalekd_last_loss(student_features, teacher_features, 
                                            query_s=feat_S_s3_spat_query, 
                                            query_f=feat_S_s3_freq_query)

        # Add ScaleKD losses to total loss and loss dict
        total_loss += (scalekd_n_spat[0] + scalekd_n_freq) * scalekd_n_weight
        total_loss += scalekd_last_dict['loss'] * scalekd_last_weight
        loss_dict['loss_scalekd_n_spat'] = scalekd_n_spat[0] * scalekd_n_weight
        loss_dict['loss_scalekd_n_freq'] = scalekd_n_freq * scalekd_n_weight
        loss_dict['loss_scalekd_n_similarity'] = scalekd_n_spat[1] * scalekd_n_weight
        loss_dict['loss_scalekd_last'] = scalekd_last_dict['loss'] * scalekd_last_weight
        loss_dict['loss_scalekd_last_similarity'] = scalekd_last_dict['cosine_similarity'] * scalekd_last_weight
        loss_dict['loss_scalekd_last_spatial_loss'] = scalekd_last_dict['spatial_loss'] * scalekd_last_weight
        loss_dict['loss_scalekd_last_frequency_loss'] = scalekd_last_dict['frequency_loss'] * scalekd_last_weight

        # Handle other losses
        for name, loss_fn in self.losses.items():
            if name not in ['scalekd_n', 'scalekd_last']:  # Skip ScaleKD losses as they're already processed
                weight = self.loss_weights[name]
                curr_loss = loss_fn(student_features, teacher_features, *args, **kwargs)
                
                if isinstance(curr_loss, dict):
                    for k, v in curr_loss.items():
                        if k == 'loss':
                            weighted_loss = v * weight
                            total_loss += weighted_loss
                            loss_dict[f'loss_{name}'] = weighted_loss
                        else:
                            loss_dict[f'{name}_{k}'] = v
                else:
                    weighted_loss = curr_loss * weight
                    total_loss += weighted_loss
                    loss_dict[f'loss_{name}'] = weighted_loss

        loss_dict['loss'] = total_loss
        return loss_dict
    def training_step(self, batch, batch_idx):
        """Training step with detailed gradient debugging."""

        
        # Get features with gradient checking
        features = self._extract_features(batch)

        # Compute losses with gradient tracking
        losses = self._compute_losses(
            features['student']['res4'],
            features['student']['res5'], 
            features['teacher']
        )
        
        self._log_training_metrics(losses, features)

        return losses['loss']

    def validation_step(self, batch, batch_idx):
        features = self._extract_features(batch)
        losses = self._compute_losses(features['student']['res4'],features['student']['res5'], features['teacher'])
        self._log_validation_metrics(losses, features)

    def _extract_features(self, batch):
        """Extract features from both models."""
        global_crops = batch["collated_global_crops"]
        
        with torch.no_grad():
            teacher_output = self.teacher(global_crops)
            teacher_features = teacher_output[self.cfg.teacher.teacher_key]

        student_output = self.student(global_crops)
        return {
            'student': student_output,
            'teacher': teacher_features
        }

    def _log_training_metrics(self, losses, features):
        """Log training metrics."""
        # Log all your training losses and metrics here
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, sync_dist=True)
        

    def _log_validation_metrics(self, losses, features):
        """Log validation metrics."""
        # Log all your validation losses and metrics here
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, sync_dist=True)


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
        # Collect parameters from both student and losses
        param_groups = []
        
        # Student parameters
        student_params = list(self.student.parameters())
        param_groups.append({
            'params': student_params,
            'name': 'student'
        })
        
        # Loss function parameters
        for loss_name, loss_module in self.losses.items():
            loss_params = list(loss_module.parameters())
            if loss_params:  # Only add if there are parameters
                param_groups.append({
                    'params': loss_params,
                    'name': f'loss_{loss_name}'
                })
                print(f"Added {len(loss_params)} parameters from {loss_name} loss")

        optimizer = getattr(torch.optim, self.cfg['optimizer']['type'])(
            param_groups,
            **self.cfg['optimizer'].get('kwargs', {})
        )
        
        # Configure scheduler if specified
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
    

