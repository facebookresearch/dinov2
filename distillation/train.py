import warnings
import sys
import os
from typing import Dict, Any
from dataclasses import dataclass
import argparse
sys.path.append('../')
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from models import DINOv2ViT
from models.model_wrapper import ModelWrapper
from train.distillation_module import DistillationModule
from datasets.GTA5 import CustomDataModule
from dinov2.data.augmentations import DataAugmentationDINO


@dataclass
class TrainingConfig:
    """Configuration for training setup."""
    max_epochs: int
    precision: int
    learning_rate: float
    alpha: float
    beta: float


class DistillationTrainer:
    """Handles the training pipeline for knowledge distillation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.training_config = self._setup_training_config()
        
        # Initialize components
        self.transform = self._create_transform()
        self.data_module = self._create_data_module()
        self.teacher, self.student = self._create_models()
        self.distillation_module = self._create_distillation_module()
        self.trainer = self._create_trainer()

    def _setup_training_config(self) -> TrainingConfig:
        """Setup training configuration."""
        return TrainingConfig(
            max_epochs=self.cfg['train']['max_epochs'],
            precision=self.cfg.get('precision', 16),
            learning_rate=self.cfg['optimizer']['kwargs']['lr'],
            alpha=self.cfg['loss']['alpha'],
            beta=self.cfg['loss']['beta']
        )

    def _create_transform(self) -> DataAugmentationDINO:
        """Create data transformation pipeline."""
        return DataAugmentationDINO(
            global_crops_scale=tuple(self.cfg['data_transform']['global_crops_scale']),
            local_crops_scale=tuple(self.cfg['data_transform']['local_crops_scale']),
            local_crops_number=self.cfg['data_transform']['n_local_crops'],
            global_crops_size=tuple(self.cfg['data_transform']['global_crops_size']),
            local_crops_size=tuple(self.cfg['data_transform']['local_crops_size']),
        )

    def _create_data_module(self) -> CustomDataModule:
        """Create data module."""
        return CustomDataModule(
            data_dir=self.cfg['data_loader'].get('data_dir', '/home/arda/data/train2017'),
            transform=self.transform,
            batch_size=self.cfg['data_loader']['batch_size'],
            num_workers=self.cfg['data_loader']['num_workers']
        )

    def _create_models(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Create teacher and student models."""
        teacher = DINOv2ViT(
            model_name=self.cfg['teacher']['model_name'],
        )
        student = ModelWrapper(
            model_type=self.cfg['student']['model_name'],
            n_patches=self.cfg.teacher.n_patches,
            target_feature=[self.cfg['student']['student_key']],
            feature_matcher_config=self.cfg['teacher']['feature_matcher'],
            **self.cfg['student']['kwargs']
        )
        return teacher, student

    def _create_distillation_module(self) -> DistillationModule:
        """Create distillation module."""
        return DistillationModule(
            student=self.student,
            teacher=self.teacher,
            cfg=self.cfg
        )

    def _create_trainer(self) -> pl.Trainer:
        """Create PyTorch Lightning trainer."""
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.cfg.checkpoints.dirpath + f"/{self.cfg.train.name}",
            filename="{epoch}-{val_similarity:.2f}",
            monitor=self.cfg.checkpoints.monitor,
            mode=self.cfg.checkpoints.mode,
            save_top_k=self.cfg.checkpoints.save_top_k
        )

        logger = TensorBoardLogger(f"logs/{self.cfg.train.name}", name="distillation")
        logger.log_hyperparams(self.cfg)
    
        # Also save config as text for better readability
        experiment_dir = logger.log_dir
        os.makedirs(experiment_dir, exist_ok=True)
        config_path = os.path.join(experiment_dir, 'config.yaml')
        OmegaConf.save(self.cfg, config_path)

        return pl.Trainer(
            max_epochs=self.training_config.max_epochs,
            accelerator=self.cfg.train.accelerator,
            devices=self.cfg.train.devices,
            num_nodes=self.cfg.train.num_nodes,
            strategy=self.cfg.train.strategy,  # 'ddp', 'deepspeed', etc.
            precision=self.training_config.precision,
            callbacks=[checkpoint_callback],
            logger=logger,
        )

    def train(self):
        """Execute training pipeline."""
        self.trainer.fit(self.distillation_module, self.data_module)


def setup_environment():
    """Setup environment configurations."""
    # Add the project root to Python path
    sys.path.append('../')
    
    # Configure warnings
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Set precision for Tensor Cores
    torch.set_float32_matmul_precision('high')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Training script for distillation')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to the config file'
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup environment
    setup_environment()
    
    # Load configuration from YAML
    with open(args.config, "r") as f:
        config = OmegaConf.load(f)
    
    trainer = DistillationTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()