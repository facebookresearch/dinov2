from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict, Any
from torch import Tensor

class BaseModel(ABC, nn.Module):
    """Base class for all models"""
    
    @abstractmethod
    def get_features(self, x: Tensor) -> Dict[str, Tensor]:
        """Get intermediate features from model"""
        pass
    
    @property
    @abstractmethod
    def feature_channels(self) -> Dict[str, int]:
        """Get number of channels for each feature level"""
        pass