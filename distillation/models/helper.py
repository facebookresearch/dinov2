import sys  
sys.path.append("../../dinov2")
from dinov2.layers.dino_head import DINOHead  # Adjust based on actual location
from .backbones import DINOv2ViT, CustomResNet
import torch.nn as nn
def get_teacher_and_student(cfg = "../config/config.yaml"):
    """
    Initialize and return the teacher and student models based on the provided configuration.

    This function creates the backbone and heads for both teacher and student models
    as specified in the configuration. It supports different architectures for the student,
    such as ResNet and DINO V2 Vision Transformer (ViT). The models are encapsulated
    within `nn.ModuleDict` objects for easy integration and training.

    Args:
        cfg (str or dict): Path to the YAML configuration file or a dictionary
            containing configuration parameters. Defaults to "../config/config.yaml".


    Returns:
        tuple:
            - teacher (nn.ModuleDict): A module dictionary containing the teacher's backbone
              and associated heads (`dino_head`, `ibot_head`).
            - student (nn.ModuleDict): A module dictionary containing the student's backbone
              and associated heads (`dino_head`, `ibot_head`).

    Raises:
        ValueError: If the student model name specified in the configuration is unsupported.
    """
    student = {}
    teacher = {}



    # Create teacher and student backbones
    teacher_backbone = DINOv2ViT(model_name=cfg["teacher"]["model_name"])
    teacher['backbone'] = teacher_backbone

    if cfg["student"]["model_name"].startswith('resnet'):
        student_backbone = CustomResNet(model_name=cfg["student"]["model_name"], pretrained=True)

    elif cfg["student"]["model_name"].startswith('dino'):
        student_backbone = DINOv2ViT(model_name=cfg["student"]["model_name"])
    else:
        raise ValueError(f"Unsupported student model: {cfg['student']['model_name']}")
    
    student['backbone'] = student_backbone

    student["dino_head"] = DINOHead(in_dim=cfg["student"]["dino_head"]["in_dim"],
                                    out_dim=cfg["student"]["dino_head"]["out_dim"], 
                                    hidden_dim=cfg["student"]["dino_head"]["hidden_dim"], 
                                    bottleneck_dim=cfg["student"]["dino_head"]["bottleneck_dim"])
    teacher["dino_head"] = DINOHead(in_dim=cfg["teacher"]["dino_head"]["in_dim"],
                                    out_dim=cfg["teacher"]["dino_head"]["out_dim"], 
                                    hidden_dim=cfg["teacher"]["dino_head"]["hidden_dim"], 
                                    bottleneck_dim=cfg["teacher"]["dino_head"]["bottleneck_dim"])
    student["ibot_head"] = DINOHead(in_dim=cfg["student"]["ibot_head"]["in_dim"],
                                    out_dim=cfg["student"]["ibot_head"]["out_dim"], 
                                    hidden_dim=cfg["student"]["ibot_head"]["hidden_dim"], 
                                    bottleneck_dim=cfg["student"]["ibot_head"]["bottleneck_dim"])
    teacher["ibot_head"] = DINOHead(in_dim=cfg["teacher"]["ibot_head"]["in_dim"], 
                                    out_dim=cfg["teacher"]["ibot_head"]["out_dim"], 
                                    hidden_dim=cfg["teacher"]["ibot_head"]["hidden_dim"], 
                                    bottleneck_dim=cfg["teacher"]["ibot_head"]["bottleneck_dim"])
    
    student = nn.ModuleDict(student)
    teacher = nn.ModuleDict(teacher)

    return teacher, student