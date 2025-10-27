"""Model backbone definitions and utilities."""

import torch
import torch.nn as nn
import timm
from torchvision.models import resnet18, ResNet18_Weights


def make_resnet18(num_classes: int, pretrained=True) -> nn.Module:
    """Create ResNet18 model with custom classifier head.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        ResNet18 model with modified classifier
    """
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=(ResNet18_Weights.DEFAULT if pretrained else None))
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def make_mlp_mixer_b16(num_classes: int, pretrained=True) -> nn.Module:
    """Create MLP-Mixer B16 model with custom classifier head.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        MLP-Mixer B16 model with modified classifier
    """
    return timm.create_model("mixer_b16_224", pretrained=pretrained, num_classes=num_classes)


def freeze_all_but_head(model: nn.Module):
    for p in model.parameters(): p.requires_grad = False
    for attr in ["fc","classifier","head"]:
        if hasattr(model, attr):
            for p in getattr(model, attr).parameters(): p.requires_grad = True
            return
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear): last = m
    if last is not None:
        for p in last.parameters(): p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters.
    
    Args:
        model: Model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True


def get_model_factory(backbone_name: str):
    """Get model factory function by name.
    
    Args:
        backbone_name: Name of the backbone ('resnet18' or 'mixer_b16_224')
        
    Returns:
        Model factory function
        
    Raises:
        ValueError: If backbone name is not supported
    """
    factories = {
        "resnet18": make_resnet18,
        "mixer_b16_224": make_mlp_mixer_b16,
    }
    
    if backbone_name not in factories:
        raise ValueError(f"Unknown backbone: {backbone_name}. Supported: {list(factories.keys())}")
    
    return factories[backbone_name]
