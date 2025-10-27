"""Quantization utilities for model compression."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torch.utils.data import DataLoader


def apply_ptq_dynamic(model: nn.Module) -> nn.Module:
    # Work on a copy so the original stays on CUDA
    model_copy = copy.deepcopy(model).to("cpu").eval()
    qmodel = torch.quantization.quantize_dynamic(
        model_copy, {nn.Linear}, dtype=torch.qint8
    )
    return qmodel


class QATLinear(nn.Module):
    """Quantization-Aware Training wrapper for Linear layers."""
    
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        
        # Activation fake quantization
        self.act_fake_quant = FakeQuantize(
            observer=MovingAverageMinMaxObserver,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        )
        
        # Weight fake quantization
        self.weight_fake_quant = FakeQuantize(
            observer=MovingAverageMinMaxObserver,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        )
    
    def forward(self, x):
        # Ensure fake quantization modules are on the same device as inputs
        self.act_fake_quant.to(x.device)
        self.weight_fake_quant.to(x.device)
        
        # Apply fake quantization
        x_quantized = self.act_fake_quant(x)
        weight_quantized = self.weight_fake_quant(self.linear.weight)
        
        return F.linear(x_quantized, weight_quantized, self.linear.bias)


def replace_head_with_qat(model: nn.Module):
    """Replace classifier head with QAT-enabled version.
    
    Args:
        model: Model to modify
        
    Returns:
        Model with QAT-enabled head
    """
    device = next(model.parameters()).device
    
    # Try common classifier attribute names
    for attr in ["fc", "classifier", "head"]:
        if hasattr(model, attr):
            head = getattr(model, attr)
            
            if isinstance(head, nn.Linear):
                setattr(model, attr, QATLinear(head).to(device))
            elif isinstance(head, nn.Sequential):
                modules = list(head.children())
                if modules and isinstance(modules[-1], nn.Linear):
                    modules[-1] = QATLinear(modules[-1]).to(device)
                    setattr(model, attr, nn.Sequential(*modules).to(device))
            
            return model
    
    return model


@torch.no_grad()
def warmup_qat_observers(
    model: nn.Module, 
    loader: DataLoader, 
    num_batches: int = 10, 
    device: torch.device = None
) -> None:
    """Warm up QAT observers with sample data.
    
    Args:
        model: Model with QAT layers
        loader: Data loader for warmup
        num_batches: Number of batches for warmup
        device: Device to run warmup on
    """
    if device is None:
        device = next(model.parameters()).device
    
    was_training = model.training
    model.train()  # Observers need to be in training mode
    
    batches_seen = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=(device.type == "cuda"))
        model(x)
        batches_seen += 1
        if batches_seen >= num_batches:
            break
    
    model.train(was_training)
