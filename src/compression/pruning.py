"""Pruning utilities for model compression."""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_pruning as tp
from torch.utils.data import DataLoader

# Device should be imported from config or set here
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def global_unstructured_prune(model: nn.Module, amount: float):
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    if not parameters_to_prune: return
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    for (m, _) in parameters_to_prune:
        try: prune.remove(m, 'weight')
        except Exception: pass


def structured_channel_prune_resnet(
    model: nn.Module,
    flops_reduction_target: float = 0.30,
    img_size: int = 224
):
    """
    ResNet-style pruning: globally prune conv output channels to hit ~flops_reduction_target.
    Keeps the stem conv and final classifier intact.
    """
    model = model.to(DEVICE).eval()
    example_inputs = torch.randn(1, 3, img_size, img_size).to(DEVICE)
    
    # Ignore stem conv and final classifier
    ignored_layers = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name.startswith("conv1"):
            ignored_layers.add(module)
    
    if hasattr(model, "fc"):
        ignored_layers.add(model.fc)
    
    # L1 magnitude-based importance
    importance = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        global_pruning=True,
        pruning_ratio=flops_reduction_target,
        iterative_steps=1,
        ignored_layers=list(ignored_layers),
    )
    
    pruner.step()
    return model


def structured_channel_prune_mixer(
    model: nn.Module,
    flops_reduction_target: float = 0.30,
    img_size: int = 224
):
    """
    MLP-Mixer pruning: globally prune large Linear outputs while keeping the final classifier intact.
    """
    model = model.to(DEVICE).eval()
    example_inputs = torch.randn(1, 3, img_size, img_size).to(DEVICE)
    
    # Ignore final classifier
    ignored_layers = set()
    for attr in ["head", "classifier", "fc"]:
        if hasattr(model, attr):
            ignored_layers.add(getattr(model, attr))
    
    importance = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        global_pruning=True,
        pruning_ratio=flops_reduction_target,
        iterative_steps=1,
        ignored_layers=list(ignored_layers),
    )
    
    pruner.step()
    return model


@torch.no_grad()
def recalibrate_batch_norm(
    model: nn.Module, 
    loader: DataLoader, 
    num_batches: int = 10, 
    device: torch.device = None
) -> None:
    """Recalibrate BatchNorm statistics after structured pruning.
    
    Args:
        model: Model with BatchNorm layers to recalibrate
        loader: Data loader for recalibration
        num_batches: Number of batches to use for recalibration
        device: Device to run recalibration on
    """
    if device is None:
        device = next(model.parameters()).device
    
    was_training = model.training
    model.train()
    
    batches_seen = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=(device.type == "cuda"))
        model(x)
        batches_seen += 1
        if batches_seen >= num_batches:
            break
    
    model.train(was_training)
