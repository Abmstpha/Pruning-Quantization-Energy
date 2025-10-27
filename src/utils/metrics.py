"""Model metrics and profiling utilities."""

import io
import copy
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
import thop


def count_parameters(model: nn.Module) -> float:
    """Count total number of parameters in millions.
    
    Args:
        model: Model to analyze
        
    Returns:
        Number of parameters in millions
    """
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_flops(model: nn.Module, img_size: int = 224) -> float:
    """Estimate FLOPs in billions for given input size.
    
    Args:
        model: Model to analyze
        img_size: Input image size
        
    Returns:
        FLOPs in billions
    """
    # Create CPU copy and disable quantization artifacts
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    
    try:
        # Try to disable fake quantization and observers
        import torch.ao.quantization as aoq
        try:
            aoq.disable_fake_quant(model_cpu)
        except Exception:
            pass
        try:
            aoq.disable_observer(model_cpu)
        except Exception:
            pass
    except Exception:
        # Manual fallback for modules with these methods
        for module in model_cpu.modules():
            if hasattr(module, "disable_fake_quant"):
                module.disable_fake_quant()
            if hasattr(module, "disable_observer"):
                module.disable_observer()
    
    try:
        # Try ptflops first
        macs, _ = get_model_complexity_info(
            model_cpu, 
            (3, img_size, img_size), 
            as_strings=False,
            print_per_layer_stat=False, 
            verbose=False
        )
        return float(macs * 2) / 1e9  # MACs to FLOPs
    except Exception:
        # Fallback to thop
        dummy_input = torch.randn(1, 3, img_size, img_size)
        flops, _ = thop.profile(model_cpu, inputs=(dummy_input,), verbose=False)
        return float(flops) / 1e9


def calculate_model_size(model: nn.Module) -> float:
    """Calculate model size in MB.
    
    Args:
        model: Model to analyze
        
    Returns:
        Model size in MB
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / (1024**2)


def calculate_compression_ratio(
    original_size: float, 
    compressed_size: float
) -> float:
    """Calculate compression ratio.
    
    Args:
        original_size: Original model size
        compressed_size: Compressed model size
        
    Returns:
        Compression ratio (original/compressed)
    """
    if compressed_size == 0:
        return float('inf')
    return original_size / compressed_size


def calculate_speedup(
    original_latency: float, 
    compressed_latency: float
) -> float:
    """Calculate inference speedup.
    
    Args:
        original_latency: Original model latency
        compressed_latency: Compressed model latency
        
    Returns:
        Speedup ratio (original/compressed)
    """
    if compressed_latency == 0:
        return float('inf')
    return original_latency / compressed_latency
