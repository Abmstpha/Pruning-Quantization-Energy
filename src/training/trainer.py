"""Training and evaluation utilities."""

import time
from contextlib import nullcontext
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    """Training and evaluation handler with AMP support."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.use_amp = (device.type == "cuda")
        self.amp_ctx = torch.cuda.amp.autocast if self.use_amp else nullcontext
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
    
    def train_epoch(
        self, 
        model: nn.Module, 
        loader: DataLoader, 
        optimizer: optim.Optimizer, 
        criterion: nn.Module
    ) -> float:
        """Train model for one epoch.
        
        Args:
            model: Model to train
            loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Average loss for the epoch
        """
        model.train()
        running_loss = 0.0
        
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            with self.amp_ctx():
                outputs = model(x)
                loss = criterion(outputs, y)
            
            optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
        
        return running_loss / max(1, len(loader))
    
    @torch.no_grad()
    def evaluate(
        self, 
        model: nn.Module, 
        loader: DataLoader
    ) -> Tuple[float, float, Optional[float]]:
        """Evaluate model on given data loader.
        
        Args:
            model: Model to evaluate
            loader: Data loader for evaluation
            
        Returns:
            Tuple of (accuracy, latency_ms_per_image, peak_memory_mb)
        """
        model = model.to(self.device)
        model.eval()
        
        correct = 0
        total = 0
        start_time = time.time()
        
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        with torch.inference_mode(), self.amp_ctx():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                
                logits = model(x)
                predictions = logits.argmax(1)
                correct += (predictions == y).sum().item()
                total += y.numel()
        
        accuracy = correct / max(1, total)
        latency_ms = (time.time() - start_time) / max(1, total) * 1000.0
        peak_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024**2) 
            if self.device.type == "cuda" else None
        )
        
        return accuracy, latency_ms, peak_memory_mb
    
    @torch.no_grad()
    def evaluate_on_device(
        self, 
        model: nn.Module, 
        loader: DataLoader, 
        target_device: torch.device
    ) -> Tuple[float, float, Optional[float]]:
        """Evaluate model on specific device (useful for quantized models).
        
        Args:
            model: Model to evaluate
            loader: Data loader for evaluation
            target_device: Device to run evaluation on
            
        Returns:
            Tuple of (accuracy, latency_ms_per_image, peak_memory_mb)
        """
        model = model.to(target_device).eval()
        
        correct = 0
        total = 0
        start_time = time.time()
        
        with torch.inference_mode():
            for x, y in loader:
                x = x.to(target_device, non_blocking=(target_device.type == "cuda"))
                y = y.to(target_device, non_blocking=(target_device.type == "cuda"))
                
                logits = model(x)
                predictions = logits.argmax(1)
                correct += (predictions == y).sum().item()
                total += y.numel()
        
        accuracy = correct / max(1, total)
        latency_ms = (time.time() - start_time) / max(1, total) * 1000.0
        peak_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024**2) 
            if target_device.type == "cuda" else None
        )
        
        return accuracy, latency_ms, peak_memory_mb
    
    def create_optimizer(
        self, 
        model: nn.Module, 
        lr: float = 1e-3, 
        weight_decay: float = 0.01
    ) -> optim.Optimizer:
        """Create AdamW optimizer for trainable parameters.
        
        Args:
            model: Model to optimize
            lr: Learning rate
            weight_decay: Weight decay factor
            
        Returns:
            AdamW optimizer
        """
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(
            trainable_params, 
            lr=lr, 
            weight_decay=weight_decay,
            fused=(self.device.type == "cuda")
        )
    
    def create_criterion(self, label_smoothing: float = 0.0) -> nn.Module:
        """Create cross-entropy loss criterion.
        
        Args:
            label_smoothing: Label smoothing factor
            
        Returns:
            CrossEntropyLoss criterion
        """
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
