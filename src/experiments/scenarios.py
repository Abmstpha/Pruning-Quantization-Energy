"""Experiment scenarios for pruning and quantization evaluation."""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

from ..data import create_dataloaders
from ..models import get_model_factory, freeze_all_but_head, unfreeze_all
from ..training import Trainer
from ..compression import (
    global_unstructured_prune,
    structured_channel_prune_resnet,
    structured_channel_prune_mixer,
    apply_ptq_dynamic,
    replace_head_with_qat,
    recalibrate_batch_norm,
    warmup_qat_observers
)
from ..utils import (
    count_parameters,
    estimate_flops,
    calculate_model_size,
    track_emissions
)


class ExperimentRunner:
    """Orchestrates different compression experiments."""
    
    def __init__(
        self,
        device: torch.device,
        results_dir: Path,
        ptq_device: torch.device = None
    ):
        self.device = device
        self.results_dir = results_dir
        self.ptq_device = ptq_device or torch.device("cpu")
        self.trainer = Trainer(device)
        
        # Results columns
        self.columns = [
            "Backbone", "Scenario", "Dataset", "Top-1 Acc.", "Model Size (MB)",
            "Params (M)", "FLOPs (G)", "Energy (kWh)", "Emissions (kgCO2eq)",
            "Train Time (s)", "Latency (ms/img)", "Peak Mem (MB)", "Eval Device"
        ]
    
    def run_baseline_scenario(
        self,
        backbone_name: str,
        train_dataset,
        test_dataset,
        num_classes: int,
        epochs: int = 10,
        batch_train: int = 32,
        batch_eval: int = 64
    ) -> pd.DataFrame:
        """Run baseline fine-tuning scenario.
        
        Args:
            backbone_name: Name of backbone architecture
            train_dataset: Training dataset
            test_dataset: Test dataset
            num_classes: Number of output classes
            epochs: Number of training epochs
            batch_train: Training batch size
            batch_eval: Evaluation batch size
            
        Returns:
            DataFrame with experiment results
        """
        results_df = pd.DataFrame(columns=self.columns)
        
        # Create model
        model_factory = get_model_factory(backbone_name)
        model = model_factory(num_classes, pretrained=True).to(self.device)
        freeze_all_but_head(model)
        
        # Create data loaders
        train_loader, test_loader = create_dataloaders(
            train_dataset, test_dataset, batch_train, batch_eval, self.device.type
        )
        
        # Training
        optimizer = self.trainer.create_optimizer(model, lr=1e-3)
        criterion = self.trainer.create_criterion(label_smoothing=0.1)
        
        start_time = time.time()
        with track_emissions(f"baseline::{backbone_name}", self.results_dir):
            for epoch in range(epochs):
                self.trainer.train_epoch(model, train_loader, optimizer, criterion)
        
        train_time = time.time() - start_time
        energy_data = track_emissions.last_results
        
        # Evaluation
        accuracy, latency_ms, peak_mem = self.trainer.evaluate(model, test_loader)
        
        # Metrics
        model_size = calculate_model_size(model)
        params_m = count_parameters(model)
        flops_g = estimate_flops(model)
        
        # Record results
        row = {
            "Backbone": backbone_name,
            "Scenario": "Baseline FT",
            "Dataset": "Flowers102",
            "Top-1 Acc.": accuracy,
            "Model Size (MB)": model_size,
            "Params (M)": params_m,
            "FLOPs (G)": flops_g,
            "Energy (kWh)": energy_data.get("energy_kwh"),
            "Emissions (kgCO2eq)": energy_data.get("emissions_kg"),
            "Train Time (s)": train_time,
            "Latency (ms/img)": latency_ms,
            "Peak Mem (MB)": peak_mem,
            "Eval Device": self.device.type,
        }
        
        return pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    
    def run_unstructured_ptq_scenario(
        self,
        backbone_name: str,
        train_dataset,
        test_dataset,
        num_classes: int,
        sparsities: Tuple[float, ...] = (0.3, 0.6),
        ft_epochs: int = 3,
        batch_train: int = 32,
        batch_eval: int = 64
    ) -> pd.DataFrame:
        """Run unstructured pruning + PTQ scenario.
        
        Args:
            backbone_name: Name of backbone architecture
            train_dataset: Training dataset
            test_dataset: Test dataset
            num_classes: Number of output classes
            sparsities: Sparsity levels to test
            ft_epochs: Fine-tuning epochs after pruning
            batch_train: Training batch size
            batch_eval: Evaluation batch size
            
        Returns:
            DataFrame with experiment results
        """
        results_df = pd.DataFrame(columns=self.columns)
        
        for sparsity in sparsities:
            # Create model
            model_factory = get_model_factory(backbone_name)
            model = model_factory(num_classes, pretrained=True).to(self.device)
            freeze_all_but_head(model)
            
            # Apply unstructured pruning
            global_unstructured_prune(model, sparsity)
            
            # Create data loaders
            train_loader, test_loader = create_dataloaders(
                train_dataset, test_dataset, batch_train, batch_eval, self.device.type
            )
            
            # Fine-tuning after pruning
            optimizer = self.trainer.create_optimizer(model, lr=1e-3)
            criterion = self.trainer.create_criterion(label_smoothing=0.1)
            
            start_time = time.time()
            with track_emissions(f"unstructured::{backbone_name}::s{int(sparsity*100)}", self.results_dir):
                for epoch in range(ft_epochs):
                    self.trainer.train_epoch(model, train_loader, optimizer, criterion)
            
            train_time = time.time() - start_time
            energy_ft = track_emissions.last_results
            
            # Apply PTQ
            quantized_model = apply_ptq_dynamic(model)
            
            # Evaluate quantized model
            with track_emissions(f"ptq_infer::{backbone_name}::s{int(sparsity*100)}", self.results_dir):
                q_accuracy, q_latency_ms, q_peak_mem = self.trainer.evaluate_on_device(
                    quantized_model, test_loader, self.ptq_device
                )
            energy_inf = track_emissions.last_results
            
            # Metrics
            float_size = calculate_model_size(model)
            try:
                quant_size = calculate_model_size(quantized_model)
            except Exception:
                quant_size = float_size
            
            params_m = count_parameters(model)
            flops_g = estimate_flops(model)
            
            # Record results
            row = {
                "Backbone": backbone_name,
                "Scenario": f"Unstructured+PTQ (s={int(sparsity*100)}%)",
                "Dataset": "Flowers102",
                "Top-1 Acc.": q_accuracy,
                "Model Size (MB)": quant_size,
                "Params (M)": params_m,
                "FLOPs (G)": flops_g,
                "Energy (kWh)": energy_inf.get("energy_kwh"),
                "Emissions (kgCO2eq)": energy_inf.get("emissions_kg"),
                "Train Time (s)": train_time,
                "Latency (ms/img)": q_latency_ms,
                "Peak Mem (MB)": q_peak_mem,
                "Eval Device": self.ptq_device.type,
            }
            
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        
        return results_df
    
    def run_structured_scenario(
        self,
        backbone_name: str,
        train_dataset,
        test_dataset,
        num_classes: int,
        flops_drop: float = 0.30,
        ft_epochs: int = 5,
        batch_train: int = 32,
        batch_eval: int = 64
    ) -> pd.DataFrame:
        """Run structured pruning scenario.
        
        Args:
            backbone_name: Name of backbone architecture
            train_dataset: Training dataset
            test_dataset: Test dataset
            num_classes: Number of output classes
            flops_drop: Target FLOPs reduction
            ft_epochs: Fine-tuning epochs after pruning
            batch_train: Training batch size
            batch_eval: Evaluation batch size
            
        Returns:
            DataFrame with experiment results
        """
        results_df = pd.DataFrame(columns=self.columns)
        
        # Create model
        model_factory = get_model_factory(backbone_name)
        model = model_factory(num_classes, pretrained=True).to(self.device)
        freeze_all_but_head(model)
        
        # Apply structured pruning
        if backbone_name == "resnet18":
            model = structured_channel_prune_resnet(model, flops_drop)
        elif backbone_name == "mixer_b16_224":
            model = structured_channel_prune_mixer(model, flops_drop)
        else:
            raise ValueError(f"Structured pruning not implemented for {backbone_name}")
        
        # Create data loaders
        train_loader, test_loader = create_dataloaders(
            train_dataset, test_dataset, batch_train, batch_eval, self.device.type
        )
        
        # Recalibrate BatchNorm
        recalibrate_batch_norm(model, train_loader, num_batches=10, device=self.device)
        
        # Fine-tuning after pruning
        optimizer = self.trainer.create_optimizer(model, lr=1e-3)
        criterion = self.trainer.create_criterion(label_smoothing=0.1)
        
        start_time = time.time()
        with track_emissions(f"structured::{backbone_name}::drop{int(flops_drop*100)}", self.results_dir):
            for epoch in range(ft_epochs):
                self.trainer.train_epoch(model, train_loader, optimizer, criterion)
        
        train_time = time.time() - start_time
        energy_data = track_emissions.last_results
        
        # Evaluation
        accuracy, latency_ms, peak_mem = self.trainer.evaluate(model, test_loader)
        
        # Metrics
        model_size = calculate_model_size(model)
        params_m = count_parameters(model)
        flops_g = estimate_flops(model)
        
        # Record results
        row = {
            "Backbone": backbone_name,
            "Scenario": f"Structured (~{int(flops_drop*100)}% FLOPs)",
            "Dataset": "Flowers102",
            "Top-1 Acc.": accuracy,
            "Model Size (MB)": model_size,
            "Params (M)": params_m,
            "FLOPs (G)": flops_g,
            "Energy (kWh)": energy_data.get("energy_kwh"),
            "Emissions (kgCO2eq)": energy_data.get("emissions_kg"),
            "Train Time (s)": train_time,
            "Latency (ms/img)": latency_ms,
            "Peak Mem (MB)": peak_mem,
            "Eval Device": self.device.type,
        }
        
        return pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    
    def run_qat_scenario(
        self,
        backbone_name: str,
        train_dataset,
        test_dataset,
        num_classes: int,
        epochs: int = 10,
        batch_train: int = 32,
        batch_eval: int = 64
    ) -> pd.DataFrame:
        """Run QAT (head-only) scenario.
        
        Args:
            backbone_name: Name of backbone architecture
            train_dataset: Training dataset
            test_dataset: Test dataset
            num_classes: Number of output classes
            epochs: Number of training epochs
            batch_train: Training batch size
            batch_eval: Evaluation batch size
            
        Returns:
            DataFrame with experiment results
        """
        results_df = pd.DataFrame(columns=self.columns)
        
        # Create model
        model_factory = get_model_factory(backbone_name)
        model = model_factory(num_classes, pretrained=True).to(self.device)
        freeze_all_but_head(model)
        
        # Apply QAT to head
        model = replace_head_with_qat(model)
        
        # Create data loaders
        train_loader, test_loader = create_dataloaders(
            train_dataset, test_dataset, batch_train, batch_eval, self.device.type
        )
        
        # Warm up QAT observers
        warmup_qat_observers(model, train_loader, num_batches=10, device=self.device)
        
        # Training
        optimizer = self.trainer.create_optimizer(model, lr=1e-3)
        criterion = self.trainer.create_criterion(label_smoothing=0.1)
        
        start_time = time.time()
        with track_emissions(f"qat::{backbone_name}", self.results_dir):
            for epoch in range(epochs):
                self.trainer.train_epoch(model, train_loader, optimizer, criterion)
        
        train_time = time.time() - start_time
        energy_data = track_emissions.last_results
        
        # Evaluation
        accuracy, latency_ms, peak_mem = self.trainer.evaluate(model, test_loader)
        
        # Metrics
        model_size = calculate_model_size(model)
        params_m = count_parameters(model)
        try:
            flops_g = estimate_flops(model)
        except Exception:
            flops_g = None
        
        # Record results
        row = {
            "Backbone": backbone_name,
            "Scenario": "QAT (head)",
            "Dataset": "Flowers102",
            "Top-1 Acc.": accuracy,
            "Model Size (MB)": model_size,
            "Params (M)": params_m,
            "FLOPs (G)": flops_g,
            "Energy (kWh)": energy_data.get("energy_kwh"),
            "Emissions (kgCO2eq)": energy_data.get("emissions_kg"),
            "Train Time (s)": train_time,
            "Latency (ms/img)": latency_ms,
            "Peak Mem (MB)": peak_mem,
            "Eval Device": self.device.type,
        }
        
        return pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
