#!/usr/bin/env python3
"""Main script to run all pruning and quantization experiments."""

import os
import random
import torch
from pathlib import Path

from configs import get_config_from_env
from src.data import load_flowers102
from src.experiments import ExperimentRunner
from insights import ResultsVisualizer


def setup_reproducibility(seed: int = 1337):
    """Setup reproducibility settings."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def setup_device(device_config: str = "auto") -> torch.device:
    """Setup compute device."""
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    print(f"Using device: {device}")
    return device


def main():
    """Main experiment runner."""
    print("🚀 Starting Pruning & Quantization Experiments")
    print("=" * 50)
    
    # Load configuration
    config = get_config_from_env()
    print(f"Configuration loaded: {config.to_dict()}")
    
    # Setup reproducibility
    setup_reproducibility(config.seed)
    
    # Setup devices
    device = setup_device(config.device)
    ptq_device = torch.device(config.ptq_device)
    
    # Load dataset
    print(f"\n📊 Loading {config.dataset_name} dataset...")
    train_dataset, test_dataset, num_classes = load_flowers102(
        limit=config.limit_trainset,
        img_size=config.img_size,
        data_root=str(config.data_root)
    )
    print(f"Dataset ready → train: {len(train_dataset)}, test: {len(test_dataset)}, classes: {num_classes}")
    
    # Initialize experiment runner
    runner = ExperimentRunner(
        device=device,
        results_dir=config.results_dir,
        ptq_device=ptq_device
    )
    
    # Run experiments for each backbone
    all_results = []
    
    for backbone in config.backbones:
        print(f"\n🔬 Running experiments for {backbone}")
        print("-" * 30)
        
        # 1. Baseline
        if config.run_baseline:
            print(f"  → Baseline fine-tuning...")
            baseline_results = runner.run_baseline_scenario(
                backbone, train_dataset, test_dataset, num_classes,
                epochs=config.epochs_baseline,
                batch_train=config.batch_train,
                batch_eval=config.batch_eval
            )
            all_results.append(baseline_results)
        
        # 2. Unstructured + PTQ
        if config.run_unstructured_ptq:
            print(f"  → Unstructured pruning + PTQ...")
            unstructured_results = runner.run_unstructured_ptq_scenario(
                backbone, train_dataset, test_dataset, num_classes,
                sparsities=config.sparsities,
                ft_epochs=config.ft_epochs_unstructured,
                batch_train=config.batch_train,
                batch_eval=config.batch_eval
            )
            all_results.append(unstructured_results)
        
        # 3. Structured pruning
        if config.run_structured:
            print(f"  → Structured pruning...")
            structured_results = runner.run_structured_scenario(
                backbone, train_dataset, test_dataset, num_classes,
                flops_drop=config.flops_drop,
                ft_epochs=config.ft_epochs_structured,
                batch_train=config.batch_train,
                batch_eval=config.batch_eval
            )
            all_results.append(structured_results)
        
        # 4. QAT
        if config.run_qat:
            print(f"  → Quantization-aware training...")
            qat_results = runner.run_qat_scenario(
                backbone, train_dataset, test_dataset, num_classes,
                epochs=config.epochs_qat,
                batch_train=config.batch_train,
                batch_eval=config.batch_eval
            )
            all_results.append(qat_results)
    
    # Combine all results
    if all_results:
        import pandas as pd
        final_results = pd.concat(all_results, ignore_index=True)
        
        # Save results
        results_csv = config.results_dir / "final_results.csv"
        results_xlsx = config.results_dir / "final_results.xlsx"
        
        final_results.to_csv(results_csv, index=False)
        try:
            final_results.to_excel(results_xlsx, index=False)
        except Exception:
            pass
        
        print(f"\n💾 Results saved to:")
        print(f"  • {results_csv}")
        print(f"  • {results_xlsx}")
        
        # Generate visualizations
        print(f"\n📈 Generating visualizations...")
        visualizer = ResultsVisualizer(final_results, config.results_dir / "plots")
        summary = visualizer.generate_all_plots()
        
        # Print summary
        print(f"\n📋 EXPERIMENT SUMMARY")
        print("=" * 30)
        for backbone, data in summary.items():
            print(f"\n[{backbone.upper()}]")
            print(f"  Best Accuracy: {data['best_accuracy']['experiment']} ({data['best_accuracy']['accuracy']:.2f}%)")
            print(f"  Smallest Model: {data['smallest_model']['experiment']} ({data['smallest_model']['size_mb']:.2f} MB)")
            print(f"  Best Efficiency: {data['best_efficiency']['experiment']} (Score: {data['best_efficiency']['efficiency_score']:.2f})")
        
        print(f"\n✅ All experiments completed successfully!")
        print(f"📁 Check results in: {config.results_dir}")
    
    else:
        print("⚠️ No experiments were run. Check your configuration.")


if __name__ == "__main__":
    main()
