"""Experiment configuration settings."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ExperimentConfig:
    """Configuration for pruning and quantization experiments."""
    
    # Dataset settings
    dataset_name: str = "Flowers102"
    num_classes: int = 102
    limit_trainset: int = 2000
    img_size: int = 224
    data_root: str = "data"
    
    # Training settings
    batch_train: int = 32
    batch_eval: int = 64
    epochs_baseline: int = 5
    ft_epochs_unstructured: int = 3
    ft_epochs_structured: int = 4
    epochs_qat: int = 6
    epochs_fulltrain: int = 6
    
    # Model settings
    backbones: List[str] = None
    
    # Compression settings
    sparsities: Tuple[float, ...] = (0.3, 0.6)
    flops_drop: float = 0.30
    
    # Experiment toggles
    run_baseline: bool = True
    run_unstructured_ptq: bool = True
    run_structured: bool = True
    run_qat: bool = True
    run_fulltrain: bool = False
    
    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    ptq_device: str = "cpu"
    
    # Output settings
    results_dir: str = "results"
    save_models: bool = False
    
    # Reproducibility
    seed: int = 1337
    
    # Energy tracking
    country_iso: str = "FRA"
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.backbones is None:
            self.backbones = ["resnet18", "mixer_b16_224"]
        
        # Convert string paths to Path objects
        self.results_dir = Path(self.results_dir)
        self.data_root = Path(self.data_root)
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


# Default configuration
DEFAULT_CONFIG = ExperimentConfig()


def get_config_from_env() -> ExperimentConfig:
    """Get configuration with environment variable overrides."""
    config = ExperimentConfig()
    
    # Override with environment variables if present
    if "LIMIT_TRAINSET" in os.environ:
        config.limit_trainset = int(os.environ["LIMIT_TRAINSET"])
    
    if "BATCH_TRAIN" in os.environ:
        config.batch_train = int(os.environ["BATCH_TRAIN"])
    
    if "BATCH_EVAL" in os.environ:
        config.batch_eval = int(os.environ["BATCH_EVAL"])
    
    if "EPOCHS_BASELINE" in os.environ:
        config.epochs_baseline = int(os.environ["EPOCHS_BASELINE"])
    
    if "RESULTS_DIR" in os.environ:
        config.results_dir = Path(os.environ["RESULTS_DIR"])
    
    if "CC_ISO" in os.environ:
        config.country_iso = os.environ["CC_ISO"]
    
    if "DEVICE" in os.environ:
        config.device = os.environ["DEVICE"]
    
    return config
