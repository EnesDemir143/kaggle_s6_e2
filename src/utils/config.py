from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pathlib import Path


@dataclass
class Config:
    """Configuration for TabM training pipeline."""
    
    # Data
    data_path: str = "data/playground-series-s6e2/train.csv"
    test_path: str = "data/playground-series-s6e2/test.csv"
    val_ratio: float = 0.2
    seed: int = 42
    
    # Features
    n_num_features: int = 11  # Age, BP, Cholesterol, Max HR, ST depression, etc.
    cat_cardinalities: List[int] = field(default_factory=lambda: [2, 4, 2, 3, 2, 3, 4])
    # Sex(2), Chest pain type(4), FBS over 120(2), EKG results(3), 
    # Exercise angina(2), Slope of ST(3), Thallium(4)
    
    # Model - TabM
    d_out: int = 1  # Binary classification
    n_blocks: int = 3
    d_block: int = 256
    k: int = 32  # Ensemble size
    arch_type: str = "tabm"  # tabm, tabm-mini, tabm-packed
    embedding_type: str = "piecewise_linear"  # linear_relu, piecewise_linear, periodic
    
    # Training
    batch_size: int = 256
    epochs: int = 100
    lr: float = 2e-3
    weight_decay: float = 3e-4
    patience: int = 10  # Early stopping
    
    # Scaler
    scaler_type: str = "standard"  # standard, minmax, power, quantile
    
    # Optuna
    n_trials: int = 50
    study_name: str = "tabm_heart_disease"
    
    # Paths
    runs_dir: str = "runs"
    
    def save(self, path: Path):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def update(self, **kwargs):
        """Update config with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)