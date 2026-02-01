"""Data preprocessing module for heart disease prediction."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, Literal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    PowerTransformer, 
    QuantileTransformer,
    LabelEncoder
)
import torch
from torch.utils.data import Dataset, DataLoader


# Feature columns
NUMERIC_COLS = [
    'Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression'
]

CATEGORICAL_COLS = [
    'Sex', 'Chest pain type', 'FBS over 120', 'EKG results',
    'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

TARGET_COL = 'Heart Disease'


def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare the heart disease dataset."""
    df = pd.read_csv(data_path)
    
    # Encode target: Presence=1, Absence=0
    df[TARGET_COL] = (df[TARGET_COL] == 'Presence').astype(int)
    
    return df


def get_scaler(scaler_type: str):
    """Factory function for scalers."""
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'power': PowerTransformer(method='yeo-johnson', standardize=True),
        'quantile': QuantileTransformer(output_distribution='normal', random_state=42)
    }
    if scaler_type not in scalers:
        raise ValueError(f"Unknown scaler type: {scaler_type}. Choose from {list(scalers.keys())}")
    return scalers[scaler_type]


def prepare_features(
    df: pd.DataFrame,
    scaler=None,
    cat_encoders: Optional[Dict[str, LabelEncoder]] = None,
    fit: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, any, Dict[str, LabelEncoder]]:
    """
    Prepare numeric and categorical features.
    
    Returns:
        x_num: Numeric features (scaled)
        x_cat: Categorical features (label encoded)
        y: Target
        scaler: Fitted scaler
        cat_encoders: Fitted label encoders
    """
    # Extract numeric features
    x_num = df[NUMERIC_COLS].values.astype(np.float32)
    
    # Scale numeric features
    if scaler is not None:
        if fit:
            x_num = scaler.fit_transform(x_num)
        else:
            x_num = scaler.transform(x_num)
    
    # Encode categorical features
    if cat_encoders is None:
        cat_encoders = {}
    
    x_cat_list = []
    for col in CATEGORICAL_COLS:
        if fit:
            if col not in cat_encoders:
                cat_encoders[col] = LabelEncoder()
            encoded = cat_encoders[col].fit_transform(df[col].values)
        else:
            encoded = cat_encoders[col].transform(df[col].values)
        x_cat_list.append(encoded)
    
    x_cat = np.column_stack(x_cat_list).astype(np.int64)
    
    # Target
    y = df[TARGET_COL].values.astype(np.float32)
    
    return x_num, x_cat, y, scaler, cat_encoders


def get_cat_cardinalities(cat_encoders: Dict[str, LabelEncoder]) -> list:
    """Get cardinalities for each categorical feature."""
    return [len(cat_encoders[col].classes_) for col in CATEGORICAL_COLS]


class HeartDiseaseDataset(Dataset):
    """PyTorch Dataset for heart disease data."""
    
    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray):
        self.x_num = torch.from_numpy(x_num).float()
        self.x_cat = torch.from_numpy(x_cat).long()
        self.y = torch.from_numpy(y).float().unsqueeze(1)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


def create_dataloaders(
    df: pd.DataFrame,
    scaler_type: str = 'standard',
    val_ratio: float = 0.2,
    batch_size: int = 256,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation dataloaders.
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        info: Dictionary with scaler, encoders, cardinalities
    """
    # Stratified split
    train_df, val_df = train_test_split(
        df, 
        test_size=val_ratio, 
        stratify=df[TARGET_COL],
        random_state=seed
    )
    
    # Get scaler
    scaler = get_scaler(scaler_type)
    
    # Prepare training features (fit scaler and encoders)
    x_num_train, x_cat_train, y_train, scaler, cat_encoders = prepare_features(
        train_df, scaler=scaler, fit=True
    )
    
    # Prepare validation features (transform only)
    x_num_val, x_cat_val, y_val, _, _ = prepare_features(
        val_df, scaler=scaler, cat_encoders=cat_encoders, fit=False
    )
    
    # Create datasets
    train_dataset = HeartDiseaseDataset(x_num_train, x_cat_train, y_train)
    val_dataset = HeartDiseaseDataset(x_num_val, x_cat_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Gather info
    info = {
        'scaler': scaler,
        'cat_encoders': cat_encoders,
        'cat_cardinalities': get_cat_cardinalities(cat_encoders),
        'n_num_features': len(NUMERIC_COLS),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'x_num_train': x_num_train  # For piecewise linear embeddings
    }
    
    return train_loader, val_loader, info


def compute_statistics(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute mean and std for numeric features."""
    x_num = df[NUMERIC_COLS].values.astype(np.float32)
    return {
        'mean': x_num.mean(axis=0),
        'std': x_num.std(axis=0)
    }
