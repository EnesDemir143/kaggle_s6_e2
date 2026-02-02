# 🫀 Heart Disease Prediction with TabM

A complete machine learning pipeline for Kaggle Playground Series S6E2 competition using TabM model with Optuna hyperparameter optimization.

## 🏆 Competition

**Kaggle Playground Series S6E2** - Predict the probability of heart disease based on patient health metrics.

- **Metric**: AUC-ROC
- **Task**: Binary Classification

## 📁 Project Structure

```
predicting_heart_disease/
├── data/
│   └── playground-series-s6e2/
│       ├── train.csv          # 630,000 samples
│       └── test.csv           # 420,000 samples
├── src/
│   ├── train/
│   │   └── train.py           # Main training script
│   └── utils/
│       ├── config.py          # Configuration dataclass
│       ├── logger.py          # Experiment logger
│       └── preprocessing.py   # Data preprocessing
├── notebook/
│   ├── 01_data_exploration.ipynb
│   └── kaggle_submission.ipynb  # Kaggle competition notebook
├── runs/                      # Training runs & checkpoints
└── README.md
```

## 🚀 Features

### Model: TabM
- State-of-the-art tabular deep learning model
- Ensemble of `k` predictors for robust predictions
- Numerical embeddings: `LinearReLU`, `PiecewiseLinear`, `Periodic`

### Preprocessing
- **4 Scaler Options**: StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
- Categorical encoding with LabelEncoder
- Stratified train/validation split (80/20)

### Hyperparameter Optimization
- **Optuna** with TPE sampler
- Median pruner for efficient trial termination
- Search space:
  - `scaler_type`: [standard, minmax, power, quantile]
  - `n_blocks`: [2, 3, 4]
  - `d_block`: [128, 192, 256, 384]
  - `k`: [16, 32, 64]
  - `lr`: [1e-4, 1e-2]
  - `weight_decay`: [1e-6, 1e-3]
  - `embedding_type`: [linear_relu, piecewise_linear]

### Training
- AdamW optimizer
- CosineAnnealingLR scheduler
- Gradient clipping (max_norm=1.0)
- Early stopping based on validation AUC-ROC
- Best model checkpointing

## 🛠️ Installation

```bash
# Clone and setup
cd predicting_heart_disease
uv sync

# Install dependencies
uv pip install tabm rtdl-num-embeddings optuna
```

## 💻 Usage

### Local Training

```bash
# Single model training
uv run python -m src.train.train --no_optuna --epochs 100

# With Optuna hyperparameter tuning
uv run python -m src.train.train --n_trials 50 --epochs 100
```

### Kaggle Submission

1. Upload `notebook/kaggle_submission.ipynb` to Kaggle
2. Enable GPU accelerator (Settings → GPU T4 x2)
3. Run all cells
4. Submit generated `submission.csv`

## 📊 Results

| Metric | Score |
|--------|-------|
| Validation AUC-ROC | ~0.95 |
| Validation Accuracy | ~88% |

## 🔧 Configuration

Key parameters in `src/utils/config.py`:

```python
@dataclass
class Config:
    # Data
    val_ratio: float = 0.2
    
    # Model
    n_blocks: int = 3
    d_block: int = 256
    k: int = 32
    
    # Training
    epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
```

## 📈 Logs & Monitoring

Training runs are saved to `runs/` with:
- `config.yaml` - Run configuration
- `train_metrics.csv` - Training metrics per epoch
- `val_metrics.csv` - Validation metrics per epoch
- `models/best_model.pt` - Best model checkpoint
- `figs/` - Training curves & evaluation plots

## 🔬 Tech Stack

- **Model**: TabM (tabm)
- **Embeddings**: rtdl-num-embeddings
- **HPO**: Optuna
- **Framework**: PyTorch
- **Data**: pandas, scikit-learn

## 📝 License

MIT License

## 🙏 Acknowledgments

- [TabM Paper](https://arxiv.org/abs/2310.18473)
- Kaggle Playground Series
