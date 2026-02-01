# %% [markdown]
# # Heart Disease Prediction with TabM
# 
# This notebook implements a complete pipeline for the Kaggle Playground Series S6E2 competition:
# - Data Exploration & Visualization
# - Data Preprocessing with multiple scaler options
# - TabM model with Optuna hyperparameter tuning
# - Submission file generation

# %% [markdown]
# ## 0. Install Required Libraries
# 
# TabM and rtdl-num-embeddings are not available on Kaggle by default.

# %%
# Install required packages (uncomment on Kaggle)
# !pip install -q tabm rtdl-num-embeddings optuna

# %% [markdown]
# ## 1. Setup & Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import torch
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

warnings.filterwarnings("ignore")

# TabM imports
from tabm import TabM
from rtdl_num_embeddings import LinearReLUEmbeddings, PeriodicEmbeddings, PiecewiseLinearEmbeddings

# Set random seed for reproducibility
SEED = 42

def set_seed(seed):
    """Set all seeds for reproducibility."""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Set Optuna seed (via sampler at study creation)

def get_generator(seed):
    """Get a generator for DataLoader reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g

set_seed(SEED)
G = get_generator(SEED)

# %%
# Define paths
TRAIN_DATA_PATH = "/kaggle/input/playground-series-s6e2/train.csv"
TEST_DATA_PATH = "/kaggle/input/playground-series-s6e2/test.csv"

# Check device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## 2. Load Data

# %%
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nTrain columns: {train_df.columns.tolist()}")

# %%
train_df.head()

# %%
train_df.describe()

# %% [markdown]
# ## 3. Exploratory Data Analysis

# %% [markdown]
# ### 3.1 Target Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
target_counts = train_df['Heart Disease'].value_counts()
colors = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)
axes[0].pie(target_counts, labels=['Absence', 'Presence'], autopct='%1.1f%%', 
            colors=colors, explode=explode, shadow=True, startangle=90)
axes[0].set_title('Heart Disease Distribution (Pie Chart)', fontsize=14, fontweight='bold')

# Bar chart
sns.countplot(data=train_df, x='Heart Disease', palette=colors, ax=axes[1], hue='Heart Disease', legend=False)
axes[1].set_title('Heart Disease Distribution (Bar Chart)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Heart Disease')
axes[1].set_ylabel('Count')

for i, count in enumerate(target_counts.sort_index()):
    axes[1].text(i, count + 1000, f'{count:,}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"Absence count: {target_counts['Absence']:,} ({target_counts['Absence']/len(train_df)*100:.1f}%)")
print(f"Presence count: {target_counts['Presence']:,} ({target_counts['Presence']/len(train_df)*100:.1f}%)")

# %% [markdown]
# ### 3.2 Gender Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

gender_counts = train_df['Sex'].value_counts()
labels = ['Male (1)', 'Female (0)']
colors = ['#3498db', '#e74c3c']
explode = (0.05, 0)
axes[0].pie(gender_counts, labels=labels, autopct='%1.1f%%', colors=colors, 
            explode=explode, shadow=True, startangle=90)
axes[0].set_title('Gender Distribution (Pie Chart)', fontsize=14, fontweight='bold')

sns.countplot(data=train_df, x='Sex', palette=colors, ax=axes[1], hue='Sex', legend=False)
axes[1].set_xticklabels(['Female (0)', 'Male (1)'])
axes[1].set_title('Gender Distribution (Bar Chart)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Count')

for i, count in enumerate(gender_counts.sort_index()):
    axes[1].text(i, count + 1000, f'{count:,}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.3 Age Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train_df['Age'], bins=30, color='#3498db', edgecolor='white', alpha=0.7)
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Age Distribution (Histogram)', fontsize=14, fontweight='bold')
axes[0].axvline(train_df['Age'].mean(), color='red', linestyle='--', label=f'Mean: {train_df["Age"].mean():.1f}')
axes[0].axvline(train_df['Age'].median(), color='green', linestyle='--', label=f'Median: {train_df["Age"].median():.1f}')
axes[0].legend()

sns.boxplot(data=train_df, x='Heart Disease', y='Age', palette=['#2ecc71', '#e74c3c'], ax=axes[1])
axes[1].set_title('Age by Heart Disease Status', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Heart Disease')
axes[1].set_ylabel('Age')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.4 Correlation Heatmap

# %%
# Encode target for correlation
train_encoded = train_df.copy()
train_encoded['Heart Disease'] = train_encoded['Heart Disease'].map({'Absence': 0, 'Presence': 1})

plt.figure(figsize=(14, 10))
corr_matrix = train_encoded.drop('id', axis=1).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
            fmt='.2f', linewidths=0.5, square=True)
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Data Preprocessing

# %%
# Define feature columns
NUMERIC_COLS = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
CATEGORICAL_COLS = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
                    'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']

print(f"Numeric features: {NUMERIC_COLS}")
print(f"Categorical features: {CATEGORICAL_COLS}")

# %%
def get_scaler(scaler_type: str = 'standard'):
    """Get scaler based on type."""
    if scaler_type == 'standard':
        return StandardScaler()
    elif scaler_type == 'minmax':
        return MinMaxScaler()
    elif scaler_type == 'power':
        return PowerTransformer(method='yeo-johnson')
    elif scaler_type == 'quantile':
        return QuantileTransformer(output_distribution='normal', n_quantiles=1000)
    else:
        return StandardScaler()

# %%
class HeartDiseaseDataset(Dataset):
    """PyTorch Dataset for Heart Disease data."""
    def __init__(self, x_num, x_cat, y=None):
        self.x_num = torch.FloatTensor(x_num) if x_num is not None else None
        self.x_cat = torch.LongTensor(x_cat) if x_cat is not None else None
        self.y = torch.FloatTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.x_num)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_num[idx], self.x_cat[idx], self.y[idx]
        return self.x_num[idx], self.x_cat[idx]

# %%
def prepare_data(train_df, test_df, scaler_type='standard', val_ratio=0.2, seed=42):
    """Prepare data for training."""
    
    # Encode target
    y_train = train_df['Heart Disease'].map({'Absence': 0, 'Presence': 1}).values
    
    # Split train/val
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=val_ratio, random_state=seed, stratify=y_train
    )
    
    # Prepare numeric features
    scaler = get_scaler(scaler_type)
    x_num_train = scaler.fit_transform(train_df[NUMERIC_COLS].values[train_idx])
    x_num_val = scaler.transform(train_df[NUMERIC_COLS].values[val_idx])
    x_num_test = scaler.transform(test_df[NUMERIC_COLS].values)
    
    # Prepare categorical features
    cat_encoders = {}
    x_cat_train = np.zeros((len(train_idx), len(CATEGORICAL_COLS)), dtype=np.int64)
    x_cat_val = np.zeros((len(val_idx), len(CATEGORICAL_COLS)), dtype=np.int64)
    x_cat_test = np.zeros((len(test_df), len(CATEGORICAL_COLS)), dtype=np.int64)
    
    for i, col in enumerate(CATEGORICAL_COLS):
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]]))
        cat_encoders[col] = le
        x_cat_train[:, i] = le.transform(train_df[col].values[train_idx])
        x_cat_val[:, i] = le.transform(train_df[col].values[val_idx])
        x_cat_test[:, i] = le.transform(test_df[col].values)
    
    # Get cardinalities
    cat_cardinalities = [len(cat_encoders[col].classes_) for col in CATEGORICAL_COLS]
    
    # Create datasets
    train_dataset = HeartDiseaseDataset(x_num_train, x_cat_train, y_train[train_idx])
    val_dataset = HeartDiseaseDataset(x_num_val, x_cat_val, y_train[val_idx])
    test_dataset = HeartDiseaseDataset(x_num_test, x_cat_test, None)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'cat_cardinalities': cat_cardinalities,
        'n_num_features': len(NUMERIC_COLS),
        'x_num_train': x_num_train,
        'test_ids': test_df['id'].values
    }

# %% [markdown]
# ## 5. Model Definition

# %%
def get_num_embeddings(embedding_type, n_features, d_embedding=24, x_train=None):
    """Factory for numerical embeddings."""
    if embedding_type == "linear_relu":
        return LinearReLUEmbeddings(n_features, d_embedding)
    elif embedding_type == "piecewise_linear":
        if x_train is not None:
            from rtdl_num_embeddings import compute_bins
            bins = compute_bins(torch.from_numpy(x_train).float(), n_bins=32)
            return PiecewiseLinearEmbeddings(bins, d_embedding, activation=True, version='B')
        return LinearReLUEmbeddings(n_features, d_embedding)
    elif embedding_type == "periodic":
        return PeriodicEmbeddings(n_features, d_embedding)
    else:
        return LinearReLUEmbeddings(n_features, d_embedding)

# %%
def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for x_num, x_cat, y in train_loader:
        x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        
        # TabM returns (batch_size, k, 1) - average over k
        if len(logits.shape) == 3:
            logits = logits.squeeze(-1)  # (batch_size, k)
            loss = criterion(logits, y.unsqueeze(-1).expand_as(logits)).mean()
        else:
            loss = criterion(logits.squeeze(), y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# %%
def evaluate(model, data_loader, device):
    """Evaluate model."""
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                x_num, x_cat, y = batch
                y = y.to(device)
            else:
                x_num, x_cat = batch
                y = None
            
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            logits = model(x_num, x_cat)
            
            if len(logits.shape) == 3:
                logits = logits.squeeze(-1).mean(dim=1)
            else:
                logits = logits.squeeze()
            
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            
            if y is not None:
                all_labels.append(y.cpu().numpy())
                total_loss += criterion(logits, y).item()
    
    all_probs = np.concatenate(all_probs)
    
    if len(all_labels) > 0:
        all_labels = np.concatenate(all_labels)
        metrics = {
            'loss': total_loss / len(data_loader),
            'auc_roc': roc_auc_score(all_labels, all_probs),
            'accuracy': accuracy_score(all_labels, (all_probs > 0.5).astype(int)),
            'f1': f1_score(all_labels, (all_probs > 0.5).astype(int))
        }
        return metrics, all_probs, all_labels
    
    return None, all_probs, None

# %% [markdown]
# ## 6. Optuna Hyperparameter Tuning

# %%
def objective(trial, data_info, device):
    """Optuna objective function."""
    
    # Sample hyperparameters
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax', 'power', 'quantile'])
    n_blocks = trial.suggest_int('n_blocks', 2, 4)
    d_block = trial.suggest_categorical('d_block', [128, 192, 256, 384])
    k = trial.suggest_categorical('k', [16, 32, 64])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    embedding_type = trial.suggest_categorical('embedding_type', ['linear_relu', 'piecewise_linear'])
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    
    # Prepare data with selected scaler
    data = prepare_data(train_df, test_df, scaler_type=scaler_type)
    
    train_loader = DataLoader(data['train_dataset'], batch_size=batch_size, shuffle=True, generator=get_generator(SEED))
    val_loader = DataLoader(data['val_dataset'], batch_size=batch_size, shuffle=False)
    
    # Create model
    num_embeddings = get_num_embeddings(
        embedding_type, 
        data['n_num_features'],
        x_train=data['x_num_train']
    )
    
    model = TabM.make(
        n_num_features=data['n_num_features'],
        cat_cardinalities=data['cat_cardinalities'],
        num_embeddings=num_embeddings,
        n_blocks=n_blocks,
        d_block=d_block,
        d_out=1,
        arch_type='tabm',
        k=k
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, 31):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        
        if val_metrics['auc_roc'] > best_auc:
            best_auc = val_metrics['auc_roc']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        # Pruning
        trial.report(val_metrics['auc_roc'], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_auc

# %%
# Run Optuna study
print("Starting Optuna hyperparameter tuning...")

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)

study.optimize(
    lambda trial: objective(trial, None, DEVICE),
    n_trials=30,  # Adjust based on time constraints
    timeout=7200,  # 2 hours max
    show_progress_bar=True
)

print(f"\nBest trial:")
print(f"  AUC-ROC: {study.best_trial.value:.4f}")
print(f"  Params: {study.best_trial.params}")

# %% [markdown]
# ## 7. Train Final Model with Best Hyperparameters

# %%
# Get best params
best_params = study.best_trial.params
print("Training final model with best hyperparameters...")

# Prepare data with best scaler
data = prepare_data(train_df, test_df, scaler_type=best_params['scaler_type'])

train_loader = DataLoader(data['train_dataset'], batch_size=best_params['batch_size'], shuffle=True, generator=G)
val_loader = DataLoader(data['val_dataset'], batch_size=best_params['batch_size'], shuffle=False)
test_loader = DataLoader(data['test_dataset'], batch_size=best_params['batch_size'], shuffle=False)

# Create final model
num_embeddings = get_num_embeddings(
    best_params['embedding_type'],
    data['n_num_features'],
    x_train=data['x_num_train']
)

final_model = TabM.make(
    n_num_features=data['n_num_features'],
    cat_cardinalities=data['cat_cardinalities'],
    num_embeddings=num_embeddings,
    n_blocks=best_params['n_blocks'],
    d_block=best_params['d_block'],
    d_out=1,
    arch_type='tabm',
    k=best_params['k']
).to(DEVICE)

optimizer = AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
scheduler = CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.BCEWithLogitsLoss()

# %%
# Training loop with more epochs
best_auc = 0
best_model_state = None
train_losses = []
val_aucs = []

for epoch in range(1, 101):  # 100 epochs for final model
    train_loss = train_epoch(final_model, train_loader, optimizer, criterion, DEVICE)
    val_metrics, _, _ = evaluate(final_model, val_loader, DEVICE)
    scheduler.step()
    
    train_losses.append(train_loss)
    val_aucs.append(val_metrics['auc_roc'])
    
    if val_metrics['auc_roc'] > best_auc:
        best_auc = val_metrics['auc_roc']
        best_model_state = final_model.state_dict().copy()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val AUC: {val_metrics['auc_roc']:.4f}")

print(f"\nBest Validation AUC-ROC: {best_auc:.4f}")

# Load best model
final_model.load_state_dict(best_model_state)

# %%
# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')

axes[1].plot(val_aucs)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC-ROC')
axes[1].set_title('Validation AUC-ROC')
axes[1].axhline(y=best_auc, color='r', linestyle='--', label=f'Best: {best_auc:.4f}')
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Evaluation & Visualization

# %%
# Final evaluation
val_metrics, val_probs, val_labels = evaluate(final_model, val_loader, DEVICE)

print("Final Validation Metrics:")
print(f"  AUC-ROC:  {val_metrics['auc_roc']:.4f}")
print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
print(f"  F1 Score: {val_metrics['f1']:.4f}")

# %%
# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(val_labels, (val_probs > 0.5).astype(int))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Absence', 'Presence'], yticklabels=['Absence', 'Presence'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')

# ROC Curve
fpr, tpr, _ = roc_curve(val_labels, val_probs)
axes[1].plot(fpr, tpr, label=f'AUC = {val_metrics["auc_roc"]:.4f}')
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Generate Submission

# %%
# Get test predictions
_, test_probs, _ = evaluate(final_model, test_loader, DEVICE)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': data['test_ids'],
    'Heart Disease': test_probs
})

# Save submission
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("Submission file saved!")
print(f"\nSubmission shape: {submission.shape}")
print(f"\nSubmission head:")
print(submission.head(10))

# %%
# Verify submission format
print(f"\nPrediction statistics:")
print(f"  Min: {test_probs.min():.4f}")
print(f"  Max: {test_probs.max():.4f}")
print(f"  Mean: {test_probs.mean():.4f}")
print(f"  Std: {test_probs.std():.4f}")
