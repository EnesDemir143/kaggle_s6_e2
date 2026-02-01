"""TabM training pipeline with Optuna hyperparameter tuning."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tabm import TabM
from rtdl_num_embeddings import LinearReLUEmbeddings, PiecewiseLinearEmbeddings, PeriodicEmbeddings

from src.utils.config import Config
from src.utils.preprocessing import load_data, create_dataloaders
from src.utils.logger import ExperimentLogger

def get_num_embeddings(
    embedding_type: str, 
    n_features: int, 
    d_embedding: int = 24,
    x_train: np.ndarray = None
):
    """Factory for numerical embeddings.
    
    Args:
        embedding_type: One of 'linear_relu', 'piecewise_linear', 'periodic', or 'none'
        n_features: Number of numeric features
        d_embedding: Embedding dimension
        x_train: Training data for computing bins (required for piecewise_linear)
    """
    if embedding_type == "linear_relu":
        return LinearReLUEmbeddings(n_features, d_embedding)
    elif embedding_type == "piecewise_linear":
        if x_train is None:
            # Fallback to linear_relu if no training data provided
            print("Warning: piecewise_linear requires x_train, falling back to linear_relu")
            return LinearReLUEmbeddings(n_features, d_embedding)
        from rtdl_num_embeddings import compute_bins
        bins = compute_bins(torch.from_numpy(x_train).float(), n_bins=32)
        return PiecewiseLinearEmbeddings(bins, d_embedding, activation=True, version='B')
    elif embedding_type == "periodic":
        return PeriodicEmbeddings(n_features, d_embedding)
    elif embedding_type == "none":
        return None
    else:
        return LinearReLUEmbeddings(n_features, d_embedding)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for x_num, x_cat, y in train_loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - TabM returns (batch, k, d_out)
        y_pred = model(x_num, x_cat)  # (B, k, 1)
        
        # Calculate loss for each ensemble member and average
        # BCEWithLogitsLoss expects (B, k, 1) vs (B, 1) -> broadcast y
        y_expanded = y.unsqueeze(1).expand_as(y_pred)  # (B, k, 1)
        loss = criterion(y_pred, y_expanded)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_probs = []
    all_labels = []
    
    for x_num, x_cat, y in val_loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)
        
        y_pred = model(x_num, x_cat)  # (B, k, 1)
        
        # Loss
        y_expanded = y.unsqueeze(1).expand_as(y_pred)
        loss = criterion(y_pred, y_expanded)
        total_loss += loss.item()
        n_batches += 1
        
        # Average probabilities across k ensemble members
        probs = torch.sigmoid(y_pred).mean(dim=1).squeeze(-1)  # (B,)
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.squeeze(-1).cpu().numpy())
    
    # Calculate metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)
    
    metrics = {
        'loss': total_loss / n_batches,
        'auc_roc': roc_auc_score(all_labels, all_probs),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0)
    }
    
    return metrics, all_probs, all_labels


def train_model(
    config: Config,
    train_loader,
    val_loader,
    info: dict,
    device: torch.device,
    logger: ExperimentLogger = None,
    trial: optuna.Trial = None
):
    """Train TabM model with given configuration."""
    
    # Create model
    num_embeddings = get_num_embeddings(
        config.embedding_type, 
        info['n_num_features'],
        x_train=info.get('x_num_train')
    )
    
    model = TabM.make(
        n_num_features=info['n_num_features'],
        cat_cardinalities=info['cat_cardinalities'],
        num_embeddings=num_embeddings,
        n_blocks=config.n_blocks,
        d_block=config.d_block,
        d_out=config.d_out,
        arch_type=config.arch_type,
        k=config.k
    ).to(device)
    
    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_auc = 0.0
    patience_counter = 0
    
    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Evaluate
        val_metrics, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        
        # Log
        if logger:
            logger.log_metrics('train_metrics', epoch, {'loss': train_loss, 'lr': current_lr})
            logger.log_metrics('val_metrics', epoch, val_metrics)
            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"AUC-ROC: {val_metrics['auc_roc']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f}"
            )
        
        # Save best model
        if val_metrics['auc_roc'] > best_auc:
            best_auc = val_metrics['auc_roc']
            patience_counter = 0
            
            if logger:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_auc': best_auc,
                    'config': config.__dict__
                }, logger.models_dir / 'best.pt')
                
                # Plot ROC curve for best model
                val_preds = (val_probs > 0.5).astype(int)
                logger.plot_confusion_matrix(val_labels, val_preds, epoch)
                logger.plot_roc_curve(val_labels, val_probs, epoch)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            if logger:
                logger.info(f"Early stopping at epoch {epoch}")
            break
        
        # Optuna pruning
        if trial:
            trial.report(val_metrics['auc_roc'], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    # Final plots
    if logger:
        logger.plot_metrics()
    
    return best_auc


def objective(trial: optuna.Trial, base_config: Config, df, device: torch.device):
    """Optuna objective function."""
    
    # Sample hyperparameters
    config = Config()
    config.scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax', 'power', 'quantile'])
    config.n_blocks = trial.suggest_int('n_blocks', 2, 4)
    config.d_block = trial.suggest_categorical('d_block', [128, 192, 256, 384])
    config.k = trial.suggest_categorical('k', [16, 24, 32])
    config.lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    config.weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    config.embedding_type = trial.suggest_categorical('embedding_type', ['linear_relu', 'piecewise_linear', 'periodic'])
    config.epochs = base_config.epochs
    config.batch_size = base_config.batch_size
    config.patience = base_config.patience
    config.seed = base_config.seed
    
    # Create dataloaders
    train_loader, val_loader, info = create_dataloaders(
        df,
        scaler_type=config.scaler_type,
        val_ratio=config.val_ratio,
        batch_size=config.batch_size,
        seed=config.seed
    )
    
    # Train without detailed logging for Optuna trials
    try:
        best_auc = train_model(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            info=info,
            device=device,
            logger=None,
            trial=trial
        )
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0
    
    return best_auc


def main():
    parser = argparse.ArgumentParser(description='TabM Training with Optuna')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs per trial')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_optuna', action='store_true', help='Train single model without Optuna')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    base_config = Config()
    base_config.epochs = args.epochs
    base_config.batch_size = args.batch_size
    base_config.patience = args.patience
    base_config.seed = args.seed
    
    df = load_data(base_config.data_path)
    print(f"Loaded {len(df)} samples")
    
    if args.no_optuna:
        # Single run training
        print("Training single model...")
        
        train_loader, val_loader, info = create_dataloaders(
            df,
            scaler_type=base_config.scaler_type,
            val_ratio=base_config.val_ratio,
            batch_size=base_config.batch_size,
            seed=base_config.seed
        )
        
        with ExperimentLogger(
            model_name=f"tabm_{base_config.scaler_type}",
            config=base_config.__dict__
        ) as logger:
            logger.info(f"Training on {info['train_size']} samples, validating on {info['val_size']} samples")
            logger.info(f"Categorical cardinalities: {info['cat_cardinalities']}")
            
            best_auc = train_model(
                config=base_config,
                train_loader=train_loader,
                val_loader=val_loader,
                info=info,
                device=device,
                logger=logger
            )
            
            logger.info(f"Training completed. Best AUC-ROC: {best_auc:.4f}")
            print(f"\nResults saved to: {logger.run_dir}")
    
    else:
        # Optuna hyperparameter tuning
        print(f"Starting Optuna study with {args.n_trials} trials...")
        
        study = optuna.create_study(
            study_name=base_config.study_name,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        study.optimize(
            lambda trial: objective(trial, base_config, df, device),
            n_trials=args.n_trials,
            show_progress_bar=True
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("OPTUNA STUDY RESULTS")
        print("=" * 60)
        print(f"Best trial AUC-ROC: {study.best_trial.value:.4f}")
        print("\nBest hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        
        # Train final model with best params
        print("\n" + "=" * 60)
        print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
        print("=" * 60)
        
        best_config = Config()
        best_config.epochs = args.epochs
        best_config.batch_size = args.batch_size
        best_config.patience = args.patience
        best_config.seed = args.seed
        
        for key, value in study.best_trial.params.items():
            setattr(best_config, key, value)
        
        train_loader, val_loader, info = create_dataloaders(
            df,
            scaler_type=best_config.scaler_type,
            val_ratio=best_config.val_ratio,
            batch_size=best_config.batch_size,
            seed=best_config.seed
        )
        
        with ExperimentLogger(
            model_name="tabm_best",
            config=best_config.__dict__
        ) as logger:
            logger.info(f"Training on {info['train_size']} samples, validating on {info['val_size']} samples")
            logger.info(f"Best hyperparameters from Optuna:")
            for key, value in study.best_trial.params.items():
                logger.info(f"  {key}: {value}")
            
            best_auc = train_model(
                config=best_config,
                train_loader=train_loader,
                val_loader=val_loader,
                info=info,
                device=device,
                logger=logger
            )
            
            logger.info(f"Final training completed. Best AUC-ROC: {best_auc:.4f}")
            print(f"\nFinal model saved to: {logger.run_dir}")


if __name__ == "__main__":
    main()
