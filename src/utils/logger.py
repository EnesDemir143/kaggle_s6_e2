"""Experiment logger for heart disease prediction."""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, roc_curve, auc


class ExperimentLogger:
    """Logger for training experiments with CSV metrics and plotting."""
    
    def __init__(
        self,
        run_dir: Optional[Union[str, Path]] = None,
        model_name: str = "tabm",
        config: Optional[Dict] = None,
        resume: bool = False,
        runs_base_dir: str = "runs"
    ):
        self.resume = resume
        self.runs_base_dir = Path(runs_base_dir)
        
        if resume and run_dir:
            self.run_dir = Path(run_dir)
            if not self.run_dir.exists():
                raise ValueError(f"Run directory does not exist: {run_dir}")
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = f"{model_name}_{timestamp}"
            self.run_dir = self.runs_base_dir / run_name
        
        self.figs_dir = self.run_dir / "figs"
        self.models_dir = self.run_dir / "models"
        
        self._setup_directories()
        self._setup_logging()
        
        if config and not resume:
            self.save_config(config)
        
        self._csv_files: Dict[str, Any] = {}
        self._csv_writers: Dict[str, Any] = {}
        self._csv_headers_written: Dict[str, bool] = {}
    
    def _setup_directories(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.figs_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self):
        log_file = self.run_dir / "train.log"
        
        self.logger = logging.getLogger(f"heart_disease_{self.run_dir.name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        file_mode = 'a' if self.resume else 'w'
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        if self.resume:
            self.logger.info("=" * 60)
            self.logger.info("RESUMING TRAINING")
            self.logger.info("=" * 60)
    
    def save_config(self, config: Dict):
        config_path = self.run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        self.logger.info(f"Config saved to {config_path}")
    
    def load_config(self) -> Dict:
        config_path = self.run_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_csv_path(self, name: str) -> Path:
        return self.run_dir / f"{name}.csv"
    
    def _init_csv(self, name: str, headers: List[str]):
        csv_path = self._get_csv_path(name)
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        
        if self.resume and file_exists:
            self._csv_files[name] = open(csv_path, 'a', newline='')
            self._csv_writers[name] = csv.DictWriter(
                self._csv_files[name], 
                fieldnames=headers,
                extrasaction='ignore'
            )
            self._csv_headers_written[name] = True
        else:
            self._csv_files[name] = open(csv_path, 'w', newline='')
            self._csv_writers[name] = csv.DictWriter(
                self._csv_files[name],
                fieldnames=headers,
                extrasaction='ignore'
            )
            self._csv_writers[name].writeheader()
            self._csv_files[name].flush()
            self._csv_headers_written[name] = True
    
    def log_metrics(self, name: str, epoch: int, metrics: Dict[str, float]):
        all_metrics = {'epoch': epoch, **metrics}
        headers = list(all_metrics.keys())
        
        if name not in self._csv_writers:
            self._init_csv(name, headers)
        
        self._csv_writers[name].writerow(all_metrics)
        self._csv_files[name].flush()
    
    def load_metrics_csv(self, name: str) -> List[Dict]:
        csv_path = self._get_csv_path(name)
        if not csv_path.exists():
            return []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def get_last_epoch(self) -> int:
        metrics = self.load_metrics_csv('train_metrics')
        if not metrics:
            return 0
        return int(metrics[-1].get('epoch', 0))
    
    def log(self, message: str, level: str = "info"):
        getattr(self.logger, level.lower())(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def plot_metrics(
        self,
        train_csv: str = "train_metrics",
        val_csv: str = "val_metrics"
    ):
        train_data = self.load_metrics_csv(train_csv)
        val_data = self.load_metrics_csv(val_csv)
        
        if not train_data and not val_data:
            return
        
        self._plot_loss(train_data, val_data)
        self._plot_lr(train_data)
        self._plot_classification_metrics(val_data)
    
    def _plot_loss(self, train_data: List[Dict], val_data: List[Dict]):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if train_data:
            epochs = [int(d['epoch']) for d in train_data]
            train_loss = [float(d.get('loss', 0)) for d in train_data]
            ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        
        if val_data:
            epochs = [int(d['epoch']) for d in val_data]
            val_loss = [float(d.get('loss', 0)) for d in val_data if 'loss' in d]
            if val_loss:
                ax.plot(epochs[:len(val_loss)], val_loss, 'r-', label='Val Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training & Validation Loss', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figs_dir / 'loss.png', dpi=150)
        plt.close(fig)
    
    def _plot_lr(self, train_data: List[Dict]):
        if not train_data or 'lr' not in train_data[0]:
            return
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        epochs = [int(d['epoch']) for d in train_data]
        lrs = [float(d.get('lr', 0)) for d in train_data]
        
        ax.plot(epochs, lrs, 'g-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figs_dir / 'learning_rate.png', dpi=150)
        plt.close(fig)
    
    def _plot_classification_metrics(self, val_data: List[Dict]):
        if not val_data:
            return
        
        metrics_to_plot = ['auc_roc', 'accuracy', 'f1', 'precision', 'recall']
        available_metrics = [m for m in metrics_to_plot if m in val_data[0]]
        
        if not available_metrics:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = [int(d['epoch']) for d in val_data]
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for metric, color in zip(available_metrics, colors):
            values = [float(d.get(metric, 0)) for d in val_data]
            label = metric.upper().replace('_', '-')
            linewidth = 3 if metric == 'auc_roc' else 2
            ax.plot(epochs, values, color=color, label=label, linewidth=linewidth)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Classification Metrics', fontsize=14)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figs_dir / 'classification_metrics.png', dpi=150)
        plt.close(fig)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, epoch: int):
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        classes = ['Absence', 'Presence']
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            title=f'Confusion Matrix (Epoch {epoch})',
            ylabel='True label',
            xlabel='Predicted label'
        )
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(self.figs_dir / f'confusion_matrix_epoch_{epoch}.png', dpi=150)
        plt.close(fig)
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, epoch: int):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve (Epoch {epoch})', fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figs_dir / f'roc_curve_epoch_{epoch}.png', dpi=150)
        plt.close(fig)
    
    def close(self):
        for name, f in self._csv_files.items():
            if not f.closed:
                f.close()
        self._csv_files.clear()
        self._csv_writers.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
