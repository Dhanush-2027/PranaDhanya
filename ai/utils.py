import os
import random
import time
import numpy as np
import tensorflow as tf
import logging
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from ai.config import LOG_DIR, RANDOM_SEED

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "training_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AI_Pipeline")

def set_seeds(seed=RANDOM_SEED):
    """Set seeds for reproducibility across random, numpy and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seeds set to {seed}")

def detect_device():
    """Detect CUDA / GPU devices and display configuration details."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            device_name = gpus[0].name
            logger.info(f"GPU Support Detected: {len(gpus)} GPU(s) available. Primary: {device_name}")
            # Log CUDA information
            logger.info(f"TensorFlow Built with CUDA: {tf.test.is_built_with_cuda()}")
            return "GPU"
        except RuntimeError as e:
            logger.error(f"Error initializing GPU memory growth: {e}")
            return "CPU"
    else:
        logger.info("GPU unavailable. Defaulting to CPU.")
        return "CPU"

def get_system_usage():
    """Get CPU, RAM and GPU usage if available."""
    usage = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": round(psutil.virtual_memory().used / (1024 ** 3), 2),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu_info": "Unavailable"
    }
    
    # Check GPU memory via TF if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # We can log GPU availability as OK
            usage["gpu_info"] = f"Detected {len(gpus)} GPU(s) active"
        except Exception:
            pass
            
    return usage

def plot_and_save_curves(history, output_dir, name):
    """Plot and save training history curves (Loss, Accuracy, Precision, Recall, etc.)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify available metrics
    metrics = history.history.keys()
    
    # Loss curve
    if 'loss' in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in metrics:
            plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'{name} - Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"{name.lower()}_loss_curve.png")
        plt.close()

    # Accuracy curve
    acc_key = next((k for k in metrics if 'accuracy' in k or 'acc' in k), None)
    if acc_key:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[acc_key], label='Train Accuracy')
        val_acc_key = 'val_' + acc_key
        if val_acc_key in metrics:
            plt.plot(history.history[val_acc_key], label='Val Accuracy')
        plt.title(f'{name} - Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"{name.lower()}_accuracy_curve.png")
        plt.close()

    # Precision curve
    prec_key = next((k for k in metrics if 'precision' in k or 'prec' in k), None)
    if prec_key:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[prec_key], label='Train Precision')
        val_prec_key = 'val_' + prec_key
        if val_prec_key in metrics:
            plt.plot(history.history[val_prec_key], label='Val Precision')
        plt.title(f'{name} - Precision Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"{name.lower()}_precision_curve.png")
        plt.close()

    # Recall curve
    rec_key = next((k for k in metrics if 'recall' in k or 'rec' in k), None)
    if rec_key:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[rec_key], label='Train Recall')
        val_rec_key = 'val_' + rec_key
        if val_rec_key in metrics:
            plt.plot(history.history[val_rec_key], label='Val Recall')
        plt.title(f'{name} - Recall Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"{name.lower()}_recall_curve.png")
        plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_dir, name):
    """Generate and save confusion matrix plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{name} - Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / f"{name.lower()}_confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true_onehot, y_pred_probs, classes, output_dir, name):
    """Generate and save ROC curve plot for multi-class or binary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    n_classes = len(classes)
    if n_classes == 2:
        # Binary ROC
        fpr, tpr, _ = roc_curve(y_true_onehot, y_pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    else:
        # Multiclass ROC (One-vs-Rest)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC of class {classes[i]} (area = {roc_auc:.4f})')
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} - ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_dir / f"{name.lower()}_roc_curve.png")
    plt.close()
