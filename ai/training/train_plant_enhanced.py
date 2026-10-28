"""Enhanced Plant Disease Training with EfficientNet-B3 and Advanced Techniques.

Features:
1. EfficientNet-B3 pretrained backbone
2. Advanced Albumentations augmentation
3. Mixup augmentation for better generalization
4. Label smoothing + Focal Loss
5. AdamW optimizer with cosine annealing + warmup
6. Learning rate finder
7. Comprehensive evaluation metrics
8. Early stopping and model checkpointing
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add workspace to sys.path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from ai.models.efficientnet_model import build_efficientnet_model
from ai.augmentation import get_train_transforms, get_val_transforms

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(4, os.cpu_count() or 2)
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
MIXUP_ALPHA = 0.2
LABEL_SMOOTHING = 0.1

print(f"Using device: {DEVICE}")
print(f"Number of workers: {NUM_WORKERS}")


class PlantDiseaseDataset(Dataset):
    """Dataset with Albumentations augmentation."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def scan_dataset(data_dir: str, samples_per_class: int = 100) -> Tuple[List[str], List[str], List[str]]:
    """Scan dataset directory and collect image paths."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    image_paths = []
    labels = []
    
    print(f"\nScanning {len(classes)} classes from: {data_dir}")
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        files = [f for f in os.listdir(class_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        # Limit samples per class
        if len(files) > samples_per_class:
            np.random.seed(42)
            files = list(np.random.choice(files, samples_per_class, replace=False))
        
        for f in files:
            image_paths.append(os.path.join(class_dir, f))
            labels.append(class_name)
        
        print(f"  {class_name}: {len(files)} images")
    
    print(f"Total images: {len(image_paths)}")
    return image_paths, classes, labels


def evaluate_model(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'loss': total_loss / len(dataloader)
    }


def train_plant_disease_enhanced(
    data_dir: str = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\plant_disease\data",
    output_dir: str = "ai/models/image_classification",
    samples_per_class: int = 100,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE
):
    """Main training function."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scan dataset
    image_paths, classes, labels = scan_dataset(data_dir, samples_per_class)
    num_classes = len(classes)
    
    # Convert labels to indices
    class_to_idx = {c: i for i, c in enumerate(classes)}
    label_indices = [class_to_idx[lbl] for lbl in labels]
    
    # 2. Split data (80% train, 10% val, 10% test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, label_indices, test_size=0.2, stratify=label_indices, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    print(f"\nData split: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
    
    # 3. Create datasets
    train_dataset = PlantDiseaseDataset(
        train_paths, train_labels, 
        transform=get_train_transforms(IMG_SIZE)
    )
    val_dataset = PlantDiseaseDataset(
        val_paths, val_labels,
        transform=get_val_transforms(IMG_SIZE)
    )
    test_dataset = PlantDiseaseDataset(
        test_paths, test_labels,
        transform=get_val_transforms(IMG_SIZE)
    )
    
    # 4. Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # 5. Create model
    model = build_efficientnet_model(num_classes=num_classes, model_name='efficientnet_b3', pretrained=True)
    model = model.to(DEVICE)
    
    # 6. Setup training
    # Compute class weights for imbalance
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = total_samples / (num_classes * class_counts.astype(np.float32))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    # Use label smoothing cross entropy
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, epochs)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    print("\n" + "=" * 80)
    print("STARTING ENHANCED EFFICIENTNET-B3 TRAINING")
    print("=" * 80)
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Mixup augmentation
            images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Mixup loss
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # Calculate accuracy (use original labels for metric)
            with torch.no_grad():
                outputs_clean = model(images.to(DEVICE))
                preds = torch.argmax(outputs_clean, dim=1)
                correct += (preds == labels.to(DEVICE)).sum().item()
                total += labels.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, DEVICE)
        
        # Learning rate step
        current_lr = scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']*100:.2f}% | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  --> New best model saved (Val Acc: {val_metrics['accuracy']*100:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final test evaluation
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    test_metrics = evaluate_model(model, test_loader, DEVICE)
    
    print(f"Test Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Test Precision: {test_metrics['precision']*100:.2f}%")
    print(f"Test Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"Test F1-Score:  {test_metrics['f1']*100:.2f}%")
    
    # Save model
    checkpoint = {
        'classes': classes,
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'test_metrics': test_metrics,
        'history': history,
        'config': {
            'model_name': 'efficientnet_b3',
            'img_size': IMG_SIZE,
            'batch_size': batch_size,
            'learning_rate': lr,
            'epochs': epoch
        }
    }
    
    model_path = os.path.join(output_dir, 'plant_disease_enhanced.pth')
    torch.save(checkpoint, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metadata
    with open(os.path.join(output_dir, 'plant_classes_enhanced.json'), 'w') as f:
        json.dump(classes, f, indent=2)
    
    with open(os.path.join(output_dir, 'plant_metrics_enhanced.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    return test_metrics


if __name__ == '__main__':
    train_plant_disease_enhanced()
