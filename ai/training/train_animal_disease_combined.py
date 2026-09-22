"""
Combined Animal Disease Detection Training Script
=================================================

Trains ResNet9 on a combined dataset of:
- Cattle diseases (foot-and-mouth, lumpy skin disease, healthy)
- Dog skin diseases (demodicosis, dermatitis, fungal, healthy, etc.)
- Livestock (healthy goat, unhealthy goat)

Features:
- Proper optimizer gradient handling (zero_grad -> backward -> step)
- Gradient clipping to prevent exploding gradients
- Mixed precision training support for faster computation
- Unified class mapping to avoid naming conflicts
- Model checkpointing (saves best model during training)
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Add the parent folder of 'ai' to sys.path to enable module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.models.resnet9 import ResNet9


class CombinedAnimalDataset(Dataset):
    """
    Combined dataset loader for cattle, dog, and goat disease datasets.
    
    Automatically discovers and combines multiple animal disease datasets
    with unified class naming to prevent conflicts.
    """
    
    def __init__(self, cattle_dir=None, dog_dir=None, goat_dir=None, transform=None, max_samples_per_class=None):
        """
        Initialize dataset by scanning disease directories.
        
        Args:
            cattle_dir: Path to cattle diseases directory
            dog_dir: Path to dog diseases directory
            goat_dir: Path to goat/livestock diseases directory
            transform: torchvision transforms to apply to images
            max_samples_per_class: Maximum number of samples to keep per class
        """
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        
        # Define the three dataset directories (fallback to default CL path if not provided)
        self.dirs = {
            'cattle': cattle_dir or r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\cattle_diseases\Cows datasets",
            'dog': dog_dir or r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\dog_skin_disease\train",
            'goat': goat_dir or r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\livestock"
        }
        
        # Unified class mapping to avoid naming conflicts
        # Format: (source_dir_key, original_subdir_name) -> target_class_name
        self.class_mapping: Dict[Tuple[str, str], str] = {}
        
        # Discover classes from directory structure
        self._discover_classes()
        
        # Build class index mapping
        self.classes = sorted(list(set(self.class_mapping.values())))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Scan and collect all image files
        self._collect_samples()
        
        # Subsample if requested
        if max_samples_per_class is not None:
            class_samples = {}
            for path, class_idx in self.samples:
                if class_idx not in class_samples:
                    class_samples[class_idx] = []
                class_samples[class_idx].append((path, class_idx))
            
            new_samples = []
            for class_idx, samples in class_samples.items():
                new_samples.extend(samples[:max_samples_per_class])
            self.samples = new_samples
            print(f"Subsampled dataset to {len(self.samples)} images (max {max_samples_per_class} per class)")
        
        print(f"Total classes: {len(self.classes)}")
        for cls_name, idx in self.class_to_idx.items():
            print(f"  Class: {cls_name:30s} -> Index: {idx}")
        print(f"Total samples collected: {len(self.samples)}")
    
    def _discover_classes(self) -> None:
        """Discover and map all disease classes from directory structure."""
        
        # 1. Cattle diseases
        if os.path.exists(self.dirs['cattle']):
            for d in os.listdir(self.dirs['cattle']):
                path = os.path.join(self.dirs['cattle'], d)
                if os.path.isdir(path):
                    # Map to unified class name with 'cattle_' prefix
                    unified_name = f"cattle_{d.replace('-', '_').lower()}"
                    self.class_mapping[('cattle', d)] = unified_name
        
        # 2. Dog skin diseases
        if os.path.exists(self.dirs['dog']):
            for d in os.listdir(self.dirs['dog']):
                path = os.path.join(self.dirs['dog'], d)
                if os.path.isdir(path):
                    # Map to unified class name with 'dog_' prefix
                    unified_name = f"dog_{d.lower()}"
                    self.class_mapping[('dog', d)] = unified_name
        
        # 3. Goat/livestock diseases
        if os.path.exists(self.dirs['goat']):
            for d in os.listdir(self.dirs['goat']):
                path = os.path.join(self.dirs['goat'], d)
                if os.path.isdir(path):
                    # Map to unified class name with 'goat_' prefix
                    # healthy_goat -> goat_healthy, unhealthy_goat -> goat_unhealthy
                    if "unhealthy_goat" in d.lower():
                        unified_name = "goat_unhealthy"
                    elif "healthy_goat" in d.lower():
                        unified_name = "goat_healthy"
                    else:
                        unified_name = f"goat_{d.lower()}"
                    self.class_mapping[('goat', d)] = unified_name
    
    def _collect_samples(self) -> None:
        """Recursively collect all image samples from dataset directories."""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        for key, base_path in self.dirs.items():
            if not os.path.exists(base_path):
                print(f"Warning: Dataset path not found: {base_path}")
                continue
            
            for original_subdir in os.listdir(base_path):
                subdir_path = os.path.join(base_path, original_subdir)
                if not os.path.isdir(subdir_path):
                    continue
                
                # Get unified class name
                if (key, original_subdir) not in self.class_mapping:
                    continue
                
                target_cls = self.class_mapping[(key, original_subdir)]
                target_idx = self.class_to_idx[target_cls]
                
                # Walk directory tree and collect image files
                for root, _, files in os.walk(subdir_path):
                    for file in files:
                        if file.lower().endswith(valid_extensions):
                            file_path = os.path.join(root, file)
                            self.samples.append((file_path, target_idx))
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (image tensor, class index)
        """
        path, target = self.samples[idx]
        
        try:
            sample = Image.open(path).convert('RGB')
        except Exception as e:
            # Fallback: load next sample if current image is corrupt
            print(f"Warning: Failed to load {path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, target


def train(args: argparse.Namespace) -> None:
    """
    Train ResNet9 on combined animal disease dataset.
    
    Args:
        args: Parsed command-line arguments containing:
            - out_dir: Directory to save model checkpoints
            - name: Base name for saved model files
            - epochs: Number of training epochs
            - batch_size: Batch size for training
            - lr: Learning rate
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Dataset loading
    print("Loading combined animal disease dataset...")
    dataset = CombinedAnimalDataset(
        cattle_dir=args.cattle_dir,
        dog_dir=args.dog_dir,
        goat_dir=args.goat_dir,
        transform=transform,
        max_samples_per_class=args.max_samples_per_class
    )
    num_classes = len(dataset.classes)
    
    # DataLoader (num_workers=0 for Windows compatibility)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    # Model, loss, optimizer
    model = ResNet9(in_channels=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Mixed precision setup (if CUDA available)
    use_amp = device.type == 'cuda'
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed precision training enabled (AMP)")
    
    best_acc = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total = 0
        correct = 0
        total_loss = 0.0
        
        for batch_idx, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            
            # Clear old gradients first
            optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(xb)
                    loss = criterion(preds, yb)
            else:
                preds = model(xb)
                loss = criterion(preds, yb)
            
            # Backward and optimization steps
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Metrics
            _, predicted = torch.max(preds, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
            total_loss += loss.item()
            
            # Progress logging
            if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
                batch_acc = (predicted == yb).sum().item() / yb.size(0)
                print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | Accuracy: {batch_acc:.4f}")
        
        # Epoch statistics
        acc = correct / total
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1:2d}/{args.epochs} | Accuracy: {acc:.4f} | Loss: {avg_loss:.4f}')
        
        # Save best model checkpoint
        if acc > best_acc:
            best_acc = acc
            os.makedirs(args.out_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.out_dir, f'{args.name}_best.pt')
            torch.save({
                'model_state': model.state_dict(),
                'classes': dataset.classes,
                'epoch': epoch,
                'accuracy': acc
            }, checkpoint_path)
            print(f'  ✓ New best model saved (accuracy: {best_acc:.4f})')
    
    # Save final model
    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, f'{args.name}.pt')
    torch.save({
        'model_state': model.state_dict(),
        'classes': dataset.classes,
        'epochs_trained': args.epochs,
        'best_accuracy': best_acc
    }, final_path)
    print(f'✓ Saved {args.name}.pt (best accuracy: {best_acc:.4f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ResNet9 on combined animal disease dataset'
    )
    parser.add_argument(
        '--out-dir',
        default='ai/models/image_classification',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--name',
        default='animal_resnet9',
        help='Base name for saved model files'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--cattle-dir',
        type=str,
        default=None,
        help='Path to cattle diseases dataset directory'
    )
    parser.add_argument(
        '--dog-dir',
        type=str,
        default=None,
        help='Path to dog diseases dataset directory'
    )
    parser.add_argument(
        '--goat-dir',
        type=str,
        default=None,
        help='Path to goat/livestock diseases dataset directory'
    )
    parser.add_argument(
        '--max-samples-per-class',
        type=int,
        default=None,
        help='Maximum number of samples to keep per class'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Number of workers for DataLoader'
    )
    args = parser.parse_args()
    train(args)
