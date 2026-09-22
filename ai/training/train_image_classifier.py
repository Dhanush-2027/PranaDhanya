"""
Image Classifier Training Script for ResNet9
=============================================

Trains a ResNet9 model on ImageFolder-format datasets with:
- Proper optimizer gradient handling (zero_grad -> backward -> step)
- Gradient clipping to prevent exploding gradients
- Mixed precision training support for faster computation
- Model checkpointing (saves best model during training)
"""

import argparse
import os
import sys
from typing import Dict, Tuple

# Add the parent folder of 'ai' to sys.path to enable module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from ai.models.resnet9 import ResNet9


def train(args: argparse.Namespace) -> None:
    """
    Train ResNet9 on ImageFolder dataset.
    
    Args:
        args: Parsed command-line arguments containing:
            - data_dir: Path to ImageFolder-format dataset
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
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Dataset loading
    print(f"Loading dataset from: {args.data_dir}")
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    
    # Subsample if requested
    if args.max_samples_per_class is not None:
        class_samples = {}
        for path, class_idx in dataset.samples:
            if class_idx not in class_samples:
                class_samples[class_idx] = []
            class_samples[class_idx].append((path, class_idx))
        
        new_samples = []
        for class_idx, samples in class_samples.items():
            new_samples.extend(samples[:args.max_samples_per_class])
        
        dataset.samples = new_samples
        dataset.imgs = new_samples
        print(f"Subsampled dataset to {len(dataset)} images (max {args.max_samples_per_class} per class)")

    num_classes = len(dataset.classes)
    print(f"Found {len(dataset)} images with {num_classes} classes")
    print(f"Classes: {dataset.classes}")
    
    # DataLoader
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
        description='Train ResNet9 on ImageFolder-format dataset'
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Path to ImageFolder-format dataset'
    )
    parser.add_argument(
        '--out-dir',
        default='ai/models/image_classification',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--name',
        default='resnet9',
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
        '--num-workers',
        type=int,
        default=0,
        help='Number of workers for DataLoader'
    )
    parser.add_argument(
        '--max-samples-per-class',
        type=int,
        default=None,
        help='Maximum number of samples to use per class (useful for fast training/debugging)'
    )
    args = parser.parse_args()
    train(args)
