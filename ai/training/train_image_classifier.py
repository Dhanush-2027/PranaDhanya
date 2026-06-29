import argparse
import os
import sys

# Add the parent folder of 'ai' to sys.path to enable module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from ai.models.resnet9 import ResNet9

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224,224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Loading dataset from: {args.data_dir}")
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    num_classes = len(dataset.classes)
    print(f"Found {len(dataset)} images with {num_classes} classes")
    print(f"Classes: {dataset.classes}")
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    model = ResNet9(in_channels=3, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total = 0
        correct = 0
        total_loss = 0
        
        for batch_idx, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(preds, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
            total_loss += loss.item()
            
            if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}")
        
        acc = correct / total
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1:2d}/{args.epochs} | Accuracy: {acc:.4f} | Loss: {avg_loss:.4f}')
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save({'model_state': model.state_dict(), 'classes': dataset.classes}, 
                      os.path.join(args.out_dir, f'{args.name}_best.pt'))
    
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'classes': dataset.classes}, 
              os.path.join(args.out_dir, f'{args.name}.pt'))
    print(f'✓ Saved {args.name}.pt (best accuracy: {best_acc:.4f})')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Image folder root (ImageFolder format)')
    parser.add_argument('--out-dir', default='ai/models/image_classification')
    parser.add_argument('--name', default='resnet9')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
