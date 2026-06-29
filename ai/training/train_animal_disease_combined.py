import argparse
import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Add the parent folder of 'ai' to sys.path to enable module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.models.resnet9 import ResNet9

class CombinedAnimalDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []
        
        # Define the three directories
        dirs = {
            'cattle': r"C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets\cattle_diseases\Cows datasets",
            'dog': r"C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets\dog_skin_disease\train",
            'goat': r"C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets\livestock"
        }
        
        # Define class name mapping to avoid conflicts and be clear
        # Mapping format: (source_dir_key, original_subdir_name) -> target_class_name
        self.class_mapping = {}
        
        # 1. Cattle diseases
        for d in os.listdir(dirs['cattle']):
            if os.path.isdir(os.path.join(dirs['cattle'], d)):
                self.class_mapping[('cattle', d)] = f"cattle_{d.replace('-', '_').lower()}"
                
        # 2. Dog diseases
        for d in os.listdir(dirs['dog']):
            if os.path.isdir(os.path.join(dirs['dog'], d)):
                self.class_mapping[('dog', d)] = f"dog_{d.lower()}"
                
        # 3. Goat diseases
        for d in os.listdir(dirs['goat']):
            if os.path.isdir(os.path.join(dirs['goat'], d)):
                # healthy_goat -> goat_healthy, unhealthy_goat -> goat_unhealthy
                mapped_name = f"goat_healthy" if "healthy_goat" in d.lower() else "goat_unhealthy"
                self.class_mapping[('goat', d)] = mapped_name
        
        # Unique list of target classes
        self.classes = sorted(list(set(self.class_mapping.values())))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Scan and collect all image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        for key, base_path in dirs.items():
            for original_subdir in os.listdir(base_path):
                subdir_path = os.path.join(base_path, original_subdir)
                if not os.path.isdir(subdir_path):
                    continue
                
                target_cls = self.class_mapping[(key, original_subdir)]
                target_idx = self.class_to_idx[target_cls]
                
                for root, _, files in os.walk(subdir_path):
                    for file in files:
                        if file.lower().endswith(valid_extensions):
                            file_path = os.path.join(root, file)
                            self.samples.append((file_path, target_idx))
                            
        print(f"Total classes: {len(self.classes)}")
        for cls_name, idx in self.class_to_idx.items():
            print(f"  Class: {cls_name} -> Index: {idx}")
        print(f"Total samples collected: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            sample = Image.open(path).convert('RGB')
        except Exception as e:
            # Fallback in case of corrupt image, get another one
            return self.__getitem__((idx + 1) % len(self.samples))
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample, target

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CombinedAnimalDataset(transform=transform)
    num_classes = len(dataset.classes)
    
    # On Windows, num_workers=0 is safest
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
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
    parser.add_argument('--out-dir', default='ai/models/image_classification')
    parser.add_argument('--name', default='animal_resnet9')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
