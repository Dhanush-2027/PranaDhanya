import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Force UTF-8 encoding
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai.models.resnet9 import ResNet9

def audit_dataset_loading(data_dir):
    print("\n=== 1. Dataset Loading Audit ===")
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"✗ Dataset directory not found: {data_dir}")
        return None
        
    class_subdirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"Found {len(class_subdirs)} class directories in {data_dir}")
    
    corrupted_images = []
    class_counts = {}
    total_images = 0
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    for subdir in class_subdirs:
        class_name = subdir.name
        img_files = [f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        class_counts[class_name] = len(img_files)
        total_images += len(img_files)
        
        for img_file in img_files[:2]:
            try:
                with Image.open(img_file) as img:
                    img.verify()
                # Try opening and loading actual data
                with Image.open(img_file) as img:
                    img.load()
            except Exception as e:
                corrupted_images.append((str(img_file), str(e)))
                
    print(f"Total images scanned: {total_images}")
    print(f"Corrupted images found: {len(corrupted_images)}")
    if corrupted_images:
        for idx, (path, err) in enumerate(corrupted_images[:5]):
            print(f"  - {path}: {err}")
        if len(corrupted_images) > 5:
            print("  ... and more")
            
    # Class imbalance analysis
    counts = list(class_counts.values())
    if counts:
        min_c = min(counts)
        max_c = max(counts)
        mean_c = np.mean(counts)
        print(f"Class distribution - Min: {min_c}, Max: {max_c}, Mean: {mean_c:.2f}")
        imbalance_ratio = max_c / max(1, min_c)
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 1.5:
            print("  [Warning] High class imbalance detected!")
            
    return {
        "class_counts": class_counts,
        "corrupted_images": corrupted_images,
        "total_images": total_images
    }

def audit_preprocessing_and_inference(model_dir, data_dir):
    print("\n=== 2. Preprocessing & Prediction Pipeline Audit ===")
    
    # Load model checkpoint
    model_path = Path(model_dir) / "plant_resnet9_best.pt"
    if not model_path.exists():
        print(f"✗ Model not found at {model_path}")
        return
        
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        print("✓ Successfully loaded PyTorch model checkpoint.")
        classes = checkpoint.get("classes", [])
        print(f"Checkpoint contains {len(classes)} classes.")
        model_state = checkpoint.get("model_state")
    except Exception as e:
        print(f"✗ Error loading model checkpoint: {e}")
        return
        
    # Build model
    model = ResNet9(in_channels=3, num_classes=len(classes))
    model.load_state_dict(model_state)
    model.eval()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Find 5 known images from different classes
    data_path = Path(data_dir)
    images_to_test = []
    
    class_subdirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    for subdir in class_subdirs[:5]:
        img_files = [f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        if img_files:
            images_to_test.append((img_files[0], subdir.name))
            
    print(f"Running predictions on {len(images_to_test)} sample images:")
    for path, true_class in images_to_test:
        try:
            img = Image.open(path).convert("RGB")
            # Apply preprocessing
            tensor = transform(img)
            tensor_batch = tensor.unsqueeze(0)
            
            # Print tensor information
            print(f"\nImage: {path.name} (True class: {true_class})")
            print(f"  Input tensor shape: {tensor_batch.shape}")
            print(f"  Pixel value range: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
            
            with torch.no_grad():
                logits = model(tensor_batch)
                probs = torch.softmax(logits, dim=1).numpy()[0]
                
            pred_idx = np.argmax(probs)
            pred_class = classes[pred_idx]
            
            print(f"  Predicted class index: {pred_idx}")
            print(f"  Predicted class name: {pred_class}")
            print(f"  Confidence: {probs[pred_idx]*100:.2f}%")
            print(f"  Top 3 probabilities:")
            top3_idx = np.argsort(probs)[::-1][:3]
            for idx in top3_idx:
                print(f"    - {classes[idx]}: {probs[idx]:.4f}")
                
            # Save preprocessed image for visual inspection
            # Unnormalize: img = std * tensor + mean
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            unnorm_tensor = tensor * std + mean
            unnorm_tensor = torch.clamp(unnorm_tensor, 0, 1)
            unnorm_img = transforms.ToPILImage()(unnorm_tensor)
            
            out_inspect_path = Path("artifacts/scratch")
            out_inspect_path.mkdir(parents=True, exist_ok=True)
            unnorm_img.save(out_inspect_path / f"inspect_{path.name}")
            print(f"  Saved preprocessed image to {out_inspect_path / f'inspect_{path.name}'}")
            
        except Exception as e:
            print(f"  ✗ Error testing image: {e}")

def evaluate_metrics(model_dir, data_dir):
    print("\n=== 3. Evaluation Metrics ===")
    
    model_path = Path(model_dir) / "plant_resnet9_best.pt"
    if not model_path.exists():
        print(f"✗ Model not found at {model_path}")
        return
        
    checkpoint = torch.load(model_path, map_location="cpu")
    classes = checkpoint["classes"]
    model_state = checkpoint["model_state"]
    
    model = ResNet9(in_channels=3, num_classes=len(classes))
    model.load_state_dict(model_state)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load dataset
    print(f"Loading full evaluation dataset from {data_dir}...")
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Take a subset of 100 images for evaluation to avoid running too slow
    # We will stratify by selecting 1-2 samples per class
    indices = []
    class_to_indices = {}
    for idx, (_, class_idx) in enumerate(dataset.samples):
        if class_idx not in class_to_indices:
            class_to_indices[class_idx] = []
        class_to_indices[class_idx].append(idx)
        
    for class_idx, idxs in class_to_indices.items():
        indices.extend(idxs[:2]) # 2 per class = 142 samples
        
    print(f"Evaluating model on {len(indices)} samples across {len(classes)} classes...")
    subset_dataset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds)
            
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Weighted metrics on eval subset:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    per_class_acc = {}
    misclassified = []
    
    for i, class_name in enumerate(classes):
        total_samples = cm[i].sum()
        if total_samples > 0:
            correct_samples = cm[i][i]
            class_acc = correct_samples / total_samples
            per_class_acc[class_name] = class_acc
            
            # Find misclassified samples
            for j in range(len(classes)):
                if i != j and cm[i][j] > 0:
                    misclassified.append((class_name, classes[j], int(cm[i][j])))
                    
    print("\nPer-class Accuracy (top 5 and bottom 5):")
    sorted_class_acc = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    print("  Top 5:")
    for name, class_acc in sorted_class_acc[:5]:
        print(f"    - {name}: {class_acc:.2%}")
    print("  Bottom 5:")
    for name, class_acc in sorted_class_acc[-5:]:
        print(f"    - {name}: {class_acc:.2%}")
        
    print(f"\nTotal misclassified pairs: {len(misclassified)}")
    if misclassified:
        print("Sample misclassifications (True -> Predicted: Count):")
        for t_cls, p_cls, count in misclassified[:5]:
            print(f"  - {t_cls} -> {p_cls}: {count}")

def test_tiny_overfitting(data_dir):
    print("\n=== 4. Sanity Check: Overfitting on Tiny Subset ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Select 20 samples from the dataset
    np.random.seed(42)
    indices = np.random.choice(len(dataset), 20, replace=False)
    tiny_subset = torch.utils.data.Subset(dataset, indices)
    
    loader = DataLoader(tiny_subset, batch_size=4, shuffle=True)
    
    # Initialize a new ResNet9 model
    num_classes = len(dataset.classes)
    model = ResNet9(in_channels=3, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Training on a tiny subset of 20 images for 10 epochs to verify learning capability...")
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
            
        epoch_acc = correct / total
        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {total_loss/len(loader):.4f} | Accuracy: {epoch_acc:.4f}")
        
    print(f"Final training accuracy on tiny subset: {epoch_acc:.4f}")
    if epoch_acc > 0.8:
        print("✓ Sanity Check Passed: The model architecture can successfully learn and overfit on a tiny subset.")
    else:
        print("✗ Sanity Check Failed: The model failed to overfit. There may be a fundamental bug in the model definition, activation functions, or gradients.")

if __name__ == "__main__":
    data_dir = "datasets/plant_disease/data"
    model_dir = "ai/models/image_classification"
    
    audit_dataset_loading(data_dir)
    audit_preprocessing_and_inference(model_dir, data_dir)
    evaluate_metrics(model_dir, data_dir)
    test_tiny_overfitting(data_dir)
