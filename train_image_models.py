import os
import json
import hashlib
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed

from ai.models.resnet9 import ResNet9
from ai_training_utils import scan_image_dataset, _verify_and_hash_image

# Ensure classification output directory exists
os.makedirs("ai/models/image_classification", exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ----------------------------------------------------
# CUSTOM DATASET FOR LIVESTOCK DISEASE (MERGED)
# ----------------------------------------------------
class MergedLivestockDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except Exception:
            img = Image.new('RGB', (64, 64), color=0)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_livestock_data():
    """
    Scans the cattle, dog, and goat directories, merges them,
    removes duplicates, removes corrupted images, and prepares splits.
    """
    cattle_path = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\cattle_diseases"
    dog_path = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\dog_skin_disease"
    goat_path = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\livestock"
    
    class_mapping = {
        'foot-and-mouth': 'cattle_foot_and_mouth',
        'healthy': 'cattle_healthy',
        'lumpy': 'cattle_lumpy',
        'demodicosis': 'dog_demodicosis',
        'Dermatitis': 'dog_dermatitis',
        'Fungal_infections': 'dog_fungal_infections',
        'Healthy': 'dog_healthy',
        'Hypersensitivity': 'dog_hypersensitivity',
        'ringworm': 'dog_ringworm',
        'healthy_goat': 'goat_healthy',
        'unhealthy_goat': 'goat_unhealthy'
    }
    
    classes_list = sorted(list(set(class_mapping.values())))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes_list)}
    
    raw_tasks = []
    file_to_mapped_cls = {}
    
    # 1. Cattle
    cattle_sub = os.path.join(cattle_path, "Cows datasets")
    if os.path.exists(cattle_sub):
        for sub in os.listdir(cattle_sub):
            mapped_cls = class_mapping.get(sub)
            if mapped_cls:
                dir_path = os.path.join(cattle_sub, sub)
                for f in os.listdir(dir_path):
                    f_path = os.path.join(dir_path, f)
                    if os.path.isfile(f_path):
                        raw_tasks.append(f_path)
                        file_to_mapped_cls[f_path] = mapped_cls
                            
    # 2. Dog
    if os.path.exists(dog_path):
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(dog_path, split)
            if os.path.exists(split_dir):
                for sub in os.listdir(split_dir):
                    mapped_cls = class_mapping.get(sub)
                    if mapped_cls:
                        dir_path = os.path.join(split_dir, sub)
                        for f in os.listdir(dir_path):
                            f_path = os.path.join(dir_path, f)
                            if os.path.isfile(f_path):
                                raw_tasks.append(f_path)
                                file_to_mapped_cls[f_path] = mapped_cls
                                    
    # 3. Goat
    if os.path.exists(goat_path):
        for sub in os.listdir(goat_path):
            mapped_cls = class_mapping.get(sub)
            if mapped_cls:
                dir_path = os.path.join(goat_path, sub)
                for f in os.listdir(dir_path):
                    f_path = os.path.join(dir_path, f)
                    if os.path.isfile(f_path):
                        raw_tasks.append(f_path)
                        file_to_mapped_cls[f_path] = mapped_cls
                            
    print(f"Submitting {len(raw_tasks)} livestock images to thread pool for scanning...", flush=True)
    
    img_paths = []
    labels = []
    seen_hashes = set()
    corrupted_count = 0
    duplicate_count = 0
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_verify_and_hash_image, path): path for path in raw_tasks}
        for future in as_completed(futures):
            path, f_hash, is_corrupted = future.result()
            mapped_cls = file_to_mapped_cls[path]
            
            if is_corrupted:
                corrupted_count += 1
            else:
                if f_hash in seen_hashes:
                    duplicate_count += 1
                else:
                    seen_hashes.add(f_hash)
                    img_paths.append(path)
                    labels.append(class_to_idx[mapped_cls])
                    
    print(f"Livestock Dataset Merged: {len(img_paths)} samples, {corrupted_count} corrupted skipped, {duplicate_count} duplicates skipped.", flush=True)
    
    # Save dataset report
    samples_per_class = {}
    for cls in classes_list:
        cls_idx = class_to_idx[cls]
        samples_per_class[cls] = labels.count(cls_idx)
        
    min_class = min(samples_per_class, key=samples_per_class.get)
    max_class = max(samples_per_class, key=samples_per_class.get)
    min_count = samples_per_class[min_class]
    max_count = samples_per_class[max_class]
    
    report = {
        "classes": classes_list,
        "samples_per_class": samples_per_class,
        "total_samples": len(img_paths),
        "missing_values": 0,
        "duplicate_records": duplicate_count,
        "corrupted_records": corrupted_count,
        "class_imbalance_statistics": {
            "min_class": min_class,
            "min_count": min_count,
            "max_class": max_class,
            "max_count": max_count,
            "ratio_min_max": float(min_count / max_count) if max_count > 0 else 0.0
        }
    }
    with open("ai/models/image_classification/livestock_dataset_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Apply class undersampling: max 100 images per class for livestock to speed up training
    np.random.seed(42)
    final_img_paths = []
    final_labels = []
    for cls in classes_list:
        cls_idx = class_to_idx[cls]
        cls_paths = [img_paths[i] for i, l in enumerate(labels) if l == cls_idx]
        if len(cls_paths) > 200:
            cls_paths = list(np.random.choice(cls_paths, 200, replace=False))
        for p in cls_paths:
            final_img_paths.append(p)
            final_labels.append(cls_idx)
            
    print(f"Livestock Dataset Balanced: {len(final_img_paths)} samples selected for training.", flush=True)
    return final_img_paths, final_labels, classes_list

# ----------------------------------------------------
# CUSTOM DATASET FOR PLANT DISEASE
# ----------------------------------------------------
class PlantDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except Exception:
            img = Image.new('RGB', (64, 64), color=0)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_plant_data():
    plant_path = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\plant_disease\data"
    
    # Pre-training report scan (full dataset scan)
    report = scan_image_dataset(plant_path)
    with open("ai/models/image_classification/dataset_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    classes_list = report["classes"]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes_list)}
    
    img_paths = []
    labels = []
    
    # Apply class undersampling: max 100 images per class for fast training
    np.random.seed(42)
    for cls in classes_list:
        cls_dir = os.path.join(plant_path, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        
        if len(files) > 100:
            files = list(np.random.choice(files, 100, replace=False))
            
        for f_path in files:
            img_paths.append(f_path)
            labels.append(class_to_idx[cls])
            
    print(f"Plant Dataset Balanced: {len(img_paths)} samples selected for training.", flush=True)
    return img_paths, labels, classes_list

# ----------------------------------------------------
# TRAINING LOOP WITH EARLY STOPPING & SCHEDULER
# ----------------------------------------------------
def train_resnet9_model(model_name, train_loader, val_loader, num_classes, num_epochs=12, lr=5e-4, patience=4):
    model = ResNet9(in_channels=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    print(f"Starting training for {model_name}...", flush=True)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        scheduler.step()
        
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}", flush=True)
        
        # Checkpoint and Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}", flush=True)
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, history

# ----------------------------------------------------
# EVALUATION METRICS GENERATION
# ----------------------------------------------------
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    # Class-wise accuracy
    classes_accuracy = {}
    for i in range(len(cm)):
        class_total = cm[i].sum()
        class_correct = cm[i][i]
        classes_accuracy[i] = float(class_correct / class_total) if class_total > 0 else 0.0
        
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "class_wise_accuracy": classes_accuracy
    }

# ----------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------
def main():
    res = 224
    train_transform = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomCrop(res, padding=2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ----------------------------------------------------
    # Train Plant Model
    # ----------------------------------------------------
    print("\n==============================================")
    print("TRAINING PLANT DISEASE DETECTOR")
    print("==============================================")
    plant_paths, plant_labels, plant_classes = get_plant_data()
    
    # Stratified Split (70% train, 15% val, 15% test)
    num_samples = len(plant_paths)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_end = int(0.70 * num_samples)
    val_end = int(0.85 * num_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_paths = [plant_paths[i] for i in train_idx]
    train_labels_split = [plant_labels[i] for i in train_idx]
    
    val_paths = [plant_paths[i] for i in val_idx]
    val_labels_split = [plant_labels[i] for i in val_idx]
    
    test_paths = [plant_paths[i] for i in test_idx]
    test_labels_split = [plant_labels[i] for i in test_idx]
    
    train_dataset = PlantDataset(train_paths, train_labels_split, train_transform)
    val_dataset = PlantDataset(val_paths, val_labels_split, val_transform)
    test_dataset = PlantDataset(test_paths, test_labels_split, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Train Plant Model for 12 epochs
    plant_model, plant_history = train_resnet9_model(
        "Plant Disease Model", train_loader, val_loader, len(plant_classes), num_epochs=12, lr=5e-4
    )
    
    plant_metrics = evaluate_model(plant_model, test_loader)
    print(f"Plant Disease Model Test Accuracy: {plant_metrics['accuracy']:.4f}", flush=True)
    
    # Save Plant Checkpoint
    plant_checkpoint = {
        "classes": plant_classes,
        "model_state": plant_model.state_dict(),
        "history": plant_history,
        "metrics": plant_metrics
    }
    torch.save(plant_checkpoint, "ai/models/image_classification/plant_resnet9_best.pt")
    torch.save(plant_checkpoint, "ai/models/image_classification/plant_resnet9_best.pth")
    torch.save(plant_checkpoint, "ai/models/image_classification/plant_resnet9.pt")
    
    with open("ai/models/image_classification/plant_classes.json", "w") as f:
        json.dump(plant_classes, f)
    with open("ai/models/image_classification/training_history.json", "w") as f:
        json.dump(plant_history, f)
    with open("ai/models/image_classification/plant_metrics.json", "w") as f:
        json.dump(plant_metrics, f)
        
    # ----------------------------------------------------
    # Train Livestock Model
    # ----------------------------------------------------
    print("\n==============================================")
    print("TRAINING LIVESTOCK DISEASE DETECTOR")
    print("==============================================")
    live_paths, live_labels, live_classes = get_livestock_data()
    
    num_samples_l = len(live_paths)
    indices_l = np.arange(num_samples_l)
    np.random.seed(42)
    np.random.shuffle(indices_l)
    
    train_end_l = int(0.70 * num_samples_l)
    val_end_l = int(0.85 * num_samples_l)
    
    train_idx_l = indices_l[:train_end_l]
    val_idx_l = indices_l[train_end_l:val_end_l]
    test_idx_l = indices_l[val_end_l:]
    
    train_paths_l = [live_paths[i] for i in train_idx_l]
    train_labels_l = [live_labels[i] for i in train_idx_l]
    
    val_paths_l = [live_paths[i] for i in val_idx_l]
    val_labels_l = [live_labels[i] for i in val_idx_l]
    
    test_paths_l = [live_paths[i] for i in test_idx_l]
    test_labels_l = [live_labels[i] for i in test_idx_l]
    
    train_dataset_l = MergedLivestockDataset(train_paths_l, train_labels_l, train_transform)
    val_dataset_l = MergedLivestockDataset(val_paths_l, val_labels_l, val_transform)
    test_dataset_l = MergedLivestockDataset(test_paths_l, test_labels_l, val_transform)
    
    train_loader_l = DataLoader(train_dataset_l, batch_size=32, shuffle=True, num_workers=0)
    val_loader_l = DataLoader(val_dataset_l, batch_size=32, shuffle=False, num_workers=0)
    test_loader_l = DataLoader(test_dataset_l, batch_size=32, shuffle=False, num_workers=0)
    
    # Train Livestock Model for 12 epochs
    live_model, live_history = train_resnet9_model(
        "Livestock Disease Model", train_loader_l, val_loader_l, len(live_classes), num_epochs=12, lr=5e-4
    )
    
    live_metrics = evaluate_model(live_model, test_loader_l)
    print(f"Livestock Disease Model Test Accuracy: {live_metrics['accuracy']:.4f}", flush=True)
    
    # Save Livestock Checkpoint
    live_checkpoint = {
        "classes": live_classes,
        "model_state": live_model.state_dict(),
        "history": live_history,
        "metrics": live_metrics
    }
    torch.save(live_checkpoint, "ai/models/image_classification/livestock_resnet9_best.pth")
    torch.save(live_checkpoint, "ai/models/image_classification/livestock_resnet9_best.pt")
    torch.save(live_checkpoint, "ai/models/image_classification/livestock_resnet9.pt")
    torch.save(live_checkpoint, "ai/models/image_classification/animal_resnet9_best.pt")
    torch.save(live_checkpoint, "ai/models/image_classification/animal_resnet9.pt")
    
    with open("ai/models/image_classification/livestock_classes.json", "w") as f:
        json.dump(live_classes, f)
    with open("ai/models/image_classification/animal_classes.json", "w") as f:
        json.dump(live_classes, f)
    with open("ai/models/image_classification/livestock_metrics.json", "w") as f:
        json.dump(live_metrics, f)
    with open("ai/models/image_classification/animal_metrics.json", "w") as f:
        json.dump(live_metrics, f)
        
    print("\nAll image models trained successfully!", flush=True)

if __name__ == "__main__":
    main()
