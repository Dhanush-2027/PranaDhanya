"""
Full Animal Disease Detection Training Script (ResNet9 CNN)
============================================================

Trains ResNet9 on the complete attached datasets:
1. Cattle diseases: foot-and-mouth, healthy, lumpy (datasets/cattle_diseases)
2. Dog skin diseases: Dermatitis, Fungal_infections, Healthy, Hypersensitivity, demodicosis, ringworm (datasets/dog_skin_disease)
3. Livestock/Goat diseases: healthy_goat, unhealthy_goat (datasets/livestock)

Features:
- Full dataset loading across all subdirectories and splits without data loss
- Corrupted image filtering
- High-speed parallel in-memory pre-caching
- Stratified 80% Train / 10% Val / 10% Test split
- Data augmentation (flips, rotations, jitter)
- Inverse frequency class-weighted CrossEntropyLoss
- AdamW optimizer with Cosine Annealing learning rate schedule
- Comprehensive evaluation on held-out test split
- Saves complete model checkpoints, class mappings, metrics, and disease metadata
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Maximize CPU parallelism
num_threads = min(8, os.cpu_count() or 4)
torch.set_num_threads(num_threads)
print(f"PyTorch using {num_threads} CPU threads for high-speed training.", flush=True)

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ai.models.resnet9 import ResNet9

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {DEVICE}", flush=True)


CATTLE_MAPPING = {
    'foot-and-mouth': 'cattle_foot_and_mouth',
    'healthy': 'cattle_healthy',
    'lumpy': 'cattle_lumpy'
}

DOG_MAPPING = {
    'demodicosis': 'dog_demodicosis',
    'dermatitis': 'dog_dermatitis',
    'fungal_infections': 'dog_fungal_infections',
    'healthy': 'dog_healthy',
    'hypersensitivity': 'dog_hypersensitivity',
    'ringworm': 'dog_ringworm'
}

GOAT_MAPPING = {
    'healthy_goat': 'goat_healthy',
    'unhealthy_goat': 'goat_unhealthy',
    'healthy': 'goat_healthy',
    'unhealthy': 'goat_unhealthy'
}


class FastTensorDataset(Dataset):
    """Fast in-memory tensor dataset with dynamic data augmentation."""
    def __init__(self, tensors: torch.Tensor, labels: torch.Tensor, is_train: bool = False):
        self.tensors = tensors
        self.labels = labels
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.tensors[idx]
        if self.is_train:
            # Dynamic augmentation on tensor
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])  # Horizontal flip
            if torch.rand(1).item() > 0.7:
                img = torch.flip(img, dims=[1])  # Vertical flip
            if torch.rand(1).item() > 0.6:
                # Slight brightness adjustment
                factor = 0.85 + 0.3 * torch.rand(1).item()
                img = torch.clamp(img * factor, -3.0, 3.0)
            if torch.rand(1).item() > 0.7:
                # Random 90 deg rotation
                k = int(torch.randint(1, 4, (1,)).item())
                img = torch.rot90(img, k, [1, 2])
        return img, self.labels[idx]


def _process_single_image(args: Tuple[str, int, int]):
    path, label_idx, res = args
    try:
        with Image.open(path) as raw:
            img = raw.convert('RGB').resize((res, res), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            arr = (arr - mean) / std
            return torch.from_numpy(arr), label_idx, path, True
    except Exception:
        return None, label_idx, path, False


def scan_and_collect_dataset(
    cattle_dir: str,
    dog_dir: str,
    goat_dir: str
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Scan all three datasets and map folder names to unified class names."""
    raw_tasks = []
    
    # 1. Cattle
    cows_sub = os.path.join(cattle_dir, "Cows datasets") if os.path.exists(os.path.join(cattle_dir, "Cows datasets")) else cattle_dir
    if os.path.exists(cows_sub):
        for sub in os.listdir(cows_sub):
            mapped = CATTLE_MAPPING.get(sub.lower())
            if not mapped:
                mapped = f"cattle_{sub.lower().replace('-', '_')}"
            sub_path = os.path.join(cows_sub, sub)
            if os.path.isdir(sub_path):
                for root, _, files in os.walk(sub_path):
                    for f in files:
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                            raw_tasks.append((os.path.join(root, f), mapped))
                            
    # 2. Dog (scan train, valid, test or any subdirs)
    if os.path.exists(dog_dir):
        for root, _, files in os.walk(dog_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    parent = os.path.basename(root).lower()
                    mapped_name = DOG_MAPPING.get(parent)
                    if not mapped_name:
                        mapped_name = f"dog_{parent.replace('-', '_')}"
                    raw_tasks.append((os.path.join(root, f), mapped_name))

    # 3. Livestock / Goat
    if os.path.exists(goat_dir):
        for root, _, files in os.walk(goat_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    parent = os.path.basename(root).lower()
                    mapped_name = GOAT_MAPPING.get(parent)
                    if not mapped_name:
                        if 'unhealthy' in parent:
                            mapped_name = 'goat_unhealthy'
                        elif 'healthy' in parent:
                            mapped_name = 'goat_healthy'
                        else:
                            mapped_name = f"goat_{parent}"
                    raw_tasks.append((os.path.join(root, f), mapped_name))

    classes = sorted(list(set(mapped for _, mapped in raw_tasks)))
    print(f"\nDiscovered {len(classes)} Unified Classes:")
    for idx, c in enumerate(classes):
        cnt = sum(1 for _, m in raw_tasks if m == c)
        print(f"  [{idx:2d}] {c:<25} : {cnt} images")
    print(f"Total raw images collected: {len(raw_tasks)}", flush=True)
    return raw_tasks, classes


def preload_images_to_tensors(tasks: List[Tuple[str, str]], classes: List[str], res: int = 224):
    """Pre-cache images in parallel into CPU memory tensors."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    prep_items = [(path, class_to_idx[cls_name], res) for path, cls_name in tasks]
    
    print(f"\nLoading and pre-caching {len(prep_items)} images into memory using 16 threads...", flush=True)
    t0 = time.time()
    tensors = []
    labels = []
    valid_paths = []
    corrupt_count = 0
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(_process_single_image, item) for item in prep_items]
        for f in as_completed(futures):
            tensor, label, path, ok = f.result()
            if ok and tensor is not None:
                tensors.append(tensor)
                labels.append(label)
                valid_paths.append(path)
            else:
                corrupt_count += 1
                
    tensor_stack = torch.stack(tensors)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    t_elapsed = time.time() - t0
    print(f"Successfully cached {len(tensors)} valid images in {t_elapsed:.2f}s ({corrupt_count} corrupted skipped).", flush=True)
    print(f"Tensor Shape: {list(tensor_stack.shape)} | Label Shape: {list(label_tensor.shape)}", flush=True)
    return tensor_stack, label_tensor, valid_paths


def evaluate_test_split(model: nn.Module, loader: DataLoader, classes: List[str]) -> Dict:
    """Evaluate model on test dataset and return detailed performance metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yb.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    # Class-wise metrics
    class_metrics = {}
    for idx, c_name in enumerate(classes):
        total_c = np.sum(np.array(all_targets) == idx)
        correct_c = np.sum((np.array(all_targets) == idx) & (np.array(all_preds) == idx))
        c_acc = float(correct_c / total_c) if total_c > 0 else 0.0
        class_metrics[c_name] = {
            "samples": int(total_c),
            "correct": int(correct_c),
            "accuracy": c_acc
        }
        
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "total_test_samples": len(all_targets),
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist()
    }


def train_animal_resnet9(
    cattle_dir: str = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\cattle_diseases",
    dog_dir: str = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\dog_skin_disease",
    goat_dir: str = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\livestock",
    out_dir: str = "ai/models/image_classification",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3
):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Discover all classes and samples
    raw_tasks, classes = scan_and_collect_dataset(cattle_dir, dog_dir, goat_dir)
    num_classes = len(classes)
    
    # 2. Pre-cache into tensors
    tensors, labels, _ = preload_images_to_tensors(raw_tasks, classes, res=224)
    
    # 3. Stratified 80% train / 10% val / 10% test split
    indices = np.arange(len(labels))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, labels.numpy(), test_size=0.20, stratify=labels.numpy(), random_state=42
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print(f"\nData Splits: Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")
    
    # Compute inverse frequency class weights
    class_counts = np.bincount(y_train, minlength=num_classes)
    total_train = len(y_train)
    weights = total_train / (num_classes * np.maximum(class_counts, 1).astype(np.float32))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"Class Weights configured for balanced training: {np.round(weights, 2).tolist()}")
    
    # Datasets and Loaders
    train_ds = FastTensorDataset(tensors[train_idx], labels[train_idx], is_train=True)
    val_ds = FastTensorDataset(tensors[val_idx], labels[val_idx], is_train=False)
    test_ds = FastTensorDataset(tensors[test_idx], labels[test_idx], is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Model, Optimizer, Loss, Scheduler
    model = ResNet9(in_channels=3, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print("\n" + "=" * 80)
    print("STARTING RESNET9 ANIMAL DISEASE TRAINING")
    print("=" * 80)
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                v_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                v_correct += (preds == yb).sum().item()
                v_total += yb.size(0)
                
        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        ep_time = time.time() - ep_t0
        print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%", flush=True)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  --> [*] New best model saved (Val Acc: {val_acc*100:.2f}%)")
            
    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.1f}s. Best Epoch: {best_epoch} with Val Acc: {best_val_acc*100:.2f}%")
    
    # Load best model for evaluation and saving
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Evaluate on held-out test split
    print("\nEvaluating best model on held-out test split...")
    test_metrics = evaluate_test_split(model, test_loader, classes)
    
    print("\n" + "=" * 80)
    print(f"HELD-OUT TEST RESULTS (Samples: {test_metrics['total_test_samples']}):")
    print(f"  Overall Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"  Weighted Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  Weighted Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"  Weighted F1-Score:  {test_metrics['f1_score']*100:.2f}%")
    print("-" * 80)
    print("Class-wise Accuracy:")
    for c_name, m in test_metrics["class_metrics"].items():
        print(f"  {c_name:<25} : {m['correct']}/{m['samples']} ({m['accuracy']*100:.2f}%)")
    print("=" * 80)
    
    # Save checkpoints
    checkpoint_payload = {
        "classes": classes,
        "model_state": best_model_state or model.state_dict(),
        "epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
        "test_metrics": test_metrics,
        "history": history
    }
    
    animal_best_pt = os.path.join(out_dir, "animal_resnet9_best.pt")
    animal_final_pt = os.path.join(out_dir, "animal_resnet9.pt")
    livestock_best_pt = os.path.join(out_dir, "livestock_resnet9_best.pt")
    livestock_final_pt = os.path.join(out_dir, "livestock_resnet9.pt")
    
    torch.save(checkpoint_payload, animal_best_pt)
    torch.save(checkpoint_payload, animal_final_pt)
    torch.save(checkpoint_payload, livestock_best_pt)
    torch.save(checkpoint_payload, livestock_final_pt)
    print(f"\nSaved model checkpoints to:\n  - {animal_best_pt}\n  - {livestock_best_pt}")
    
    # Save class names & metrics JSON
    with open(os.path.join(out_dir, "animal_classes.json"), "w") as f:
        json.dump(classes, f, indent=4)
    with open(os.path.join(out_dir, "livestock_classes.json"), "w") as f:
        json.dump(classes, f, indent=4)
    with open(os.path.join(out_dir, "animal_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    with open(os.path.join(out_dir, "livestock_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    try:
        from ai_service.app import ANIMAL_DISEASE_METADATA
        with open(os.path.join(out_dir, "animal_disease_metadata.json"), "w") as f:
            json.dump(ANIMAL_DISEASE_METADATA, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save metadata json: {e}")
        
    print("Saved metadata & metric files successfully.")
    return test_metrics


if __name__ == '__main__':
    train_animal_resnet9()
