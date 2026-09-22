"""
train_plant_model.py
====================
ResNet9 CNN Training Script for Plant Disease Detection
- Architecture: Custom ResNet9 (ai.models.resnet9.ResNet9)
- Dataset: Plant Disease Dataset (71 classes, 116,147 images)
- GPU Acceleration, Mixed Precision (AMP), OneCycleLR/CosineAnnealing, Gradient Clipping
- 80/20 Train/Validation Split (100% of data used - NO subsampling)
- Saves: plant_disease_model.pth, plant_labels.json, classification report, and training history
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Set optimal CPU threads
num_threads = min(4, os.cpu_count() or 4)
torch.set_num_threads(num_threads)

# Ensure workspace root is on python path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ai.models.resnet9 import ResNet9


def _load_single_plant_image(args):
    path, label_idx, res = args
    try:
        with Image.open(path) as raw:
            img = raw.convert('RGB').resize((res, res), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)  # (3, H, W) uint8
            return torch.from_numpy(arr), label_idx, True
    except Exception:
        arr = np.zeros((3, res, res), dtype=np.uint8)
        return torch.from_numpy(arr), label_idx, False


def scan_plant_dataset(dataset_dir: str) -> Tuple[List[str], List[int], List[str], Dict[str, int]]:
    print("\n" + "="*70, flush=True)
    print(" SCANNING PLANT DISEASE DATASET", flush=True)
    print(f" Directory: {dataset_dir}", flush=True)
    print("="*70, flush=True)
    
    p = Path(dataset_dir)
    if not p.exists():
        raise FileNotFoundError(f"Plant dataset directory does not exist: {dataset_dir}")
    
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class subdirectories found in {dataset_dir}")
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    file_paths = []
    labels = []
    class_counts = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    for cls_name in classes:
        cls_dir = p / cls_name
        cls_files = [str(f) for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        class_counts[cls_name] = len(cls_files)
        for f in cls_files:
            file_paths.append(f)
            labels.append(class_to_idx[cls_name])
            
    print(f"\nDiscovered {len(classes)} Plant Disease Classes:")
    for cls_name in classes[:15]:
        print(f"  - {cls_name}: {class_counts[cls_name]:,} images")
    if len(classes) > 15:
        print(f"  ... and {len(classes) - 15} more classes (total {len(classes)} classes)")
    print(f"\nTotal Plant Disease Dataset Size: {len(file_paths):,} images (100% of data indexed)")
    print("="*70 + "\n", flush=True)
    
    return file_paths, labels, classes, class_counts


def preload_plant_images_parallel(tasks: List[Tuple[str, int]], res: int = 84, max_workers: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    print(f"Pre-caching {len(tasks):,} images into RAM using {max_workers} worker threads...", flush=True)
    t0 = time.time()
    
    items = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_load_single_plant_image, (tasks[i][0], tasks[i][1], res)): i
            for i in range(len(tasks))
        }
        for count, future in enumerate(as_completed(future_to_idx), 1):
            idx = future_to_idx[future]
            tensor, label, _ = future.result()
            items[idx] = (tensor, label)
            if count % 20000 == 0 or count == len(tasks):
                print(f"  Loaded {count:,}/{len(tasks):,} images ({count/len(tasks)*100:.1f}%) in {time.time() - t0:.1f}s", flush=True)
            
    tensors_list = [item[0] for item in items]
    labels_list = [item[1] for item in items]
    
    t_stack = torch.stack(tensors_list)  # (N, 3, H, W) uint8
    l_tensor = torch.tensor(labels_list, dtype=torch.long)
    ram_mb = t_stack.element_size() * t_stack.nelement() / (1024 * 1024)
    print(f"Successfully cached {len(tensors_list):,} image tensors ({ram_mb:.1f} MB RAM) in {time.time() - t0:.2f}s (Shape: {list(t_stack.shape)}).", flush=True)
    return t_stack, l_tensor


class FastPlantTensorDataset(Dataset):
    """In-memory Dataset that applies on-the-fly normalization & augmentation."""
    def __init__(self, tensors: torch.Tensor, labels: torch.Tensor, is_train: bool = True):
        self.tensors = tensors  # (N, 3, H, W) uint8
        self.labels = labels
        self.is_train = is_train
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.tensors[idx].float().div(255.0)
        # Fast augmentation for training
        if self.is_train:
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])  # Horizontal flip
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[1])  # Vertical flip
        img = (img - self.mean) / self.std
        return img, self.labels[idx]


def train_plant_model(
    dataset_dir: str = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\plant_disease\data",
    output_dir: str = "ai/models/image_classification",
    epochs: int = 6,
    batch_size: int = 128,
    img_size: int = 84,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    device_mode: str = "auto",
    allow_cpu: bool = True,
    num_workers: int = 0,
    seed: int = 42
):
    # 1. Device detection & validation
    print("=== DEVICE CONFIGURATION ===", flush=True)
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version: {torch.__version__}", flush=True)
    print(f"CUDA Available: {cuda_available}", flush=True)
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name} (Device 0 of {torch.cuda.device_count()})", flush=True)
        device = torch.device("cuda")
    else:
        if device_mode == "cuda":
            raise RuntimeError("FATAL: GPU requested (--device cuda) but torch.cuda.is_available() is False!")
        elif not allow_cpu and device_mode == "auto":
            raise RuntimeError("FATAL: No CUDA GPU detected! To enable CPU execution, pass --allow-cpu or --device cpu.")
        else:
            num_t = min(4, os.cpu_count() or 4)
            torch.set_num_threads(num_t)
            print(f"Running on CPU with {num_t} parallel threads (CUDA GPU not detected on this system).", flush=True)
            device = torch.device("cpu")
            
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
        
    # 2. Dataset Scanning (100% of data)
    file_paths, labels, classes, class_counts = scan_plant_dataset(dataset_dir)
    num_classes = len(classes)
    total_images = len(file_paths)
    
    # 3. 80/20 Train/Validation Split (100% of data used)
    indexed_tasks = list(zip(file_paths, labels))
    np.random.seed(seed)
    np.random.shuffle(indexed_tasks)
    
    val_ratio = 0.20
    split_point = int(total_images * (1 - val_ratio))
    
    train_tasks = indexed_tasks[:split_point]
    val_tasks = indexed_tasks[split_point:]
    
    print(f"Dataset Split Summary:")
    print(f"  - Training Set:   {len(train_tasks):,} images ({(1-val_ratio)*100:.1f}%)")
    print(f"  - Validation Set: {len(val_tasks):,} images ({val_ratio*100:.1f}%)")
    print(f"  - Total Used:     {len(train_tasks) + len(val_tasks):,} / {total_images:,} images (100.0%)\n", flush=True)
    
    # 4. Fast Parallel RAM Pre-caching
    train_tensors, train_labels = preload_plant_images_parallel(train_tasks, res=img_size, max_workers=16)
    val_tensors, val_labels = preload_plant_images_parallel(val_tasks, res=img_size, max_workers=16)
    
    train_dataset = FastPlantTensorDataset(train_tensors, train_labels, is_train=True)
    val_dataset = FastPlantTensorDataset(val_tensors, val_labels, is_train=False)
    
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_mem
    )
    
    # 5. Model Architecture
    print("Initializing ResNet9 CNN Architecture...", flush=True)
    model = ResNet9(in_channels=3, num_classes=num_classes, base_channels=32).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet9 Model initialized with {total_params:,} trainable parameters.", flush=True)
    
    # 6. Loss, Optimizer, Scheduler, AMP Scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # 7. Training Loop with Early Stopping
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": []
    }
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0
    
    print("\n" + "="*70, flush=True)
    print(f" STARTING PLANT MODEL TRAINING ({epochs} Epochs)", flush=True)
    print("="*70, flush=True)
    
    t_start = time.time()
    num_batches = len(train_loader)
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=pin_mem)
            targets = targets.to(device, non_blocking=pin_mem)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            
            if batch_idx % 100 == 0 or batch_idx == num_batches:
                cur_loss = running_loss / total
                cur_acc = correct / total
                print(f"  [Epoch {epoch:02d}/{epochs:02d} | Batch {batch_idx:04d}/{num_batches:04d}] Loss: {cur_loss:.4f}, Acc: {cur_acc*100:.1f}%", flush=True)
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation Phase
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=pin_mem)
                targets = targets.to(device, non_blocking=pin_mem)
                
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    
                val_loss_sum += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_total += targets.size(0)
                val_correct += preds.eq(targets).sum().item()
                
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step()
        epoch_dur = time.time() - epoch_start
        
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            star = " * (Best)"
        else:
            epochs_no_improve += 1
            star = ""
            
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_dur:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.6f}{star}",
            flush=True
        )
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without validation accuracy improvement.", flush=True)
            break

    total_training_time = time.time() - t_start
    print(f"\nTraining completed in {total_training_time/60:.2f} minutes. Best Val Acc: {best_val_acc*100:.2f}% (Epoch {best_epoch}).", flush=True)
    
    # 8. Load Best Weights for Final Evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    
    # 9. Final Validation Evaluation & Classification Report
    print("\nGenerating final validation classification report...", flush=True)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            
    final_acc = accuracy_score(all_targets, all_preds)
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        all_targets,
        all_preds,
        target_names=classes,
        zero_division=0
    )
    
    print("\n" + "="*70)
    print(" PLANT DISEASE MODEL - FINAL VALIDATION CLASSIFICATION REPORT")
    print("="*70)
    print(report_str)
    print(f"Overall Validation Accuracy: {final_acc*100:.2f}%")
    print("="*70)
    
    # 10. Save Deliverables
    os.makedirs(output_dir, exist_ok=True)
    
    # Labels JSON
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    label_map_payload = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "num_classes": num_classes
    }
    
    # Save label files
    with open("plant_labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map_payload, f, indent=4)
    with open(os.path.join(output_dir, "plant_labels.json"), "w", encoding="utf-8") as f:
        json.dump(label_map_payload, f, indent=4)
    with open(os.path.join(output_dir, "plant_classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=4)
        
    # Checkpoints
    best_checkpoint = {
        "model_state": best_model_state if best_model_state is not None else model.state_dict(),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "num_classes": num_classes,
        "best_val_accuracy": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs_trained": len(history["epoch"]),
        "history": history,
        "metrics": {
            "accuracy": float(final_acc),
            "macro_avg_f1": float(report_dict.get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_avg_f1": float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)),
            "classification_report": report_dict
        },
        "training_time_seconds": float(total_training_time)
    }
    
    final_checkpoint = {
        "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "classes": classes,
        "class_to_idx": class_to_idx,
        "num_classes": num_classes,
        "final_epoch": len(history["epoch"]),
        "history": history
    }
    
    # Save best checkpoints
    torch.save(best_checkpoint, "plant_disease_model.pth")
    torch.save(best_checkpoint, os.path.join(output_dir, "plant_disease_model.pth"))
    torch.save(best_checkpoint, os.path.join(output_dir, "plant_resnet9_best.pt"))
    torch.save(best_checkpoint, os.path.join(output_dir, "plant_resnet9_best.pth"))
    torch.save(best_checkpoint, os.path.join(output_dir, "plant_resnet9.pt"))
    
    # Save final checkpoint
    torch.save(final_checkpoint, "plant_disease_model_final.pth")
    torch.save(final_checkpoint, os.path.join(output_dir, "plant_disease_model_final.pth"))
    
    # Save metrics JSON
    with open(os.path.join(output_dir, "plant_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_checkpoint["metrics"], f, indent=4)
    with open(os.path.join(output_dir, "plant_training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
        
    print(f"\n[OK] Saved Plant Disease model checkpoints:")
    print(f"  - plant_disease_model.pth (Best checkpoint)")
    print(f"  - plant_disease_model_final.pth (Final epoch checkpoint)")
    print(f"  - plant_labels.json (Class mappings)")
    print(f"  - ai/models/image_classification/plant_disease_model.pth")
    
    return best_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet9 on Plant Disease Dataset")
    parser.add_argument("--dataset-dir", type=str, default=r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\plant_disease\data")
    parser.add_argument("--output-dir", type=str, default="ai/models/image_classification")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=84)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--allow-cpu", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train_plant_model(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        patience=args.patience,
        device_mode=args.device,
        allow_cpu=args.allow_cpu,
        num_workers=args.num_workers,
        seed=args.seed
    )
