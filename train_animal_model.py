"""
train_animal_model.py
=====================
ResNet9 CNN Training Script for Animal Disease Detection
- Architecture: Custom ResNet9 (ai.models.resnet9.ResNet9)
- Dataset: Merged Animal Diseases Dataset (11 classes, 8,487 images across Cattle, Dog, Goat)
- GPU Acceleration, Mixed Precision (AMP), OneCycleLR/CosineAnnealing, Gradient Clipping
- 80/20 Train/Validation Split (100% of data used - NO subsampling)
- Saves: animal_disease_model.pth, animal_labels.json, classification report, and training history
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

# Set optimal threads
num_threads = 4
torch.set_num_threads(num_threads)

# Ensure workspace root is on python path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ai.models.resnet9 import ResNet9


def _load_single_image(args):
    path, label_idx, res = args
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    try:
        with Image.open(path) as raw:
            img = raw.convert('RGB').resize((res, res), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            arr = (arr - mean) / std
            return torch.from_numpy(arr), label_idx, True
    except Exception:
        arr = np.zeros((3, res, res), dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr), label_idx, False


def scan_animal_datasets(
    cattle_path: str = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\cattle_diseases",
    dog_path: str = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\dog_skin_disease",
    goat_path: str = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\livestock"
) -> Tuple[List[Tuple[str, str]], List[str], Dict[str, int]]:
    print(f"\n=======================================================", flush=True)
    print(f" SCANNING ANIMAL DISEASE DATASETS", flush=True)
    print(f" Cattle:    {cattle_path}", flush=True)
    print(f" Dog Skin:  {dog_path}", flush=True)
    print(f" Livestock: {goat_path}", flush=True)
    print(f"=======================================================", flush=True)
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    raw_tasks = []
    
    # 1. Cattle Diseases
    cattle_sub = os.path.join(cattle_path, "Cows datasets")
    target_cattle = cattle_sub if os.path.exists(cattle_sub) else cattle_path
    if os.path.exists(target_cattle):
        for sub in sorted(os.listdir(target_cattle)):
            dir_path = os.path.join(target_cattle, sub)
            if os.path.isdir(dir_path):
                cls_name = f"cattle_{sub.lower().replace('-', '_')}"
                for f in os.listdir(dir_path):
                    f_path = os.path.join(dir_path, f)
                    if os.path.isfile(f_path) and os.path.splitext(f)[1].lower() in valid_extensions:
                        raw_tasks.append((f_path, cls_name))

    # 2. Dog Skin Diseases
    if os.path.exists(dog_path):
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(dog_path, split)
            if os.path.exists(split_dir):
                for sub in sorted(os.listdir(split_dir)):
                    dir_path = os.path.join(split_dir, sub)
                    if os.path.isdir(dir_path):
                        cls_name = f"dog_{sub.lower().replace('-', '_')}"
                        for f in os.listdir(dir_path):
                            f_path = os.path.join(dir_path, f)
                            if os.path.isfile(f_path) and os.path.splitext(f)[1].lower() in valid_extensions:
                                raw_tasks.append((f_path, cls_name))

    # 3. Livestock / Goat
    if os.path.exists(goat_path):
        for sub in sorted(os.listdir(goat_path)):
            dir_path = os.path.join(goat_path, sub)
            if os.path.isdir(dir_path):
                cleaned = sub.lower().replace("healthy_goat", "healthy").replace("unhealthy_goat", "unhealthy")
                cls_name = f"goat_{cleaned}"
                for f in os.listdir(dir_path):
                    f_path = os.path.join(dir_path, f)
                    if os.path.isfile(f_path) and os.path.splitext(f)[1].lower() in valid_extensions:
                        raw_tasks.append((f_path, cls_name))

    classes = sorted(list(set(t[1] for t in raw_tasks)))
    class_counts = {c: 0 for c in classes}
    for _, cls_name in raw_tasks:
        class_counts[cls_name] += 1
        
    print(f"\nDiscovered {len(classes)} Animal Disease Classes:")
    for cls_name in classes:
        print(f"  - {cls_name}: {class_counts[cls_name]:,} images")
    print(f"\nTotal Animal Disease Dataset Size: {len(raw_tasks):,} images (100% of data indexed)")
    print(f"=======================================================\n", flush=True)
    
    return raw_tasks, classes, class_counts


def preload_images_parallel(tasks: List[Tuple[str, int]], res: int = 224, max_workers: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    print(f"Pre-caching {len(tasks):,} images into memory using {max_workers} worker threads...", flush=True)
    t0 = time.time()
    
    items = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_load_single_image, (tasks[i][0], tasks[i][1], res)): i
            for i in range(len(tasks))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            tensor, label, ok = future.result()
            items[idx] = (tensor, label)
            
    tensors_list = [item[0] for item in items]
    labels_list = [item[1] for item in items]
    
    t_stack = torch.stack(tensors_list)
    l_tensor = torch.tensor(labels_list, dtype=torch.long)
    print(f"Successfully cached {len(tensors_list):,} image tensors in {time.time() - t0:.2f}s (Shape: {list(t_stack.shape)}).", flush=True)
    return t_stack, l_tensor


def train_animal_model(
    cattle_path: str = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\cattle_diseases",
    dog_path: str = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\dog_skin_disease",
    goat_path: str = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\livestock",
    output_dir: str = "ai/models/image_classification",
    epochs: int = 12,
    batch_size: int = 64,
    img_size: int = 112,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    device_mode: str = "auto",
    allow_cpu: bool = True,
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
            num_threads = min(4, os.cpu_count() or 4)
            torch.set_num_threads(num_threads)
            print(f"Running on CPU with {num_threads} parallel threads (CUDA GPU not detected on this system).", flush=True)
            device = torch.device("cpu")
            
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
        
    # 2. Dataset Scanning (100% of data)
    raw_tasks, classes, class_counts = scan_animal_datasets(
        cattle_path=cattle_path,
        dog_path=dog_path,
        goat_path=goat_path
    )
    num_classes = len(classes)
    total_images = len(raw_tasks)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # 3. 80/20 Train/Validation Split (100% of data used)
    indexed_tasks = [(p, class_to_idx[c]) for p, c in raw_tasks]
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
    
    # 4. Preload tensors into memory
    train_tensors, train_labels = preload_images_parallel(train_tasks, res=img_size, max_workers=16)
    val_tensors, val_labels = preload_images_parallel(val_tasks, res=img_size, max_workers=16)
    
    num_train = len(train_labels)
    num_val = len(val_labels)
    
    # 5. Model Architecture
    print("\nInitializing ResNet9 CNN Architecture...", flush=True)
    model = ResNet9(in_channels=3, num_classes=num_classes).to(device)
    
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
    
    print(f"\n=======================================================", flush=True)
    print(f" STARTING ANIMAL MODEL TRAINING ({epochs} Epochs)", flush=True)
    print(f"=======================================================", flush=True)
    
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Fast tensor batch iteration
        perm = torch.randperm(num_train)
        num_batches = (num_train + batch_size - 1) // batch_size
        
        for batch_i in range(num_batches):
            idx_start = batch_i * batch_size
            idx_end = min(idx_start + batch_size, num_train)
            batch_idx = perm[idx_start:idx_end]
            
            images = train_tensors[batch_idx]
            targets = train_labels[batch_idx]
            
            # Fast on-the-fly tensor flip augmentation
            if torch.rand(1).item() > 0.5:
                images = torch.flip(images, dims=[3])  # horizontal flip
            
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
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
            
            if (batch_i + 1) % 25 == 0 or (batch_i + 1) == num_batches:
                cur_loss = running_loss / total
                cur_acc = correct / total
                print(f"  [Epoch {epoch:02d}/{epochs:02d} | Batch {batch_i+1:03d}/{num_batches:03d}] Loss: {cur_loss:.4f}, Acc: {cur_acc*100:.1f}%", flush=True)
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation Phase
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        
        val_batches = (num_val + batch_size - 1) // batch_size
        with torch.no_grad():
            for batch_i in range(val_batches):
                idx_start = batch_i * batch_size
                idx_end = min(idx_start + batch_size, num_val)
                images = val_tensors[idx_start:idx_end].to(device)
                targets = val_labels[idx_start:idx_end].to(device)
                
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
            f"LR: {current_lr:.6f}{star}\n",
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
        for batch_i in range(val_batches):
            idx_start = batch_i * batch_size
            idx_end = min(idx_start + batch_size, num_val)
            images = val_tensors[idx_start:idx_end].to(device)
            targets = val_labels[idx_start:idx_end]
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
    print(" ANIMAL DISEASE MODEL - FINAL VALIDATION CLASSIFICATION REPORT")
    print("="*70)
    print(report_str)
    print(f"Overall Validation Accuracy: {final_acc*100:.2f}%")
    print("="*70)
    
    # 10. Save Deliverables
    os.makedirs(output_dir, exist_ok=True)
    
    label_map_payload = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "num_classes": num_classes
    }
    
    # Save label files
    with open("animal_labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map_payload, f, indent=4)
    with open(os.path.join(output_dir, "animal_labels.json"), "w", encoding="utf-8") as f:
        json.dump(label_map_payload, f, indent=4)
    with open(os.path.join(output_dir, "animal_classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=4)
    with open(os.path.join(output_dir, "livestock_classes.json"), "w", encoding="utf-8") as f:
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
    torch.save(best_checkpoint, "animal_disease_model.pth")
    torch.save(best_checkpoint, os.path.join(output_dir, "animal_disease_model.pth"))
    torch.save(best_checkpoint, os.path.join(output_dir, "animal_resnet9_best.pt"))
    torch.save(best_checkpoint, os.path.join(output_dir, "animal_resnet9.pt"))
    torch.save(best_checkpoint, os.path.join(output_dir, "livestock_resnet9_best.pth"))
    torch.save(best_checkpoint, os.path.join(output_dir, "livestock_resnet9_best.pt"))
    torch.save(best_checkpoint, os.path.join(output_dir, "livestock_resnet9.pt"))
    
    # Save final checkpoint
    torch.save(final_checkpoint, "animal_disease_model_final.pth")
    torch.save(final_checkpoint, os.path.join(output_dir, "animal_disease_model_final.pth"))
    
    # Save metrics JSON
    with open(os.path.join(output_dir, "animal_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_checkpoint["metrics"], f, indent=4)
    with open(os.path.join(output_dir, "animal_training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
        
    print(f"\n[OK] Saved Animal Disease model checkpoints:")
    print(f"  - animal_disease_model.pth (Best checkpoint)")
    print(f"  - animal_disease_model_final.pth (Final epoch checkpoint)")
    print(f"  - animal_labels.json (Class mappings)")
    print(f"  - ai/models/image_classification/animal_disease_model.pth")
    
    return best_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet9 on Animal Disease Dataset")
    parser.add_argument("--cattle-path", type=str, default=r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\cattle_diseases")
    parser.add_argument("--dog-path", type=str, default=r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\dog_skin_disease")
    parser.add_argument("--goat-path", type=str, default=r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\livestock")
    parser.add_argument("--output-dir", type=str, default="ai/models/image_classification")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--allow-cpu", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train_animal_model(
        cattle_path=args.cattle_path,
        dog_path=args.dog_path,
        goat_path=args.goat_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        patience=args.patience,
        device_mode=args.device,
        allow_cpu=args.allow_cpu,
        seed=args.seed
    )
