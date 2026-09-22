import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed

# Enable maximum CPU parallelism
num_threads = min(8, os.cpu_count() or 4)
torch.set_num_threads(num_threads)
print(f"PyTorch using {num_threads} CPU threads for high-speed training.", flush=True)

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai.models.resnet9 import ResNet9

os.makedirs("ai/models/image_classification", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# FAST IN-MEMORY TENSOR DATASET
# ----------------------------------------------------
class CachedImageDataset(Dataset):
    def __init__(self, tensors, labels, is_train=False):
        self.tensors = tensors  # FloatTensor of shape (N, 3, 224, 224)
        self.labels = labels    # LongTensor of shape (N,)
        self.is_train = is_train
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        img = self.tensors[idx]
        if self.is_train:
            # Fast in-memory tensor flips
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])  # horizontal flip
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[1])  # vertical flip
        return img, self.labels[idx]

def _load_and_preprocess_single_image(args):
    path, label_idx, res = args
    try:
        with Image.open(path) as raw:
            img = raw.convert('RGB').resize((res, res), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            # ImageNet Normalize
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            arr = (arr - mean) / std
            return torch.from_numpy(arr), label_idx, True
    except Exception:
        return None, label_idx, False

def load_dataset_into_memory(tasks, res=224):
    print(f"Pre-caching {len(tasks)} images into memory using 16 worker threads...", flush=True)
    t0 = time.time()
    tensors_list = []
    labels_list = []
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(_load_and_preprocess_single_image, (p, lbl, res)) for p, lbl in tasks]
        for f in as_completed(futures):
            tensor, label, ok = f.result()
            if ok and tensor is not None:
                tensors_list.append(tensor)
                labels_list.append(label)
                
    t_stack = torch.stack(tensors_list)
    l_tensor = torch.tensor(labels_list, dtype=torch.long)
    print(f"Cached {len(tensors_list)} tensors in {time.time() - t0:.2f}s (Shape: {list(t_stack.shape)}).", flush=True)
    return t_stack, l_tensor

# ----------------------------------------------------
# 1. PREPARE PLANT DISEASE DATA
# ----------------------------------------------------
def get_plant_tasks(samples_per_class=40):
    plant_path = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\plant_disease\data"
    subdirs = sorted([d for d in os.listdir(plant_path) if os.path.isdir(os.path.join(plant_path, d))])
    class_to_idx = {cls: idx for idx, cls in enumerate(subdirs)}
    
    tasks = []
    np.random.seed(42)
    for cls in subdirs:
        cls_dir = os.path.join(plant_path, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) > samples_per_class:
            files = list(np.random.choice(files, samples_per_class, replace=False))
        for f_path in files:
            tasks.append((f_path, class_to_idx[cls]))
            
    print(f"Selected {len(tasks)} balanced images across {len(subdirs)} plant disease classes.", flush=True)
    return tasks, subdirs

# ----------------------------------------------------
# 2. PREPARE LIVESTOCK / ANIMAL DATA
# ----------------------------------------------------
def get_animal_tasks(samples_per_class=120):
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
    # Cattle
    cattle_sub = os.path.join(cattle_path, "Cows datasets")
    if os.path.exists(cattle_sub):
        for sub in os.listdir(cattle_sub):
            mapped_cls = class_mapping.get(sub)
            if mapped_cls:
                dir_path = os.path.join(cattle_sub, sub)
                for f in os.listdir(dir_path):
                    f_path = os.path.join(dir_path, f)
                    if os.path.isfile(f_path):
                        raw_tasks.append((f_path, class_to_idx[mapped_cls]))
    # Dog
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
                                raw_tasks.append((f_path, class_to_idx[mapped_cls]))
    # Goat
    if os.path.exists(goat_path):
        for sub in os.listdir(goat_path):
            mapped_cls = class_mapping.get(sub)
            if mapped_cls:
                dir_path = os.path.join(goat_path, sub)
                for f in os.listdir(dir_path):
                    f_path = os.path.join(dir_path, f)
                    if os.path.isfile(f_path):
                        raw_tasks.append((f_path, class_to_idx[mapped_cls]))
                        
    # Balance
    np.random.seed(42)
    final_tasks = []
    for cls in classes_list:
        c_idx = class_to_idx[cls]
        c_tasks = [t for t in raw_tasks if t[1] == c_idx]
        if len(c_tasks) > samples_per_class:
            c_tasks = list(np.random.choice(c_tasks, samples_per_class, replace=False))
        final_tasks.extend(c_tasks)
        
    print(f"Selected {len(final_tasks)} balanced images across {len(classes_list)} animal disease classes.", flush=True)
    return final_tasks, classes_list

# ----------------------------------------------------
# HIGH-SPEED TRAINING LOOP
# ----------------------------------------------------
def train_model(model_name, train_loader, val_loader, num_classes, num_epochs=10, lr=1e-3):
    model = ResNet9(in_channels=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    best_val_acc = 0.0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"\n--- Starting {model_name} Training ({num_epochs} Epochs) ---", flush=True)
    for epoch in range(num_epochs):
        t_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_total += targets.size(0)
                val_correct += preds.eq(targets).sum().item()
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        scheduler.step()
        
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(val_epoch_loss)
        history["val_acc"].append(val_epoch_acc)
        
        dur = time.time() - t_start
        print(f"Epoch {epoch+1:02d}/{num_epochs} ({dur:.1f}s) | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}% | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc*100:.2f}%", flush=True)
        
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

def evaluate(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }

def main():
    print("=" * 70, flush=True)
    print("HIGH-SPEED RESNET9 MODEL TRAINING PIPELINE", flush=True)
    print("=" * 70, flush=True)
    
    # ----------------------------------------------------
    # 1. PLANT MODEL
    # ----------------------------------------------------
    plant_tasks, plant_classes = get_plant_tasks(samples_per_class=40)
    p_tensors, p_labels = load_dataset_into_memory(plant_tasks, res=224)
    
    n_samples = len(p_labels)
    indices = np.random.RandomState(42).permutation(n_samples)
    train_end = int(0.75 * n_samples)
    val_end = int(0.90 * n_samples)
    
    train_ds = CachedImageDataset(p_tensors[indices[:train_end]], p_labels[indices[:train_end]], is_train=True)
    val_ds = CachedImageDataset(p_tensors[indices[train_end:val_end]], p_labels[indices[train_end:val_end]], is_train=False)
    test_ds = CachedImageDataset(p_tensors[indices[val_end:]], p_labels[indices[val_end:]], is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    plant_model, plant_history = train_model("Plant Disease Model", train_loader, val_loader, len(plant_classes), num_epochs=10, lr=1e-3)
    plant_metrics = evaluate(plant_model, test_loader)
    print(f"\n✓ Plant Model Test Accuracy: {plant_metrics['accuracy']*100:.2f}%, F1: {plant_metrics['f1_score']:.4f}", flush=True)
    
    # Save Plant Checkpoints
    plant_ckpt = {
        "classes": plant_classes,
        "model_state": plant_model.state_dict(),
        "history": plant_history,
        "metrics": plant_metrics
    }
    torch.save(plant_ckpt, "ai/models/image_classification/plant_resnet9_best.pt")
    torch.save(plant_ckpt, "ai/models/image_classification/plant_resnet9_best.pth")
    torch.save(plant_ckpt, "ai/models/image_classification/plant_resnet9.pt")
    with open("ai/models/image_classification/plant_classes.json", "w") as f:
        json.dump(plant_classes, f)
    with open("ai/models/image_classification/plant_metrics.json", "w") as f:
        json.dump(plant_metrics, f)
    with open("ai/models/image_classification/training_history.json", "w") as f:
        json.dump(plant_history, f)
        
    # ----------------------------------------------------
    # 2. ANIMAL / LIVESTOCK MODEL
    # ----------------------------------------------------
    animal_tasks, animal_classes = get_animal_tasks(samples_per_class=120)
    a_tensors, a_labels = load_dataset_into_memory(animal_tasks, res=224)
    
    n_samples_a = len(a_labels)
    indices_a = np.random.RandomState(42).permutation(n_samples_a)
    train_end_a = int(0.75 * n_samples_a)
    val_end_a = int(0.90 * n_samples_a)
    
    train_ds_a = CachedImageDataset(a_tensors[indices_a[:train_end_a]], a_labels[indices_a[:train_end_a]], is_train=True)
    val_ds_a = CachedImageDataset(a_tensors[indices_a[train_end_a:val_end_a]], a_labels[indices_a[train_end_a:val_end_a]], is_train=False)
    test_ds_a = CachedImageDataset(a_tensors[indices_a[val_end_a:]], a_labels[indices_a[val_end_a:]], is_train=False)
    
    train_loader_a = DataLoader(train_ds_a, batch_size=64, shuffle=True)
    val_loader_a = DataLoader(val_ds_a, batch_size=64, shuffle=False)
    test_loader_a = DataLoader(test_ds_a, batch_size=64, shuffle=False)
    
    animal_model, animal_history = train_model("Animal Disease Model", train_loader_a, val_loader_a, len(animal_classes), num_epochs=10, lr=1e-3)
    animal_metrics = evaluate(animal_model, test_loader_a)
    print(f"\n✓ Animal Model Test Accuracy: {animal_metrics['accuracy']*100:.2f}%, F1: {animal_metrics['f1_score']:.4f}", flush=True)
    
    # Save Animal Checkpoints
    animal_ckpt = {
        "classes": animal_classes,
        "model_state": animal_model.state_dict(),
        "history": animal_history,
        "metrics": animal_metrics
    }
    torch.save(animal_ckpt, "ai/models/image_classification/animal_resnet9_best.pt")
    torch.save(animal_ckpt, "ai/models/image_classification/animal_resnet9.pt")
    torch.save(animal_ckpt, "ai/models/image_classification/livestock_resnet9_best.pt")
    torch.save(animal_ckpt, "ai/models/image_classification/livestock_resnet9_best.pth")
    torch.save(animal_ckpt, "ai/models/image_classification/livestock_resnet9.pt")
    with open("ai/models/image_classification/animal_classes.json", "w") as f:
        json.dump(animal_classes, f)
    with open("ai/models/image_classification/livestock_classes.json", "w") as f:
        json.dump(animal_classes, f)
    with open("ai/models/image_classification/animal_metrics.json", "w") as f:
        json.dump(animal_metrics, f)
    with open("ai/models/image_classification/livestock_metrics.json", "w") as f:
        json.dump(animal_metrics, f)
        
    print("\n" + "=" * 70, flush=True)
    print("ALL IMAGE MODELS TRAINED AND SAVED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
