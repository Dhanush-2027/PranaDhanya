import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.models.resnet9 import ResNet9


def run_animal_diagnostics():
    print("=" * 80)
    print("RESNET9 ANIMAL DISEASE SYSTEMATIC DIAGNOSTIC & VERIFICATION SUITE")
    print("=" * 80)

    cattle_dir = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\cattle_diseases"
    dog_dir = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\dog_skin_disease"
    goat_dir = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\livestock"

    ckpt_path = "ai/models/image_classification/animal_resnet9_best.pt"
    classes_path = "ai/models/image_classification/animal_classes.json"
    metrics_path = "ai/models/image_classification/animal_metrics.json"

    # 1. Check Checkpoint & Classes
    print("\n[CHECK 1] CHECKPOINT & CLASS VOCABULARY INTEGRITY")
    print("-" * 70)
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ckpt.get("classes", [])
    model_state = ckpt.get("model_state", {})

    print(f"  1.1 Checkpoint Epoch: {ckpt.get('epoch')}")
    print(f"  1.2 Best Validation Accuracy: {ckpt.get('best_val_accuracy', 0)*100:.2f}%")
    print(f"  1.3 Total Classes in Checkpoint: {len(classes)}")
    print(f"  1.4 Classes List: {classes}")

    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            disk_classes = json.load(f)
        print(f"  1.5 Classes file match: {'PASS (Exact match)' if disk_classes == classes else 'FAIL'}")

    # 2. Architecture & Weights
    print("\n[CHECK 2] RESNET9 MODEL ARCHITECTURE & TENSOR FORWARD PASS")
    print("-" * 70)
    model = ResNet9(in_channels=3, num_classes=len(classes))
    load_res = model.load_state_dict(model_state)
    print(f"  2.1 Model State Loaded: {load_res}")
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    print(f"  2.2 Forward Pass Output Shape: {list(out.shape)} (Expected: [2, {len(classes)}])")
    print(f"  2.3 Forward Pass Status: {'PASS' if out.shape == (2, len(classes)) else 'FAIL'}")

    # 3. Test Held-Out Metrics
    print("\n[CHECK 3] EVALUATION METRICS REPORT")
    print("-" * 70)
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            m = json.load(f)
        print(f"  3.1 Overall Accuracy:  {m.get('accuracy', 0)*100:.2f}%")
        print(f"  3.2 Weighted Precision: {m.get('precision', 0)*100:.2f}%")
        print(f"  3.3 Weighted Recall:    {m.get('recall', 0)*100:.2f}%")
        print(f"  3.4 Weighted F1-Score:  {m.get('f1_score', 0)*100:.2f}%")
        print(f"  3.5 Total Test Samples: {m.get('total_test_samples')}")
        print("\n  Class-wise Performance:")
        for c_name, cm in m.get("class_metrics", {}).items():
            print(f"    - {c_name:<25}: {cm['correct']:3d}/{cm['samples']:3d} ({cm['accuracy']*100:.2f}%)")

    # 4. End-to-End Image Inference Verification
    print("\n[CHECK 4] END-TO-END INFERENCE TEST (SAMPLE REAL IMAGES FROM EACH DATASET)")
    print("-" * 70)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Collect sample test images
    test_samples = []
    
    # Cattle samples
    for c_sub in ["foot and mouth", "healthy", "lumpy"]:
        p = os.path.join(cattle_dir, c_sub)
        if os.path.exists(p):
            imgs = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if imgs:
                test_samples.append((f"Cattle: {c_sub}", os.path.join(p, imgs[0])))

    # Dog samples
    for d_sub in ["Demodicosis", "Dermatitis", "fungal infections", "healthy", "hypersensitivity", "ringworm"]:
        p = os.path.join(dog_dir, d_sub)
        if os.path.exists(p):
            imgs = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if imgs:
                test_samples.append((f"Dog: {d_sub}", os.path.join(p, imgs[0])))

    # Goat samples
    for g_sub in ["healthy", "unhealthy"]:
        p = os.path.join(goat_dir, g_sub)
        if os.path.exists(p):
            imgs = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if imgs:
                test_samples.append((f"Livestock/Goat: {g_sub}", os.path.join(p, imgs[0])))

    print(f"  Testing {len(test_samples)} representative image samples...\n")
    print(f"  {'#':<2} | {'Category/Source':<25} | {'Predicted Disease/Class':<25} | {'Confidence':<10} | {'Top-3 Predictions'}")
    print("  " + "-" * 110)

    with torch.no_grad():
        for idx, (source_tag, img_p) in enumerate(test_samples, 1):
            img = Image.open(img_p).convert("RGB")
            inp = transform(img).unsqueeze(0)
            logits = model(inp)[0]
            probs = torch.softmax(logits, dim=0).numpy()
            pred_idx = int(np.argmax(probs))
            pred_class = classes[pred_idx]
            conf = probs[pred_idx] * 100.0

            top3 = torch.topk(torch.softmax(logits, dim=0), min(3, len(classes)))
            top3_str = ", ".join([f"{classes[i]}:{v.item()*100:.1f}%" for i, v in zip(top3.indices, top3.values)])
            print(f"  {idx:02d} | {source_tag:<25} | {pred_class:<25} | {conf:<9.2f}% | [{top3_str}]")

    print("\n" + "=" * 80)
    print("ANIMAL RESNET9 DIAGNOSTICS & VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_animal_diagnostics()
