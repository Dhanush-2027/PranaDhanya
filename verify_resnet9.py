import torch
import os
from pathlib import Path

model_dir = Path("ai/models/image_classification")
print("=== Image Classification Models ===")
for pt_file in model_dir.glob("*.pt"):
    print(f"\nFile: {pt_file.name}")
    try:
        checkpoint = torch.load(pt_file, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            print("  - Type: dictionary (checkpoint)")
            print(f"  - Keys: {list(checkpoint.keys())}")
            if "classes" in checkpoint:
                print(f"  - Classes ({len(checkpoint['classes'])}): {checkpoint['classes']}")
            if "best_accuracy" in checkpoint:
                print(f"  - Best Accuracy: {checkpoint['best_accuracy']}")
            if "best_acc" in checkpoint:
                print(f"  - Best Acc: {checkpoint['best_acc']}")
        else:
            print(f"  - Type: {type(checkpoint)}")
    except Exception as e:
        print(f"  - Error loading: {e}")
