import torch
import json
from pathlib import Path

checkpoints = [
    'plant_disease_model.pth',
    'best_model.pth',
    'plant-disease-resnet9.pth',
    'ai/models/image_classification/plant_disease_model.pth',
    'ai/models/image_classification/best_model.pth',
    'ai/models/image_classification/plant-disease-resnet9.pth',
]

for p in checkpoints:
    path = Path(p)
    if path.exists():
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict):
                classes = ckpt.get('classes', [])
                print(f"{p}: DICT with {len(classes)} classes")
            else:
                print(f"{p}: OrderedDict (state dict only)")
        except Exception as e:
            print(f"{p}: ERROR - {e}")
    else:
        print(f"{p}: NOT FOUND")
