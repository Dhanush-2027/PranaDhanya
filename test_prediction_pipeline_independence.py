import os
import sys
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import hashlib

# Ensure root directory is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from ai.models.resnet9 import ResNet9

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def test_inference_on_checkpoint(checkpoint_path, model_name, sample_images):
    print(f"\n=======================================================", flush=True)
    print(f"   TESTING INFERENCE INDEPENDENCE FOR {model_name.upper()}", flush=True)
    print(f"=======================================================", flush=True)
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file '{checkpoint_path}' does not exist!", flush=True)
        return False
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if not isinstance(checkpoint, dict) or 'classes' not in checkpoint or 'model_state' not in checkpoint:
        print(f"ERROR: Invalid checkpoint format in {checkpoint_path}", flush=True)
        return False
        
    classes = checkpoint['classes']
    model_state = checkpoint['model_state']
    
    print(f"✓ Checkpoint loaded successfully from: {checkpoint_path}")
    print(f"✓ Number of classes: {len(classes)}")
    print(f"✓ Classes: {classes[:5]} ... (showing first 5)")
    
    # Initialize ResNet9 model
    model = ResNet9(in_channels=3, num_classes=len(classes)).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    transform = get_inference_transform()
    
    results = []
    
    print(f"\nProcessing {len(sample_images)} test images...\n", flush=True)
    print(f"{'Image #':<8} | {'SHA256 (first 8)':<16} | {'Tensor Mean':<12} | {'Top Predicted Class':<35} | {'Confidence':<10}")
    print("-" * 95)
    
    for idx, (img_label, img) in enumerate(sample_images, 1):
        # Convert image to bytes to simulate upload
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        
        # Preprocessing
        tensor = transform(img).unsqueeze(0).to(device)
        tensor_mean = float(tensor.mean().item())
        tensor_min = float(tensor.min().item())
        tensor_max = float(tensor.max().item())
        
        # Inference
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        top_idx = int(np.argmax(probs))
        predicted_class = classes[top_idx]
        confidence = float(probs[top_idx] * 100.0)
        
        results.append({
            'label': img_label,
            'hash': img_hash,
            'tensor_mean': tensor_mean,
            'tensor_min': tensor_min,
            'tensor_max': tensor_max,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probs': probs
        })
        
        print(f"{idx:<8} | {img_hash[:16]:<16} | {tensor_mean:<12.4f} | {predicted_class:<35} | {confidence:<9.2f}%")
        
    # Validation checks
    print("\n--- INFERENCE AUDIT RESULTS ---", flush=True)
    
    unique_hashes = len(set(r['hash'] for r in results))
    unique_means = len(set(round(r['tensor_mean'], 5) for r in results))
    unique_predictions = len(set(r['predicted_class'] for r in results))
    confidences = [r['confidence'] for r in results]
    
    print(f"Total Images Tested:      {len(sample_images)}")
    print(f"Unique SHA256 Hashes:     {unique_hashes} / {len(sample_images)}")
    print(f"Unique Preprocessed Means:{unique_means} / {len(sample_images)}")
    print(f"Unique Predicted Classes: {unique_predictions}")
    print(f"Confidence Range:         {min(confidences):.2f}% to {max(confidences):.2f}%")
    
    if unique_hashes == len(sample_images) and unique_means == len(sample_images):
        print("✓ SUCCESS: Preprocessing and tensor computation are 100% INDEPENDENT for every image.")
    else:
        print("✗ WARNING: Duplicate preprocessed tensors detected!")
        
    if unique_predictions > 1:
        print("✓ SUCCESS: Model outputs VARY across different images (no static/hardcoded prediction).")
        return True
    else:
        print("✗ WARNING: All images received the exact same prediction!")
        return False

def generate_diverse_test_images(count=10):
    images = []
    np.random.seed(100)
    
    # 1. Real plant images if available
    plant_data_dir = os.path.join(ROOT_DIR, 'datasets', 'plant_disease', 'data')
    if os.path.exists(plant_data_dir):
        classes = [d for d in os.listdir(plant_data_dir) if os.path.isdir(os.path.join(plant_data_dir, d))]
        for cls in classes[:count]:
            cls_dir = os.path.join(plant_data_dir, cls)
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                try:
                    img = Image.open(files[0]).convert('RGB')
                    images.append((cls, img))
                except Exception:
                    pass
                    
    # 2. Add distinct synthetic test images if needed to reach count
    colors = [
        (34, 139, 34),    # Forest Green
        (139, 69, 19),    # Saddle Brown
        (255, 215, 0),    # Gold
        (220, 20, 60),    # Crimson
        (0, 128, 128),    # Teal
        (128, 0, 128),    # Purple
        (255, 140, 0),    # Dark Orange
        (70, 130, 180),   # Steel Blue
        (107, 142, 35),   # Olive Drab
        (178, 34, 34),    # Firebrick
    ]
    
    idx = len(images)
    while len(images) < count:
        color = colors[idx % len(colors)]
        img = Image.new('RGB', (300, 300), color=color)
        draw = ImageDraw.Draw(img)
        # Add random shapes to ensure distinct features
        draw.ellipse([50 + idx * 5, 50, 200, 200], fill=(255 - color[0], 255 - color[1], 128))
        draw.rectangle([10, 10, 80, 80], fill=(200, 50, idx * 20 % 255))
        images.append((f"synthetic_image_{idx+1}", img))
        idx += 1
        
    return images

if __name__ == "__main__":
    test_images = generate_diverse_test_images(20)
    
    plant_ckpt = os.path.join(ROOT_DIR, 'ai', 'models', 'image_classification', 'plant_resnet9_best.pt')
    animal_ckpt = os.path.join(ROOT_DIR, 'ai', 'models', 'image_classification', 'animal_resnet9_best.pt')
    
    print("Testing Plant Disease Checkpoint...")
    test_inference_on_checkpoint(plant_ckpt, "Plant Disease ResNet9", test_images)
    
    print("\nTesting Animal Disease Checkpoint...")
    test_inference_on_checkpoint(animal_ckpt, "Animal Disease ResNet9", test_images)
