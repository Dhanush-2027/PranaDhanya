import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Force UTF-8 encoding
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add path to ResNet9
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai.models.resnet9 import ResNet9

def main():
    model_path = 'ai/models/image_classification/plant_resnet9_best.pt'
    data_dir = 'datasets/plant_disease/data'
    
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    classes = checkpoint['classes']
    model_state = checkpoint['model_state']
    
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
    
    # Pick 10 classes
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    selected_dirs = subdirs[:10]
    
    images_to_test = []
    for sdir in selected_dirs:
        sdir_path = os.path.join(data_dir, sdir)
        files = [f for f in os.listdir(sdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            images_to_test.append((os.path.join(sdir_path, files[0]), sdir))
            
    print(f"\n--- Testing with {len(images_to_test)} known validation images ---")
    
    passed_count = 0
    for idx, (img_path, expected_label) in enumerate(images_to_test):
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            
        pred_idx = np.argmax(probs)
        predicted_label = classes[pred_idx]
        confidence = probs[pred_idx] * 100
        
        status = "PASS" if predicted_label == expected_label else "FAIL"
        if status == "PASS":
            passed_count += 1
            
        print(f"\nTest Image {idx+1}: {os.path.basename(img_path)}")
        print(f"Expected label: {expected_label}")
        print(f"Predicted label: {predicted_label}")
        print(f"Confidence: {confidence:.2f}%")
        # Format raw output vector to 4 decimal places for print readability
        raw_vec_str = ", ".join([f"{p:.4f}" for p in probs])
        print(f"Raw output vector: [{raw_vec_str}]")
        print(f"Status: {status}")
        
    accuracy = (passed_count / len(images_to_test)) * 100
    print(f"\nValidation Accuracy: {accuracy:.2f}% ({passed_count}/{len(images_to_test)} passed)")

if __name__ == '__main__':
    main()
