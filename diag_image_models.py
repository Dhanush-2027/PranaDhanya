"""Diagnostic script to identify root causes of prediction mismatches."""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter, defaultdict

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

# Import model classes
from ai.models.resnet9 import ResNet9, build_resnet9_from_state_dict

def load_checkpoint_detailed(checkpoint_path):
    """Load checkpoint and report detailed info."""
    print(f"\n{'='*80}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*80}")
    
    device = 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print(f"Checkpoint type: {type(checkpoint)}")
    print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A (not a dict)'}")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        classes = checkpoint.get('classes')
        model_state = checkpoint.get('model_state_dict') or checkpoint.get('model_state')
        
        # If no model_state, the whole checkpoint might be the state dict
        if model_state is None and 'classes' not in checkpoint:
            model_state = checkpoint
            # Try to load classes from separate file
            classes = load_classes_from_json(checkpoint_path)
        
        print(f"\nClasses found: {classes}")
        print(f"Number of classes: {len(classes) if classes else 'None'}")
        print(f"Model state dict type: {type(model_state)}")
        
        if model_state is not None and isinstance(model_state, dict):
            print(f"Model state dict keys: {list(model_state.keys())[:5]}...")
            
        # Try to load the model
        try:
            if classes is None:
                print("\nWARNING: No classes found, cannot load model properly")
                return None, None, None
                
            if build_resnet9_from_state_dict is not None:
                print("\nUsing build_resnet9_from_state_dict...")
                model = build_resnet9_from_state_dict(model_state, len(classes))
            else:
                print("\nUsing ResNet9 directly...")
                model = ResNet9(in_channels=3, num_classes=len(classes))
                model.load_state_dict(model_state)
            
            # Check missing/unexpected keys
            if isinstance(model_state, dict):
                model_keys = set(model.state_dict().keys())
                checkpoint_keys = set(model_state.keys())
                missing = model_keys - checkpoint_keys
                unexpected = checkpoint_keys - model_keys
                
                if missing:
                    print(f"\nWARNING: MISSING keys: {missing}")
                if unexpected:
                    print(f"\nWARNING: UNEXPECTED keys: {unexpected}")
            
            # Check model mode
            print(f"\nModel training mode before eval(): {model.training}")
            model.eval()
            print(f"Model training mode after eval(): {model.training}")
            
            return model, classes, checkpoint
        except Exception as e:
            print(f"\nERROR loading model: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    else:
        # checkpoint is directly the state dict
        print("\nCheckpoint is a state dict (OrderedDict)")
        classes = load_classes_from_json(checkpoint_path)
        if classes is None:
            print("ERROR: Cannot determine number of classes")
            return None, None, None
        
        try:
            if build_resnet9_from_state_dict is not None:
                model = build_resnet9_from_state_dict(checkpoint, len(classes))
            else:
                model = ResNet9(in_channels=3, num_classes=len(classes))
                model.load_state_dict(checkpoint)
            
            model.eval()
            print(f"Model loaded with {len(classes)} classes")
            return model, classes, checkpoint
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return None, None, None
    
    return None, None, None

def load_classes_from_json(checkpoint_path):
    """Try to load classes from JSON file next to checkpoint."""
    checkpoint_dir = Path(checkpoint_path).parent
    json_candidates = [
        checkpoint_dir / 'class_mapping.json',
        checkpoint_dir / 'plant_labels.json',
        checkpoint_dir / 'plant_classes.json',
        checkpoint_dir / 'plant_class_names.json',
        Path('plant_labels.json'),
        Path('ai/models/image_classification/plant_labels.json'),
    ]
    
    for json_path in json_candidates:
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'class_to_idx' in data:
                        return [k for k, v in sorted(data['class_to_idx'].items(), key=lambda x: x[1])]
                    elif isinstance(data, dict) and 'classes' in data:
                        return data['classes']
                    elif isinstance(data, list):
                        return data
            except:
                pass
    
    return None

def get_training_preprocessing_config():
    """Extract preprocessing config from training scripts."""
    print(f"\n{'='*80}")
    print("CHECKING TRAINING SCRIPTS FOR PREPROCESSING CONFIG")
    print(f"{'='*80}")
    
    configs = {}
    
    # Check train_plant_model.py
    plant_train = Path("train_plant_model.py")
    if plant_train.exists():
        content = plant_train.read_text()
        print(f"\ntrain_plant_model.py preprocessing:")
        if "Resize" in content:
            print("  - Uses Resize")
        if "Normalize" in content:
            print("  - Uses Normalize")
        if "0.485" in content:
            print("  - Uses ImageNet normalization (0.485, 0.456, 0.406)")
        if "transforms.Compose" in content:
            print("  - Uses transforms.Compose")
    
    # Check train_animal_model.py
    animal_train = Path("train_animal_model.py")
    if animal_train.exists():
        content = animal_train.read_text()
        print(f"\ntrain_animal_model.py preprocessing:")
        if "Resize" in content:
            print("  - Uses Resize")
        if "Normalize" in content:
            print("  - Uses Normalize")
        if "0.485" in content:
            print("  - Uses ImageNet normalization (0.485, 0.456, 0.406)")
    
    # Check ai_training_utils.py
    utils = Path("ai_training_utils.py")
    if utils.exists():
        content = utils.read_text()
        print(f"\nai_training_utils.py preprocessing:")
        if "Resize" in content:
            print("  - Uses Resize")
        if "Normalize" in content:
            print("  - Uses Normalize")
    
    return configs

def test_preprocessing_variations(model, classes, dataset_path, num_samples=50):
    """Test different preprocessing configurations."""
    print(f"\n{'='*80}")
    print(f"TESTING PREPROCESSING VARIATIONS")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    # Load dataset
    try:
        dataset = datasets.ImageFolder(dataset_path)
        print(f"Dataset classes: {dataset.classes}")
        print(f"Number of classes: {len(dataset.classes)}")
        print(f"Number of images: {len(dataset)}")
        
        if len(dataset.classes) <= 1:
            print(f"❌ Only {len(dataset.classes)} class found - dataset path may be wrong")
            return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Test configurations
    configs = {
        'none': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]),
        'imagenet': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        '05': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),
    }
    
    # Sample images
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    results = {}
    for config_name, transform in configs.items():
        print(f"\n--- Testing {config_name} normalization ---")
        
        # Create transformed dataset
        transformed_dataset = datasets.ImageFolder(dataset_path, transform=transform)
        loader = DataLoader(transformed_dataset, batch_size=1, shuffle=False)
        
        correct = 0
        total = 0
        predictions = []
        
        model.eval()
        with torch.no_grad():
            for idx, (img, label) in enumerate(loader):
                if idx not in sample_indices:
                    continue
                
                output = model(img)
                pred = torch.argmax(output, dim=1).item()
                
                predictions.append(pred)
                if pred == label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        pred_counter = Counter(predictions)
        
        print(f"  Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
        print(f"  Prediction distribution: {dict(pred_counter.most_common(5))}")
        
        results[config_name] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'correct': correct,
            'total': total
        }
    
    # Find best config
    best_config = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n✅ Best configuration: {best_config[0]} with {best_config[1]['accuracy']*100:.2f}% accuracy")
    
    return results

def main():
    print("="*80)
    print("IMAGE MODEL DIAGNOSTIC TOOL")
    print("="*80)
    
    # 1. Check training preprocessing
    get_training_preprocessing_config()
    
    # 2. Test Animal Model
    print("\n" + "="*80)
    print("ANIMAL MODEL DIAGNOSTICS")
    print("="*80)
    
    animal_checkpoint = "ai/models/image_classification/animal_resnet9_best.pt"
    animal_dataset = "datasets/cattle_diseases/data"
    
    if os.path.exists(animal_checkpoint):
        animal_model, animal_classes, animal_ckpt = load_checkpoint_detailed(animal_checkpoint)
        
        if animal_model is not None:
            # Try to find the dataset
            for path in ["datasets/cattle_diseases/data", "datasets/livestock/data", "datasets/animal_disease/data"]:
                if os.path.exists(path):
                    test_preprocessing_variations(animal_model, animal_classes, path, num_samples=20)
                    break
    else:
        print(f"❌ Animal checkpoint not found: {animal_checkpoint}")
    
    # 3. Test Plant Model
    print("\n" + "="*80)
    print("PLANT MODEL DIAGNOSTICS")
    print("="*80)
    
    plant_checkpoint_candidates = [
        "ai/models/image_classification/plant_disease_resnet9.pth",
        "ai/models/image_classification/best_model.pth",
        "plant_disease_model.pth"
    ]
    
    plant_dataset = "datasets/plant_disease/data"
    
    for plant_checkpoint in plant_checkpoint_candidates:
        if os.path.exists(plant_checkpoint):
            plant_model, plant_classes, plant_ckpt = load_checkpoint_detailed(plant_checkpoint)
            
            if plant_model is not None:
                test_preprocessing_variations(plant_model, plant_classes, plant_dataset, num_samples=20)
            break
    else:
        print(f"❌ Plant checkpoint not found in candidates")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
