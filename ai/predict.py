import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from pathlib import Path
from ai.config import IMAGE_HEIGHT, IMAGE_WIDTH

def load_tabular_preprocessing(model_dir):
    """Loads feature columns, label encoders, target classes, and scalers."""
    model_path = Path(model_dir)
    
    # 1. Load feature columns
    feature_columns = []
    feat_path = model_path / "feature_columns.json"
    if feat_path.exists():
        with open(feat_path, 'r', encoding='utf-8') as f:
            feature_columns = json.load(f)
            
    # 2. Load target classes
    label_classes = []
    classes_path = model_path / "label_classes.json"
    if classes_path.exists():
        with open(classes_path, 'r', encoding='utf-8') as f:
            label_classes = json.load(f)
            
    # 3. Load feature encoders
    encoders = {}
    for enc_file in model_path.glob("label_encoder_*.json"):
        col_name = enc_file.name.replace("label_encoder_", "").replace(".json", "")
        with open(enc_file, 'r', encoding='utf-8') as f:
            encoders[col_name] = json.load(f)
            
    # 4. Load scaler
    scaler = None
    scaler_path = model_path / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        
    return feature_columns, label_classes, encoders, scaler

def predict_tabular(model_dir, raw_inputs):
    """
    Runs prediction for tabular datasets.
    Handles matching column names, label encoding of inputs, scaling, and formatting.
    """
    model_path = Path(model_dir)
    
    # Load model
    model = None
    h5_model_path = model_path / "model.h5"
    sm_model_path = model_path / "saved_model"
    
    if h5_model_path.exists():
        model = tf.keras.models.load_model(h5_model_path)
    elif sm_model_path.exists():
        model = tf.keras.models.load_model(sm_model_path)
        
    if model is None:
        raise FileNotFoundError(f"No trained model found in {model_dir}")
        
    # Load preprocessing metadata
    feature_columns, label_classes, encoders, scaler = load_tabular_preprocessing(model_dir)
    
    # Format inputs into DataFrame
    if isinstance(raw_inputs, dict):
        df = pd.DataFrame([raw_inputs])
    elif isinstance(raw_inputs, list):
        df = pd.DataFrame(raw_inputs)
    else:
        df = raw_inputs.copy()
        
    # Standardize column casing and spaces
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_').str.replace('___', '_').str.lower()
    
    # Ensure all required features are present (impute missing with 0 or empty string)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0 if scaler is not None else ""
            
    # Reorder columns to match training feature signature
    df = df[feature_columns]
    
    # Encode categorical inputs
    for col, categories in encoders.items():
        if col in df.columns:
            # Map values, unrecognized become 0
            mapping = {cat: i for i, cat in enumerate(categories)}
            df[col] = df[col].astype(str).map(mapping).fillna(0).astype(int)
            
    # Scale numerical features
    if scaler is not None:
        # Determine numeric columns from scaler's expectations
        df_scaled = scaler.transform(df)
        X_pred = df_scaled
    else:
        X_pred = df.values.astype(float)
        
    # Run prediction
    preds = model.predict(X_pred)
    
    # Check output type (Regression, Binary, Multiclass)
    if len(label_classes) == 0:
        # Regression
        predicted_val = float(preds[0][0])
        # Auto determine units
        unit = "Units"
        if "yield" in str(model_path).lower():
            unit = "Metric Tons"
        elif "price" in str(model_path).lower():
            unit = "INR per Quintal"
        return {
            "predicted_value": predicted_val,
            "unit": unit
        }
    else:
        # Classification
        if preds.shape[-1] == 1:
            # Binary
            prob = float(preds[0][0])
            prob_dist = [1 - prob, prob]
            pred_idx = int(prob > 0.5)
        else:
            # Multiclass
            prob_dist = preds[0].tolist()
            pred_idx = int(np.argmax(prob_dist))
            
        predicted_class = label_classes[pred_idx]
        confidence = float(prob_dist[pred_idx] * 100.0)
        
        # Build probability distribution mapping
        distribution = {label_classes[i]: float(prob_dist[i]) for i in range(len(label_classes))}
        
        # Sorted top predictions
        top_predictions = sorted(
            [{"class": label_classes[i], "probability": float(prob_dist[i])} for i in range(len(label_classes))],
            key=lambda x: x["probability"],
            reverse=True
        )
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probability_distribution": distribution,
            "top_predictions": top_predictions
        }

def predict_image(model_dir, image_path_or_bytes):
    """
    Runs prediction for image datasets (Plant / Animal diseases).
    Supports PyTorch ResNet9 (.pth, .pt) checkpoints and legacy Keras models.
    """
    model_path = Path(model_dir)
    
    # 1. Check for PyTorch checkpoints first
    pt_candidates = [
        model_path if model_path.suffix in ['.pth', '.pt'] else None,
        model_path / "plant_disease_model.pth",
        model_path / "animal_disease_model.pth",
        model_path / "plant_resnet9_best.pt",
        model_path / "animal_resnet9_best.pt",
        model_path / "model.pth",
        model_path / "model.pt",
    ]
    pt_candidates = [p for p in pt_candidates if p is not None and p.exists()]
    
    if len(pt_candidates) > 0:
        import torch
        from ai.models.resnet9 import ResNet9
        
        target_pt = pt_candidates[0]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(target_pt, map_location=device, weights_only=False)
        
        # Load label classes
        label_classes = []
        if isinstance(checkpoint, dict) and 'classes' in checkpoint:
            label_classes = checkpoint['classes']
        else:
            for lbl_name in ['plant_labels.json', 'animal_labels.json', 'label_classes.json', 'labels.json']:
                lbl_file = (model_path if model_path.is_dir() else model_path.parent) / lbl_name
                if lbl_file.exists():
                    with open(lbl_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        label_classes = data.get('classes', data) if isinstance(data, dict) else data
                    break
                    
        model_state = checkpoint.get('model_state', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model = ResNet9(in_channels=3, num_classes=len(label_classes))
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        # Load and preprocess image
        if isinstance(image_path_or_bytes, (str, Path)):
            img = Image.open(image_path_or_bytes).convert("RGB")
        else:
            import io
            img = Image.open(io.BytesIO(image_path_or_bytes)).convert("RGB")
            
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            tensor_batch = transform(img).unsqueeze(0).to(device)
        except Exception:
            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            img_arr = np.array(img, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_arr = (img_arr - mean) / std
            tensor_batch = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
        with torch.no_grad():
            logits = model(tensor_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        pred_idx = int(np.argmax(probs))
        predicted_class = label_classes[pred_idx] if pred_idx < len(label_classes) else str(pred_idx)
        confidence = float(probs[pred_idx] * 100.0)
        
        distribution = {label_classes[i]: float(probs[i]) for i in range(len(label_classes))}
        top_predictions = sorted(
            [{"class": label_classes[i], "probability": float(probs[i])} for i in range(len(label_classes))],
            key=lambda x: x["probability"],
            reverse=True
        )
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probability_distribution": distribution,
            "top_predictions": top_predictions
        }

    # 2. Legacy Keras Fallback
    model = None
    h5_model_path = model_path / "model.h5" if model_path.is_dir() else model_path
    sm_model_path = model_path / "saved_model" if model_path.is_dir() else model_path
    
    if h5_model_path.exists():
        model = tf.keras.models.load_model(h5_model_path)
    elif sm_model_path.exists():
        model = tf.keras.models.load_model(sm_model_path)
        
    if model is None:
        raise FileNotFoundError(f"No trained model found in {model_dir}")
        
    # Load class labels
    classes_path = (model_path if model_path.is_dir() else model_path.parent) / "label_classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"label_classes.json not found in {model_dir}")
    with open(classes_path, 'r', encoding='utf-8') as f:
        label_classes = json.load(f)
        
    # Load and preprocess image
    if isinstance(image_path_or_bytes, (str, Path)):
        img = Image.open(image_path_or_bytes).convert("RGB")
    else:
        import io
        img = Image.open(io.BytesIO(image_path_or_bytes)).convert("RGB")
        
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Run prediction
    preds = model.predict(img_batch)
    
    # Format output
    prob_dist = preds[0].tolist()
    pred_idx = int(np.argmax(prob_dist))
    predicted_class = label_classes[pred_idx]
    confidence = float(prob_dist[pred_idx] * 100.0)
    
    distribution = {label_classes[i]: float(prob_dist[i]) for i in range(len(label_classes))}
    top_predictions = sorted(
        [{"class": label_classes[i], "probability": float(prob_dist[i])} for i in range(len(label_classes))],
        key=lambda x: x["probability"],
        reverse=True
    )
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probability_distribution": distribution,
        "top_predictions": top_predictions
    }

