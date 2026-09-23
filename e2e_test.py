import os
import sys
import json
import time
import psutil
import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, r2_score, mean_absolute_error, mean_squared_error
import requests
from ai.models.resnet9 import ResNet9, build_resnet9_from_state_dict

class RequestsClient:
    def post(self, path, json=None, files=None):
        if files:
            # files is a dict like {"file": (filename, file_obj, content_type)}
            # requests expects files in a similar structure but we need to pass the actual bytes/file
            # In e2e_test.py: files={"file": (img_path.name, f, "image/jpeg")}
            # requests handles this perfectly!
            return requests.post(f"http://127.0.0.1:8000{path}", files=files)
        return requests.post(f"http://127.0.0.1:8000{path}", json=json)

client = RequestsClient()

def get_process_usage():
    p = psutil.Process(os.getpid())
    return {
        "cpu_percent": p.cpu_percent(interval=0.1),
        "memory_mb": p.memory_info().rss / (1024 * 1024)
    }

def analyze_dataset_stats(df, target_col):
    total_samples = len(df)
    n_classes = df[target_col].nunique() if target_col in df.columns else 0
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    
    # Class distribution
    dist = {}
    if target_col in df.columns:
        dist = df[target_col].value_counts().to_dict()
    
    return {
        "total_samples": total_samples,
        "n_classes": n_classes,
        "missing_values": missing,
        "duplicate_records": duplicates,
        "class_distribution": dist
    }

def evaluate_crop_recommendation():
    print("\n--- Evaluating Crop Recommendation ---")
    model_path = Path("ai/models/crop_recommendation/crop_recommender.pkl")
    dataset_path = Path("datasets/crop_recommendation/Crop_recommendation.csv")
    
    if not model_path.exists() or not dataset_path.exists():
        print("✗ Model or Dataset not found")
        return None
        
    df = pd.read_csv(dataset_path)
    stats = analyze_dataset_stats(df, 'label')
    
    # Preprocessing & Split (Seed 42)
    # The training uses XGBoost directly on the raw numerical features, label encodes target
    X = df.drop(columns=['label'])
    y = df['label']
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Load model
    model = joblib.load(model_path)
    
    # Load model-specific label classes
    import json
    model_classes_path = Path("ai/models/crop_recommendation/label_classes.json")
    if model_classes_path.exists():
        with open(model_classes_path, "r", encoding="utf-8") as f:
            model_classes = json.load(f)
    else:
        model_classes = list(le.classes_)
    
    # Direct Prediction
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    direct_latency = (time.time() - start_time) / len(X_test)
    
    y_test_str = [le.classes_[idx] for idx in y_test]
    y_pred_str = [model_classes[idx] for idx in y_pred]
    acc = accuracy_score(y_test_str, y_pred_str)
    p, r, f1, _ = precision_recall_fscore_support(y_test_str, y_pred_str, average='weighted', zero_division=0)
    
    print(f"Direct Model - Test Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    # Test with API (Website Sim)
    api_matches = 0
    api_latencies = []
    
    # Compare first 50 test samples via API for detailed verification
    compare_samples = min(50, len(X_test))
    for i in range(compare_samples):
        row = X_test.iloc[i]
        true_label = le.classes_[y_test[i]]
        
        payload = {
            "state": "Maharashtra",
            "district": "Pune",
            "season": "Kharif",
            "soilType": "Clayey",
            "nitrogen": float(row['N']),
            "phosphorus": float(row['P']),
            "potassium": float(row['K']),
            "temperature": float(row['temperature']),
            "humidity": float(row['humidity']),
            "rainfall": float(row['rainfall'])
        }
        
        t0 = time.time()
        res = client.post("/api/cropRecommendation", json=payload)
        api_latencies.append(time.time() - t0)
        
        if res.status_code == 200:
            data = res.json()
            rec_crop = data['recommendedCrop']
            
            # Map index to class
            direct_pred_class = model_classes[y_pred[i]]
            
            # Clean names for comparison
            c_api = rec_crop.strip().lower().replace(" ", "").replace("_", "")
            c_direct = direct_pred_class.strip().lower().replace(" ", "").replace("_", "")
            
            if c_api == c_direct:
                api_matches += 1
            else:
                print(f"Mismatch: API='{rec_crop}', Direct='{direct_pred_class}'")
                
    api_match_rate = api_matches / compare_samples if compare_samples > 0 else 0
    print(f"API Alignment (first {compare_samples} samples): {api_match_rate*100:.1f}%")
    
    return {
        "model_name": "Crop Recommendation (XGBoost)",
        "dataset_stats": stats,
        "test_metrics": {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1)
        },
        "latencies": {
            "direct_ms": direct_latency * 1000,
            "api_ms": np.mean(api_latencies) * 1000 if api_latencies else 0
        },
        "api_match_rate": api_match_rate
    }

def evaluate_fertilizer_recommendation():
    print("\n--- Evaluating Fertilizer Recommendation ---")
    model_path = Path("ai/models/fertilizer_recommendation/fertilizer_recommender.pkl")
    dataset_path = Path("datasets/fertilizer_prediction/Fertilizer Prediction.csv")
    
    if not model_path.exists() or not dataset_path.exists():
        print("✗ Model or Dataset not found")
        return None
        
    df = pd.read_csv(dataset_path)
    
    # Strip column names
    df.columns = df.columns.str.strip()
    stats = analyze_dataset_stats(df, 'Fertilizer Name')
    
    # Load model
    model = joblib.load(model_path)
    
    # Get encoders
    encoders_path = Path("ai/models/fertilizer_recommendation/label_encoders.json")
    classes_path = Path("ai/models/fertilizer_recommendation/label_classes.json")
    
    with open(encoders_path, 'r') as f:
        encoder_mappings = json.load(f)
    with open(classes_path, 'r') as f:
        label_classes = json.load(f)
        
    # Preprocess same as training
    X = df.copy()
    X.columns = X.columns.str.replace(' ', '_').str.replace('-', '_').str.replace('___', '_').str.lower()
    
    if 'temparature' in X.columns and 'temperature' not in X.columns:
        X['temperature'] = X['temparature']
        
    feature_columns = ['temperature', 'humidity', 'moisture', 'nitrogen', 'potassium', 'phosphorous', 'soil_type', 'crop_type']
    X = X[feature_columns].copy()
    
    for col in ['soil_type', 'crop_type']:
        classes_cat = encoder_mappings[col]
        # Map values to codes
        X[col] = X[col].apply(lambda val: classes_cat.index(val) if val in classes_cat else -1)
        
    y = df['Fertilizer Name'].copy()
    y_encoded = y.apply(lambda val: label_classes.index(val) if val in label_classes else -1).values
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Direct Prediction
    start_time = time.time()
    y_pred = model.predict(X_test)
    direct_latency = (time.time() - start_time) / len(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Direct Model - Test Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    # Test with API
    api_matches = 0
    api_latencies = []
    
    compare_samples = min(50, len(X_test))
    for i in range(compare_samples):
        row = X_test.iloc[i]
        
        # Decode categorical variables to string for the API request
        soil_type_str = encoder_mappings['soil_type'][int(row['soil_type'])]
        crop_type_str = encoder_mappings['crop_type'][int(row['crop_type'])]
        
        payload = {
            "nitrogen": float(row['nitrogen']),
            "phosphorus": float(row['phosphorous']),
            "potassium": float(row['potassium']),
            "temperature": float(row['temperature']),
            "humidity": float(row['humidity']),
            "moisture": float(row['moisture']),
            "soilType": soil_type_str,
            "cropType": crop_type_str
        }
        
        t0 = time.time()
        res = client.post("/api/fertilizerRecommendation", json=payload)
        api_latencies.append(time.time() - t0)
        
        if res.status_code == 200:
            data = res.json()
            rec_fert = data.get('recommendedFertilizer', '')
            
            # Map index to class
            direct_pred_class = label_classes[int(y_pred[i])]
            
            # Clean names for comparison
            c_api = rec_fert.strip().lower().replace(" ", "").replace("-", "")
            c_direct = direct_pred_class.strip().lower().replace(" ", "").replace("-", "")
            
            if c_api == c_direct:
                api_matches += 1
            else:
                print(f"Mismatch: API='{rec_fert}', Direct='{direct_pred_class}'")
                
    api_match_rate = api_matches / compare_samples if compare_samples > 0 else 0
    print(f"API Alignment (first {compare_samples} samples): {api_match_rate*100:.1f}%")
    
    return {
        "model_name": "Fertilizer Recommendation (XGBoost)",
        "dataset_stats": stats,
        "test_metrics": {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1)
        },
        "latencies": {
            "direct_ms": direct_latency * 1000,
            "api_ms": np.mean(api_latencies) * 1000 if api_latencies else 0
        },
        "api_match_rate": api_match_rate
    }

def evaluate_yield_prediction():
    print("\n--- Evaluating Yield Prediction ---")
    model_path = Path("ai/models/yield_prediction/yield_predictor.pkl")
    dataset_path = Path("datasets/yield_prediction/data.csv")
    
    if not model_path.exists() or not dataset_path.exists():
        print("✗ Model or Dataset not found")
        return None
        
    df = pd.read_csv(dataset_path)
    stats = analyze_dataset_stats(df, 'yield')
    
    # Since dataset is too small (3 rows), we evaluate directly on the whole dataset
    X = df[['area', 'rainfall', 'fertilizer', 'temperature', 'humidity']]
    y = df['yield']
    
    model = joblib.load(model_path)
    
    start_time = time.time()
    y_pred = model.predict(X)
    direct_latency = (time.time() - start_time) / len(X)
    
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Direct Model - MAE: {mae:.4f} | R2: {r2:.4f}")
    
    # Test API
    api_latencies = []
    api_matches = 0
    for i in range(len(df)):
        row = df.iloc[i]
        payload = {
            "area": float(row['area']),
            "rainfall": float(row['rainfall']),
            "fertilizer": float(row['fertilizer']),
            "soil": "Clayey",
            "crop": "Rice",
            "temperature": float(row['temperature']),
            "humidity": float(row['humidity'])
        }
        
        t0 = time.time()
        res = client.post("/api/yieldPrediction", json=payload)
        api_latencies.append(time.time() - t0)
        
        if res.status_code == 200:
            api_yield = res.json()['predictedYield']
            direct_yield = float(y_pred[i])
            if abs(api_yield - direct_yield) < 0.05:
                api_matches += 1
            else:
                print(f"Mismatch: API={api_yield}, Direct={direct_yield}")
                
    api_match_rate = api_matches / len(df)
    print(f"API Alignment: {api_match_rate*100:.1f}%")
    
    return {
        "model_name": "Yield Prediction (Random Forest)",
        "dataset_stats": stats,
        "test_metrics": {
            "mae": float(mae),
            "mse": float(mse),
            "r2_score": float(r2)
        },
        "latencies": {
            "direct_ms": direct_latency * 1000,
            "api_ms": np.mean(api_latencies) * 1000 if api_latencies else 0
        },
        "api_match_rate": api_match_rate
    }

def evaluate_price_prediction():
    print("\n--- Evaluating Price Prediction ---")
    model_path = Path("ai/models/price_prediction/price_predictor.pkl")
    dataset_path = Path("datasets/price_prediction/data.csv")
    
    if not model_path.exists() or not dataset_path.exists():
        print("✗ Model or Dataset not found")
        return None
        
    df = pd.read_csv(dataset_path)
    stats = analyze_dataset_stats(df, 'price')
    
    X = df[['feature1', 'feature2', 'feature3']]
    y = df['price']
    
    model = joblib.load(model_path)
    
    start_time = time.time()
    y_pred = model.predict(X)
    direct_latency = (time.time() - start_time) / len(X)
    
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Direct Model - MAE: {mae:.4f} | R2: {r2:.4f}")
    
    # Test API
    api_latencies = []
    api_matches = 0
    for i in range(len(df)):
        row = df.iloc[i]
        payload = {
            "feature1": float(row['feature1']),
            "feature2": float(row['feature2']),
            "feature3": float(row['feature3'])
        }
        
        t0 = time.time()
        res = client.post("/api/pricePrediction", json=payload)
        api_latencies.append(time.time() - t0)
        
        if res.status_code == 200:
            api_price = res.json()['expectedPrice']
            direct_price = float(y_pred[i])
            if abs(api_price - direct_price) < 0.05:
                api_matches += 1
            else:
                print(f"Mismatch: API={api_price}, Direct={direct_price}")
                
    api_match_rate = api_matches / len(df)
    print(f"API Alignment: {api_match_rate*100:.1f}%")
    
    return {
        "model_name": "Price Prediction (Random Forest)",
        "dataset_stats": stats,
        "test_metrics": {
            "mae": float(mae),
            "mse": float(mse),
            "r2_score": float(r2)
        },
        "latencies": {
            "direct_ms": direct_latency * 1000,
            "api_ms": np.mean(api_latencies) * 1000 if api_latencies else 0
        },
        "api_match_rate": api_match_rate
    }

def evaluate_image_models():
    print("\n--- Evaluating Image Classification Models ---")
    
    # 1. Plant Disease ResNet9
    plant_model_path = Path("ai/models/image_classification/plant_resnet9_best.pt")
    plant_dataset_path = Path("datasets/plant_disease/data")
    
    plant_res = None
    if plant_model_path.exists() and plant_dataset_path.exists():
        try:
            checkpoint = torch.load(plant_model_path, map_location="cpu", weights_only=False)
            classes = checkpoint['classes']
            model_state = checkpoint['model_state']
            
            # Simple test sample creation or finding a leaf image
            images = list(plant_dataset_path.rglob("*.jpg")) + list(plant_dataset_path.rglob("*.png"))
            print(f"Plant Disease dataset has {len(images)} images, {len(classes)} classes")
            
            # Build and load model
            model = build_resnet9_from_state_dict(model_state, len(classes))
            model.eval()
            
            # Run evaluations on up to 20 samples
            test_samples = images[:20]
            api_matches = 0
            api_latencies = []
            direct_latencies = []
            
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            from PIL import Image
            for img_path in test_samples:
                # Direct prediction
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0)
                
                t0 = time.time()
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                direct_latencies.append(time.time() - t0)
                
                pred_idx = np.argmax(probs)
                direct_class = classes[pred_idx]
                
                # API prediction
                t0 = time.time()
                with open(img_path, 'rb') as f:
                    res = client.post("/api/predictPlant", files={"file": (img_path.name, f, "image/jpeg")})
                api_latencies.append(time.time() - t0)
                
                if res.status_code == 200:
                    api_data = res.json()
                    api_class = api_data['diseaseName']
                    
                    # Normalizing names
                    # format_plant_display_name maps Tomato___early_blight -> Tomato Early Blight
                    direct_disp = direct_class.replace("___", " ").replace("_", " ").title()
                    if "healthy" in direct_class.lower():
                        crop_prefix = direct_class.split("___")[0].replace("_", " ").title() if "___" in direct_class else "Crop"
                        direct_disp = f"{crop_prefix} Healthy Leaf"
                    
                    if api_class.strip().lower() == direct_disp.strip().lower():
                        api_matches += 1
                    else:
                        print(f"Mismatch: API='{api_class}', Direct='{direct_disp}'")
            
            plant_res = {
                "model_name": "Plant Disease Detection (ResNet9 CNN)",
                "classes_count": len(classes),
                "total_images": len(images),
                "latencies": {
                    "direct_ms": np.mean(direct_latencies) * 1000 if direct_latencies else 0,
                    "api_ms": np.mean(api_latencies) * 1000 if api_latencies else 0
                },
                "api_match_rate": api_matches / len(test_samples) if test_samples else 0
            }
            print(f"Plant ResNet9: Loaded classes={len(classes)} | API Match Rate={plant_res['api_match_rate']*100:.1f}%")
        except Exception as e:
            print(f"Error evaluating plant ResNet9: {e}")
            
    # 2. Animal Disease ResNet9
    animal_candidates = [
        Path("ai/models/image_classification/animal_disease_resnet9.pth"),
        Path("ai/models/image_classification/animal_disease_model.pth"),
        Path("ai/models/image_classification/animal_resnet9_best.pt")
    ]
    animal_model_path = next((p for p in animal_candidates if p.exists()), None)
    
    animal_res = None
    if animal_model_path and animal_model_path.exists():
        try:
            checkpoint = torch.load(animal_model_path, map_location="cpu", weights_only=False)
            classes = checkpoint.get('classes', [])
            model_state = checkpoint.get('model_state_dict') or checkpoint.get('model_state') or checkpoint
            
            # Walk through animal directories to find images
            cattle_path = Path("datasets/cattle_diseases")
            dog_path = Path("datasets/dog_skin_disease")
            goat_path = Path("datasets/livestock")
            
            images = []
            for path in [cattle_path, dog_path, goat_path]:
                if path.exists():
                    images.extend(list(path.rglob("*.jpg")) + list(path.rglob("*.png")))
            
            print(f"Animal Disease dataset has {len(images)} images, {len(classes)} classes")
            
            model = build_resnet9_from_state_dict(model_state, len(classes))
            model.eval()
            
            test_samples = images[:20]
            api_matches = 0
            api_latencies = []
            direct_latencies = []
            
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            from PIL import Image
            for img_path in test_samples:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0)
                
                t0 = time.time()
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                direct_latencies.append(time.time() - t0)
                
                pred_idx = np.argmax(probs)
                direct_class = classes[pred_idx]
                
                # API prediction
                t0 = time.time()
                with open(img_path, 'rb') as f:
                    res = client.post("/api/predictAnimal", files={"file": (img_path.name, f, "image/jpeg")})
                api_latencies.append(time.time() - t0)
                
                if res.status_code == 200:
                    api_data = res.json()
                    api_class = api_data['diseaseName']
                    
                    direct_disp = direct_class.replace("_", " ").title()
                    # Resolve detailed mapping logic
                    from ai_service.app import ANIMAL_DISEASE_METADATA
                    mapped_meta = ANIMAL_DISEASE_METADATA.get(direct_class, {})
                    direct_disp = mapped_meta.get("diseaseName", direct_disp)
                    
                    if api_class.strip().lower() == direct_disp.strip().lower():
                        api_matches += 1
                    else:
                        print(f"Mismatch: API='{api_class}', Direct='{direct_disp}'")
                        
            animal_res = {
                "model_name": "Animal Disease Detection (ResNet9 CNN)",
                "classes_count": len(classes),
                "total_images": len(images),
                "latencies": {
                    "direct_ms": np.mean(direct_latencies) * 1000 if direct_latencies else 0,
                    "api_ms": np.mean(api_latencies) * 1000 if api_latencies else 0
                },
                "api_match_rate": api_matches / len(test_samples) if test_samples else 0
            }
            print(f"Animal ResNet9: Loaded classes={len(classes)} | API Match Rate={animal_res['api_match_rate']*100:.1f}%")
        except Exception as e:
            print(f"Error evaluating animal ResNet9: {e}")
            
    return plant_res, animal_res

if __name__ == "__main__":
    t_start = time.time()
    usage_init = get_process_usage()
    
    crop_res = evaluate_crop_recommendation()
    fert_res = evaluate_fertilizer_recommendation()
    yield_res = evaluate_yield_prediction()
    price_res = evaluate_price_prediction()
    plant_res, animal_res = evaluate_image_models()
    
    usage_final = get_process_usage()
    total_time = time.time() - t_start
    
    summary = {
        "crop_recommendation": crop_res,
        "fertilizer_recommendation": fert_res,
        "yield_prediction": yield_res,
        "price_prediction": price_res,
        "plant_disease": plant_res,
        "animal_disease": animal_res,
        "performance": {
            "total_execution_time_s": total_time,
            "initial_memory_mb": usage_init["memory_mb"],
            "final_memory_mb": usage_final["memory_mb"],
            "memory_delta_mb": usage_final["memory_mb"] - usage_init["memory_mb"],
            "cpu_percent": usage_final["cpu_percent"]
        }
    }
    
    with open("e2e_summary.json", "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    print("\n=== E2E AUDIT AND EVALUATION COMPLETED SUCCESSFULLY ===")
    print(f"Saved results summary to e2e_summary.json")
