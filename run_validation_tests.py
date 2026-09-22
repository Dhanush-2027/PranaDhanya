import os
import json
import joblib
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

from ai.models.resnet9 import ResNet9

def run_tabular_tests():
    print("\n==============================================")
    print("RUNNING TABULAR MODELS VALIDATION TESTS")
    print("==============================================")
    
    # 1. Crop Recommendation
    print("\n--- Crop Recommendation Test (100 samples) ---")
    crop_model = joblib.load("ai/models/crop_recommendation/crop_recommendation_model.pkl")
    crop_data = pd.read_csv(r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\crop_recommendation\Crop_recommendation.csv")
    
    expected_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = crop_data[expected_features]
    y = crop_data["label"]
    
    with open("ai/models/crop_recommendation/label_classes.json", "r") as f:
        classes = json.load(f)
        
    _, X_test, _, y_test = train_test_split(X, y, test_size=100, random_state=42, stratify=y)
    
    preds = crop_model.predict(X_test)
    # If the model is a pipeline containing the label encoder target, or classifier outputting raw int, map properly.
    # Note: our pipeline fitted on y_encoded, so it predicts integer indices.
    # Let's map predicted indices to class names.
    pred_labels = [classes[p] for p in preds]
    
    correct = sum(p == t for p, t in zip(pred_labels, y_test))
    print(f"Tabular Crop Recommendation Accuracy: {correct}/100 ({correct:.1f}%)")
    
    print("\nSample Comparisons (First 5):")
    for i in range(5):
        print(f"Sample {i+1} - Features: {X_test.iloc[i].to_dict()}")
        print(f"  Predicted: {pred_labels[i]} | Expected: {y_test.iloc[i]}")
        
    # 2. Fertilizer Recommendation
    print("\n--- Fertilizer Recommendation Test (100 samples) ---")
    fert_model = joblib.load("ai/models/fertilizer_recommendation/fertilizer_model.pkl")
    fert_data = pd.read_csv(r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\fertilizer_prediction\Fertilizer Prediction.csv")
    fert_data.columns = fert_data.columns.str.strip()
    
    # rename synonyms
    if "Temparature" in fert_data.columns:
        fert_data = fert_data.rename(columns={"Temparature": "temperature"})
    if "Humidity" in fert_data.columns:
        fert_data = fert_data.rename(columns={"Humidity": "humidity"})
    if "Humidity " in fert_data.columns:
        fert_data = fert_data.rename(columns={"Humidity ": "humidity"})
    if "Moisture" in fert_data.columns:
        fert_data = fert_data.rename(columns={"Moisture": "moisture"})
    if "Nitrogen" in fert_data.columns:
        fert_data = fert_data.rename(columns={"Nitrogen": "nitrogen"})
    if "Potassium" in fert_data.columns:
        fert_data = fert_data.rename(columns={"Potassium": "potassium"})
    if "Phosphorous" in fert_data.columns:
        fert_data = fert_data.rename(columns={"Phosphorous": "phosphorous"})
    if "Soil Type" in fert_data.columns:
        fert_data = fert_data.rename(columns={"Soil Type": "soil_type"})
    if "Crop Type" in fert_data.columns:
        fert_data = fert_data.rename(columns={"Crop Type": "crop_type"})
        
    expected_fert_features = ["temperature", "humidity", "moisture", "nitrogen", "potassium", "phosphorous", "soil_type", "crop_type"]
    X_fert = fert_data[expected_fert_features].copy()
    y_fert = fert_data["Fertilizer Name"]
    
    with open("ai/models/fertilizer_recommendation/label_encoders.json", "r") as f:
        encoders = json.load(f)
    with open("ai/models/fertilizer_recommendation/label_classes.json", "r") as f:
        fert_classes = json.load(f)
        
    for col in ["soil_type", "crop_type"]:
        cat_list = encoders[col]
        cat_map = {cat: idx for idx, cat in enumerate(cat_list)}
        X_fert[col] = X_fert[col].map(cat_map)
        
    # Since dataset is 100 rows, test on the whole 100 rows
    preds_fert = fert_model.predict(X_fert)
    pred_fert_labels = [fert_classes[p] for p in preds_fert]
    
    correct_fert = sum(p == t for p, t in zip(pred_fert_labels, y_fert))
    print(f"Tabular Fertilizer Recommendation Accuracy: {correct_fert}/100 ({correct_fert:.1f}%)")
    
    print("\nSample Comparisons (First 5):")
    for i in range(5):
        print(f"Sample {i+1} - Features: {X_fert.iloc[i].to_dict()}")
        print(f"  Predicted: {pred_fert_labels[i]} | Expected: {y_fert.iloc[i]}")

    # 3. Yield Prediction
    print("\n--- Yield Prediction Test (3 samples) ---")
    yield_model = joblib.load("ai/models/yield_prediction/yield_model.pkl")
    yield_data = pd.read_csv(r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\yield_prediction\data.csv")
    
    X_y = yield_data[["area", "rainfall", "fertilizer", "temperature", "humidity"]]
    y_y = yield_data["yield"]
    
    preds_y = yield_model.predict(X_y)
    print("\nComparisons:")
    for i in range(len(yield_data)):
        print(f"Sample {i+1} - Predicted: {preds_y[i]:.4f} | Expected: {y_y.iloc[i]:.4f}")

    # 4. Crop Price Prediction
    print("\n--- Crop Price Prediction Test (3 samples) ---")
    price_model = joblib.load("ai/models/price_prediction/price_model.pkl")
    price_data = pd.read_csv(r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\price_prediction\data.csv")
    
    X_p = price_data[["feature1", "feature2", "feature3"]]
    y_p = price_data["price"]
    
    preds_p = price_model.predict(X_p)
    print("\nComparisons:")
    for i in range(len(price_data)):
        print(f"Sample {i+1} - Predicted: {preds_p[i]:.4f} | Expected: {y_p.iloc[i]:.4f}")

def run_image_tests():
    print("\n==============================================")
    print("RUNNING IMAGE MODELS VALIDATION TESTS")
    print("==============================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Plant Disease CNN Test (50 samples)
    print("\n--- Plant Disease Model Test (50 samples) ---")
    plant_checkpoint = torch.load("ai/models/image_classification/plant_resnet9_best.pt", map_location=device, weights_only=False)
    plant_classes = plant_checkpoint["classes"]
    
    plant_model = ResNet9(in_channels=3, num_classes=len(plant_classes))
    plant_model.load_state_dict(plant_checkpoint["model_state"])
    plant_model.to(device)
    plant_model.eval()
    
    # Load 50 random images from dataset
    plant_path = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\plant_disease\data"
    all_plant_images = []
    for cls in plant_classes:
        cls_dir = os.path.join(plant_path, cls)
        if os.path.exists(cls_dir):
            for f in os.listdir(cls_dir)[:2]: # take 2 from each class
                all_plant_images.append((os.path.join(cls_dir, f), cls))
                
    np.random.seed(42)
    test_samples = [all_plant_images[idx] for idx in np.random.choice(len(all_plant_images), 50, replace=False)]
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # match training resolution
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    correct = 0
    comparisons = []
    
    for path, expected_cls in test_samples:
        try:
            with Image.open(path) as img:
                tensor = transform(img.convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = plant_model(tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                    pred_cls = plant_classes[pred_idx]
            if pred_cls == expected_cls:
                correct += 1
            comparisons.append((os.path.basename(path), pred_cls, expected_cls))
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    print(f"Plant Disease CNN Accuracy: {correct}/50 ({correct*2:.1f}%)")
    print("\nSample Comparisons (First 5):")
    for fname, pred, exp in comparisons[:5]:
        print(f"File: {fname} | Predicted: {pred} | Expected: {exp}")
        
    # 2. Livestock Disease CNN Test (50 samples)
    print("\n--- Livestock Disease Model Test (50 samples) ---")
    live_checkpoint = torch.load("ai/models/image_classification/livestock_resnet9_best.pt", map_location=device, weights_only=False)
    live_classes = live_checkpoint["classes"]
    
    live_model = ResNet9(in_channels=3, num_classes=len(live_classes))
    live_model.load_state_dict(live_checkpoint["model_state"])
    live_model.to(device)
    live_model.eval()
    
    # Scan datasets to gather test samples
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
    
    all_live_images = []
    # scan cattle
    c_sub = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\cattle_diseases\Cows datasets"
    if os.path.exists(c_sub):
        for sub in os.listdir(c_sub):
            mapped = class_mapping.get(sub)
            if mapped:
                d = os.path.join(c_sub, sub)
                for f in os.listdir(d)[:10]:
                    all_live_images.append((os.path.join(d, f), mapped))
                    
    # scan dog
    d_sub = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\dog_skin_disease\train"
    if os.path.exists(d_sub):
        for sub in os.listdir(d_sub):
            mapped = class_mapping.get(sub)
            if mapped:
                d = os.path.join(d_sub, sub)
                for f in os.listdir(d)[:10]:
                    all_live_images.append((os.path.join(d, f), mapped))
                    
    # scan goat
    g_sub = r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\livestock"
    if os.path.exists(g_sub):
        for sub in os.listdir(g_sub):
            mapped = class_mapping.get(sub)
            if mapped:
                d = os.path.join(g_sub, sub)
                for f in os.listdir(d)[:20]:
                    all_live_images.append((os.path.join(d, f), mapped))
                    
    np.random.shuffle(all_live_images)
    test_samples_l = all_live_images[:50]
    
    correct_l = 0
    comparisons_l = []
    
    for path, expected_cls in test_samples_l:
        try:
            with Image.open(path) as img:
                tensor = transform(img.convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = live_model(tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                    pred_cls = live_classes[pred_idx]
            if pred_cls == expected_cls:
                correct_l += 1
            comparisons_l.append((os.path.basename(path), pred_cls, expected_cls))
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    print(f"Livestock Disease CNN Accuracy: {correct_l}/50 ({correct_l*2:.1f}%)")
    print("\nSample Comparisons (First 5):")
    for fname, pred, exp in comparisons_l[:5]:
        print(f"File: {fname} | Predicted: {pred} | Expected: {exp}")

if __name__ == "__main__":
    run_tabular_tests()
    run_image_tests()
