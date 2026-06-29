import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
from typing import Optional
import io
import os
import json
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# model loader in the service
from .model_loader import (
    load_crop_recommender,
    load_yield_predictor,
    load_price_predictor,
    load_fertilizer_recommender,
    load_fertilizer_metadata,
    load_image_model,
)

# attempt to import ResNet9 for inference construction
try:
    from ai.models.resnet9 import ResNet9
except Exception:
    ResNet9 = None

app = FastAPI(
    title="AGRI-DIAGNOSE AI Microservice",
    description="Python FastAPI endpoints for Plant Pathology, Livestock Health, Crop & Yield recommendations",
    version="1.0.0"
)

# Enable CORS for cross-service calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REQUEST SCHEMAS ---

class CropRecInputs(BaseModel):
    state: str
    district: str
    season: str
    soilType: str
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    rainfall: float

class YieldInputs(BaseModel):
    area: float
    rainfall: float
    fertilizer: float
    soil: str
    crop: str
    temperature: float
    humidity: float

class PriceInputs(BaseModel):
    crop: str
    state: str
    market: str
    month: str

class FertilizerInputs(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    moisture: float
    crop: Optional[str] = None
    soilType: Optional[str] = None
    cropType: Optional[str] = None

# --- ENDPOINTS ---

@app.post("/api/predictPlant")
async def predict_plant(file: UploadFile = File(...)):
    # Try model-based inference first
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        ckpt = load_image_model('ai/models/image_classification/plant_resnet9.pt')
        if ckpt and ResNet9 is not None:
            # build model and run
            classes = ckpt.get('classes') if isinstance(ckpt, dict) else None
            model_state = ckpt.get('model_state') if isinstance(ckpt, dict) else None
            if model_state is not None and classes is not None:
                model = ResNet9(in_channels=3, num_classes=len(classes))
                model.load_state_dict(model_state)
                model.eval()
                transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
                ])
                tensor = transform(img)
                tensor = torch.unsqueeze(tensor, 0)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                top_idx = int(np.argmax(probs))
                disease = classes[top_idx]
                confidence = float(probs[top_idx]*100.0)
                return {
                    'diseaseName': disease,
                    'confidence': confidence,
                    'source': 'model'
                }
    except Exception:
        # fallback to heuristic when model fails
        pass

    # Heuristic fallback based on filename
    filename = file.filename.lower()
    disease = "Rice Blast (Magnaporthe oryzae)"
    confidence = 88.5
    severity = 0.45
    desc = "Rice blast is caused by the fungus Magnaporthe oryzae. It is one of the most destructive diseases of rice worldwide."
    symptoms = "Spindle-shaped spots on leaves with gray or whitish centers and brown borders. Leaf collar rot and neck rot."
    actions = "Avoid excessive nitrogen application. Keep fields flooded. Remove infected crop residues."
    chemical = "Foliar spray of Tricyclazole 75% WP @ 120g/acre or Carbendazim 50% WP @ 200g/acre."
    organic = "Spray Pseudomonas fluorescens formulation @ 5g/liter or Neem oil @ 3%."
    prevent = "Use resistant rice varieties. Treat seeds before sowing. Maintain proper spacing."

    if "potato" in filename or "solanum" in filename:
        disease = "Potato Late Blight (Phytophthora infestans)"
        confidence = 92.0
        severity = 0.75
        desc = "Late blight is a devastating disease caused by the water mold Phytophthora infestans."
        symptoms = "Dark, water-soaked patches on leaves that enlarge rapidly, white velvet mold on leaf undersides, dry rot in tubers."
        actions = "Destroy infected foliage. Halt overhead sprinkling; apply drip irrigation instead."
        chemical = "Apply Mancozeb @ 2g/liter or Metalaxyl-M + Mancozeb formulation @ 2.5g/liter."
        organic = "Spray copper oxychloride suspensions. Ensure proper earthing-up."
        prevent = "Plant certified disease-free seed tubers. Keep a minimum 3-year crop rotation."
    elif "tomato" in filename or "lycopersicum" in filename:
        disease = "Tomato Early Blight (Alternaria solani)"
        confidence = 81.5
        severity = 0.35
        desc = "Early blight is caused by the fungus Alternaria solani. It primarily affects older foliage, causing defoliation."
        symptoms = "Dark spots with concentric target-like rings on older leaves, yellowing and leaf drops."
        actions = "Prune lower leaves that touch the ground. Apply straw mulch."
        chemical = "Spray Chlorothalonil @ 2g/liter or Copper fungicides at 7-10 day intervals."
        organic = "Apply compost tea or spray Bacillus subtilis formulations."
        prevent = "Rotate crops. Maintain proper soil fertilization. Avoid overhead irrigation."
    elif "corn" in filename or "maize" in filename or "rust" in filename:
        disease = "Corn Common Rust (Puccinia sorghi)"
        confidence = 89.0
        severity = 0.30
        desc = "Common rust is caused by the fungus Puccinia sorghi. It is favored by high humidity."
        symptoms = "Golden-brown to reddish-orange powdery pustules on both upper and lower leaf surfaces."
        actions = "Tillage to bury infected crop residues. Harvest early if crop is mature."
        chemical = "Foliar application of Mancozeb or Pyraclostrobin if rust appears early."
        organic = "Dust with sulfur powder. Use neem oil extracts."
        prevent = "Plant rust-resistant hybrids. Balance nitrogen application."

    return {
        "diseaseName": disease,
        "confidence": confidence,
        "severityScore": severity,
        "description": desc,
        "symptoms": symptoms,
        "immediateActions": actions,
        "treatment": "Apply appropriate fungicides and regulate irrigation.",
        "recommendedMedicines": chemical,
        "organicTreatment": organic,
        "preventiveMeasures": prevent,
        "source": "heuristic"
    }

@app.post("/api/predictAnimal")
async def predict_animal(file: UploadFile = File(...)):
    # Try model-based inference first
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        ckpt = load_image_model('ai/models/image_classification/animal_resnet9.pt')
        if ckpt and ResNet9 is not None:
            classes = ckpt.get('classes') if isinstance(ckpt, dict) else None
            model_state = ckpt.get('model_state') if isinstance(ckpt, dict) else None
            if model_state is not None and classes is not None:
                model = ResNet9(in_channels=3, num_classes=len(classes))
                model.load_state_dict(model_state)
                model.eval()
                transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
                ])
                tensor = transform(img)
                tensor = torch.unsqueeze(tensor, 0)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                top_idx = int(np.argmax(probs))
                disease = classes[top_idx]
                confidence = float(probs[top_idx]*100.0)
                return {
                    'diseaseName': disease,
                    'confidence': confidence,
                    'source': 'model'
                }
    except Exception:
        pass

    # Heuristic fallback
    filename = file.filename.lower()
    animal = "Cow"
    disease = "Foot and Mouth Disease (FMD)"
    confidence = 94.2
    severity = "High"
    symptoms = "High fever, drooling, stringy saliva, vesicles/blisters on the mouth, tongue, and interdigital space of hooves. Lameness."
    treatment = "Symptomatic treatment. Wash hooves with mild antiseptic, apply soda ash on lesions, feed soft mash."
    isolation = "IMMEDIATE QUARANTINE. Separate infected animal at least 100 meters away from healthy herd."
    emergency = "Immediately notify local veterinary officer. Disinfect barns with 4% sodium carbonate. Block milk transport."
    nearby_vet = "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"

    if "sheep" in filename or "goat" in filename:
        animal = "Goat"
        disease = "Peste des Petits Ruminants (PPR)"
        confidence = 86.8
        severity = "High"
        symptoms = "High fever, nasal discharge, mouth sores, gum necrosis, foul diarrhea, labored respiration."
        treatment = "Supportive care. Broad-spectrum antibiotics for secondary pneumonia. Rehydration."
        isolation = "Isolate goat in warm, dry shelter. Quarantine new livestock for 21 days."
        emergency = "Highly contagious virus. Restrict ruminant shipping. Trigger ring vaccination."
        nearby_vet = "Dr. Amit Verma, Regional Outpost (+91 9444455555)"
    elif "buffalo" in filename or "mastitis" in filename:
        animal = "Buffalo"
        disease = "Mastitis"
        confidence = 95.0
        severity = "Medium"
        symptoms = "Hot, swollen, painful udder quarters, watery or clotted milk, decline in milk yields."
        treatment = "Intramammary antibiotic infusion by a vet. Regular complete stripping of infected milk."
        isolation = "Milk affected buffalo last. Disinfect milking equipment with chlorine wash."
        emergency = "Run CMT (California Mastitis Test) on whole herd. Maintain dry barn bedding."
        nearby_vet = "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"
    elif "lumpy" in filename or "lsd" in filename:
        animal = "Cow"
        disease = "Lumpy Skin Disease (LSD)"
        confidence = 90.5
        severity = "Medium"
        symptoms = "Fever, eruption of firm round skin nodules (2-5cm), swollen limbs, nasal discharge."
        treatment = "Symptomatic treatment. Antibiotics for secondary infection, anti-inflammatory drugs."
        isolation = "Strict isolation. Vectors control (spray barns to kill mosquitoes, flies, ticks)."
        emergency = "Report outbreak to authorities. Vaccinate surrounding healthy stock."
        nearby_vet = "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"

    return {
        "animalType": animal,
        "diseaseName": disease,
        "confidence": confidence,
        "severity": severity,
        "symptoms": symptoms,
        "treatment": treatment,
        "isolationGuidance": isolation,
        "emergencyAdvice": emergency,
        "nearbyVet": nearby_vet,
        "source": "heuristic"
    }

@app.post("/api/cropRecommendation")
async def crop_recommendation(inputs: CropRecInputs):
    # Try to use a saved XGBoost model if available
    try:
        model = load_crop_recommender()
        if model is not None:
            # build dataframe with numeric fields only - training must match this ordering
            df = pd.DataFrame([{
                'nitrogen': inputs.nitrogen,
                'phosphorus': inputs.phosphorus,
                'potassium': inputs.potassium,
                'temperature': inputs.temperature,
                'humidity': inputs.humidity,
                'rainfall': inputs.rainfall
            }])
            try:
                preds = model.predict(df)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(df)
                    conf = float(np.max(proba) * 100.0)
                else:
                    conf = 90.0
                return {
                    'recommendedCrop': str(preds[0]),
                    'confidence': conf,
                    'source': 'model'
                }
            except Exception:
                # if model prediction fails, fall back to heuristic
                pass
    except Exception:
        pass

    # Heuristic fallback
    n, p, k = inputs.nitrogen, inputs.phosphorus, inputs.potassium
    rainfall = inputs.rainfall

    recommended = "Maize"
    confidence = 88.0
    expected_yield = 4.5
    reason = "Balanced NPK and moderate rainfall suited for Zea mays grain development."

    if rainfall > 180:
        recommended = "Rice"
        confidence = 94.0
        expected_yield = 3.9
        reason = "Paddy rice requires waterlogged soil profile (>180mm rainfall) and clayey conditions."
    elif k > 140:
        recommended = "Grapes"
        confidence = 92.5
        expected_yield = 12.0
        reason = "Horticultural grapes require elevated Potassium (K) levels for sugar and skin development."
    elif n < 30 and p > 60:
        recommended = "Chickpea"
        confidence = 89.0
        expected_yield = 1.9
        reason = "Chickpea is a legume fixing atmospheric nitrogen, requiring higher Phosphorus (P) for root nodulation."
    elif rainfall < 60:
        recommended = "Mung Bean"
        confidence = 86.5
        expected_yield = 1.3
        reason = "Short-duration pulse highly suited for semi-arid and low moisture regions."

    return {
        "recommendedCrop": recommended,
        "confidence": confidence,
        "expectedYield": expected_yield,
        "reason": reason,
        "source": "heuristic"
    }

@app.post("/api/yieldPrediction")
async def yield_prediction(inputs: YieldInputs):
    # Try to use saved RF regressor
    try:
        model = load_yield_predictor()
        if model is not None:
            df = pd.DataFrame([{
                'area': inputs.area,
                'rainfall': inputs.rainfall,
                'fertilizer': inputs.fertilizer,
                'temperature': inputs.temperature,
                'humidity': inputs.humidity
            }])
            try:
                pred = model.predict(df)
                return {'predictedYield': float(pred[0]), 'unit': 'Metric Tons', 'source': 'model'}
            except Exception:
                pass
    except Exception:
        pass

    # Heuristic fallback
    base = 3.6
    crop_lower = inputs.crop.lower()
    if "rice" in crop_lower:
        base = 4.1
    elif "maize" in crop_lower:
        base = 5.2
    elif "grapes" in crop_lower:
        base = 14.8
    elif "chickpea" in crop_lower:
        base = 2.1

    fert_factor = min(1.3, 0.7 + (inputs.fertilizer / 220.0))
    rain_factor = min(1.2, 0.8 + (inputs.rainfall / 1100.0))
    
    yield_val = inputs.area * base * fert_factor * rain_factor
    yield_val = round(yield_val, 2)

    return {
        "predictedYield": yield_val,
        "unit": "Metric Tons",
        "source": "heuristic"
    }

@app.post("/api/pricePrediction")
async def price_prediction(inputs: PriceInputs):
    # Try model-based RF price predictor
    try:
        model = load_price_predictor()
        if model is not None:
            # attempt numeric features where possible
            df = pd.DataFrame([{
                'crop': inputs.crop,
                'state': inputs.state,
                'market': inputs.market,
                'month': inputs.month
            }])
            try:
                pred = model.predict(df)
                return {'expectedPrice': float(pred[0]), 'currency': 'INR per Quintal', 'source': 'model'}
            except Exception:
                pass
    except Exception:
        pass

    # Heuristic fallback
    crop_lower = inputs.crop.lower()
    base = 2100.0
    trend = "Stable"

    if "rice" in crop_lower:
        base = 2450.0
        trend = "Bullish (Strong export margins and food grain demand)"
    elif "maize" in crop_lower:
        base = 2080.0
        trend = "Slightly Bullish (Increasing feed mill demands)"
    elif "grapes" in crop_lower:
        base = 8200.0
        trend = "Volatile (Highly dependent on cold storage and export contracts)"
    elif "chickpea" in crop_lower:
        base = 5250.0
        trend = "Bearish (Large harvest buffers in central depots)"

    price_val = base + random.uniform(-80.0, 120.0)
    price_val = round(price_val, 2)

    return {
        "expectedPrice": price_val,
        "marketTrend": trend,
        "currency": "INR per Quintal",
        "source": "heuristic"
    }

@app.post("/api/fertilizerRecommendation")
async def fertilizer_recommendation(inputs: FertilizerInputs):
    # Try model-based prediction first
    try:
        model = load_fertilizer_recommender()
        metadata = load_fertilizer_metadata()
        if model is not None:
            soil_value = inputs.soilType or inputs.crop or "unknown"
            crop_value = inputs.cropType or inputs.crop or "unknown"
            df = pd.DataFrame([{
                'temperature': inputs.temperature,
                'humidity': inputs.humidity,
                'moisture': inputs.moisture,
                'nitrogen': inputs.nitrogen,
                'potassium': inputs.potassium,
                'phosphorous': inputs.phosphorus,
                'soil_type': soil_value,
                'crop_type': crop_value
            }])
            feature_columns = metadata.get('feature_columns', [
                'temperature', 'humidity', 'moisture', 'nitrogen', 'potassium', 'phosphorous', 'soil_type', 'crop_type'
            ])
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_columns]

            label_encoders = metadata.get('label_encoders', {})
            for col in ['soil_type', 'crop_type']:
                if col in df.columns and col in label_encoders:
                    value = df.at[0, col]
                    try:
                        df.at[0, col] = label_encoders[col].index(value)
                    except ValueError:
                        df.at[0, col] = 0

            df = df.astype(float)
            try:
                preds = model.predict(df)
                label_classes = metadata.get('label_classes')
                predicted_index = int(preds[0])
                recommended = label_classes[predicted_index] if label_classes and predicted_index < len(label_classes) else str(predicted_index)
                conf = 92.0
                return {
                    'recommendedFertilizer': recommended,
                    'confidence': conf,
                    'source': 'model'
                }
            except Exception:
                pass
    except Exception:
        pass

    # Heuristic fallback
    n, p, k = inputs.nitrogen, inputs.phosphorus, inputs.potassium
    fert = "NPK 19-19-19 (Balanced)"
    qty = "120 kg per acre"
    organic = "Apply compost manure @ 5 tons/acre. Spray bio-extracts."

    if n < 40:
        fert = "Urea (46% Nitrogen)"
        qty = "100 kg per acre"
        organic = "Grow green manure crops. Apply neem seed cake @ 150 kg/acre."
    elif p < 30:
        fert = "DAP (Diammonium Phosphate)"
        qty = "80 kg per acre"
        organic = "Apply rock phosphate + Phosphate Solubilizing Bacteria (PSB) cultures."
    elif k < 30:
        fert = "MOP (Muriate of Potash)"
        qty = "60 kg per acre"
        organic = "Apply wood ash @ 200 kg per acre or banana peel compost."

    return {
        "recommendedFertilizer": fert,
        "applicationQuantity": qty,
        "organicAlternatives": organic,
        "source": "heuristic"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
