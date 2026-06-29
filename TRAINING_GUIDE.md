# Smart Agri Portal - Model Training & Integration Guide

## Overview
This document provides step-by-step instructions to train all 6 machine learning models and integrate them into the Smart Agri Portal backend.

## Models to Train

| Model | Type | Framework | Dataset | Target |
|-------|------|-----------|---------|--------|
| Crop Recommendation | Classification | XGBoost | `datasets/crop_recommendation/Crop_recommendation.csv` | `label` |
| Yield Prediction | Regression | Random Forest | `datasets/yield_prediction/data.csv` | `yield` |
| Price Prediction | Regression | Random Forest | `datasets/price_prediction/data.csv` | `price` |
| Fertilizer Recommendation | Classification | XGBoost | `datasets/fertilizer_prediction/Fertilizer Prediction.csv` | `fertilizer` |
| Plant Disease Detection | Image Classification | ResNet9 CNN | `datasets/plant_disease/data/` | Disease class |
| Animal Disease Detection | Image Classification | ResNet9 CNN | `datasets/dog_skin_disease/train/` | Disease class |

## Prerequisites

### 1. Install Dependencies

```bash
# Navigate to project root
cd "C:\Users\Dhanush\OneDrive\Desktop\C&L"

# Install AI module dependencies
pip install -r ai/requirements.txt

# Install AI service dependencies
pip install -r ai_service/requirements.txt
```

### 2. Verify Dataset Availability

```bash
# Check if all datasets are present
dir datasets\crop_recommendation\
dir datasets\yield_prediction\
dir datasets\price_prediction\
dir datasets\fertilizer_prediction\
dir datasets\plant_disease\data\
dir datasets\dog_skin_disease\train\
```

## Training Process

### Option 1: Train All Models (Recommended)

```bash
# From project root directory
python train_all_models.py --datasets "C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets"
```

**Features:**
- Trains all 6 models in sequence
- Provides real-time progress feedback
- Verifies trained models after training
- Generates comprehensive summary report

**Example Output:**
```
======================================================================
 SMART AGRI PORTAL - MODEL TRAINING PIPELINE
======================================================================
Dataset Path: C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets
======================================================================

======================================================================
▶ Training Crop Recommendation (XGBoost)
======================================================================
Loading dataset from C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets\crop_recommendation\Crop_recommendation.csv
Dataset shape: (2200, 8)
Columns: ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall', 'ph', 'label']
Training XGBoost classifier with 22 classes
Training accuracy: 0.9500
Validation accuracy: 0.9200
✓ Saved crop_recommender.pkl

[... more models training ...]

======================================================================
▶ Verifying trained models
======================================================================
✓ Crop Recommendation Model             | Size: 0.45 MB
✓ Yield Prediction Model                | Size: 1.23 MB
✓ Price Prediction Model                | Size: 0.89 MB
✓ Fertilizer Recommendation Model       | Size: 0.52 MB
✓ Plant Disease Detection Model         | Size: 45.67 MB
✓ Animal Disease Detection Model        | Size: 48.23 MB

======================================================================
 TRAINING SUMMARY
======================================================================
✓ PASSED   | crop_recommendation
✓ PASSED   | yield_prediction
✓ PASSED   | price_prediction
✓ PASSED   | fertilizer_recommendation
✓ PASSED   | plant_disease
✓ PASSED   | animal_disease
======================================================================
Results: 6/6 models trained successfully

✓ All models trained and verified successfully!

Models are ready for integration with the AI service.
The FastAPI service will auto-load these models from ai/models/
```

### Option 2: Train Individual Models

```bash
# Crop Recommendation
python ai/training/train_crop_recommendation.py \
  --input "datasets/crop_recommendation/Crop_recommendation.csv" \
  --target label \
  --out-dir ai/models/crop_recommendation

# Yield Prediction
python ai/training/train_yield_prediction.py \
  --input "datasets/yield_prediction/data.csv" \
  --target yield \
  --out-dir ai/models/yield_prediction

# Price Prediction
python ai/training/train_price_prediction.py \
  --input "datasets/price_prediction/data.csv" \
  --target price \
  --out-dir ai/models/price_prediction

# Fertilizer Recommendation
python ai/training/train_fertilizer_recommendation.py \
  --input "datasets/fertilizer_prediction/Fertilizer Prediction.csv" \
  --target fertilizer \
  --out-dir ai/models/fertilizer_recommendation

# Plant Disease Detection (ResNet9)
python ai/training/train_image_classifier.py \
  --data-dir "datasets/plant_disease/data" \
  --out-dir ai/models/image_classification \
  --name plant_resnet9 \
  --epochs 10 \
  --batch-size 32

# Animal Disease Detection (ResNet9)
python ai/training/train_image_classifier.py \
  --data-dir "datasets/dog_skin_disease/train" \
  --out-dir ai/models/image_classification \
  --name animal_resnet9 \
  --epochs 10 \
  --batch-size 32
```

### Option 3: Train Specific Models Only

```bash
# Skip image models, train only tabular
python train_all_models.py \
  --datasets "C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets" \
  --skip-image

# Skip tabular models, train only image
python train_all_models.py \
  --datasets "C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets" \
  --skip-tabular

# Train only specific models
python train_all_models.py \
  --datasets "C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets" \
  --skip-yield \
  --skip-price
```

## Model Output Structure

After successful training, the following directory structure will be created:

```
ai/models/
├── crop_recommendation/
│   ├── crop_recommender.pkl          # Trained model
│   ├── feature_columns.json          # Feature names
│   └── label_classes.json            # Crop class names
├── yield_prediction/
│   ├── yield_predictor.pkl           # Trained model
│   └── feature_columns.json          # Feature names
├── price_prediction/
│   ├── price_predictor.pkl           # Trained model
│   └── feature_columns.json          # Feature names
├── fertilizer_recommendation/
│   ├── fertilizer_recommender.pkl    # Trained model
│   ├── feature_columns.json          # Feature names
│   ├── label_classes.json            # Fertilizer classes
│   └── label_encoders.json           # Category encoders
├── image_classification/
│   ├── plant_resnet9.pt              # Plant disease model checkpoint
│   ├── plant_resnet9_best.pt         # Best plant disease model
│   ├── animal_resnet9.pt             # Animal disease model checkpoint
│   └── animal_resnet9_best.pt        # Best animal disease model
└── resnet9.py                        # Model architecture
```

## Integration with AI Service

The FastAPI AI service automatically loads trained models through the `model_loader.py` module:

### 1. Start the AI Service

```bash
# From project root
cd ai_service
pip install -r requirements.txt

# Run the service
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 2. Verify Model Loading

The service will automatically load models on startup:

```python
# api_service/model_loader.py handles automatic loading
load_crop_recommender()           # Loads crop_recommender.pkl
load_yield_predictor()            # Loads yield_predictor.pkl
load_price_predictor()            # Loads price_predictor.pkl
load_fertilizer_recommender()     # Loads fertilizer_recommender.pkl
load_image_model()                # Loads ResNet9 models
```

### 3. API Endpoints

Once the AI service is running, you can use these endpoints:

```bash
# Crop Recommendation
curl -X POST "http://localhost:8000/api/cropRecommendation" \
  -H "Content-Type: application/json" \
  -d '{
    "nitrogen": 90,
    "phosphorus": 42,
    "potassium": 43,
    "temperature": 20.88,
    "humidity": 82.00,
    "rainfall": 202.94,
    "ph": 6.0,
    "soilType": "loamy"
  }'

# Yield Prediction
curl -X POST "http://localhost:8000/api/yieldPrediction" \
  -H "Content-Type: application/json" \
  -d '{
    "area": 0.5,
    "rainfall": 1202.94,
    "fertilizer": 100.0,
    "soil": "loamy",
    "crop": "Rice",
    "temperature": 25.0,
    "humidity": 75.0
  }'

# Price Prediction
curl -X POST "http://localhost:8000/api/pricePrediction" \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "Rice",
    "state": "Maharashtra",
    "market": "Pune",
    "month": "January"
  }'

# Fertilizer Recommendation
curl -X POST "http://localhost:8000/api/fertilizerRecommendation" \
  -H "Content-Type: application/json" \
  -d '{
    "nitrogen": 50,
    "phosphorus": 30,
    "potassium": 20,
    "temperature": 25.0,
    "humidity": 70.0,
    "moisture": 40.0,
    "cropType": "Maize"
  }'

# Plant Disease Detection
curl -X POST "http://localhost:8000/api/predictPlant" \
  -F "file=@path/to/plant/image.jpg"

# Animal Disease Detection
curl -X POST "http://localhost:8000/api/predictAnimal" \
  -F "file=@path/to/animal/image.jpg"
```

## Deployment with Docker

### Build AI Service Docker Image

```bash
cd ai_service
docker build -t agri-ai-service:latest .
docker run -p 8000:8000 \
  -v $(pwd)/../ai/models:/app/ai/models \
  agri-ai-service:latest
```

### Full Stack Deployment

```bash
# From project root with docker-compose.yml
docker-compose up --build

# Expected services:
# - AI Service: http://localhost:8000
# - Backend: http://localhost:8080
# - Frontend: http://localhost:3000 (if available)
```

## Troubleshooting

### Common Issues

#### 1. CSV File Not Found
```
Error: Dataset path does not exist
Solution: Check dataset paths are correct and files exist
          Ensure dataset folder structure matches expectations
```

#### 2. CUDA/GPU Issues
```
Error: CUDA out of memory
Solution: Reduce batch size: --batch-size 16
          Use CPU: Set CUDA_VISIBLE_DEVICES=""
          python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. Missing Classes in Image Dataset
```
Error: No image files found in directory
Solution: Ensure ImageFolder structure: data_dir/class1/image1.jpg
          Each subdirectory is treated as a class
```

#### 4. Model Not Loading in API
```
Error: Model loading failed
Solution: Verify model files exist in ai/models/
          Check file permissions
          Restart AI service
```

### Validation Checklist

After training, verify:

```bash
# ✓ Check model files exist and have reasonable sizes
ls -lh ai/models/crop_recommendation/
ls -lh ai/models/yield_prediction/
ls -lh ai/models/price_prediction/
ls -lh ai/models/fertilizer_recommendation/
ls -lh ai/models/image_classification/

# ✓ Verify metadata files
cat ai/models/crop_recommendation/label_classes.json
cat ai/models/fertilizer_recommendation/feature_columns.json

# ✓ Test model loading
python -c "
from ai_service.model_loader import *
print('Loading models...')
print('Crop:', load_crop_recommender() is not None)
print('Yield:', load_yield_predictor() is not None)
print('Price:', load_price_predictor() is not None)
print('Fertilizer:', load_fertilizer_recommender() is not None)
print('Image:', load_image_model() is not None)
print('All models loaded successfully!')
"

# ✓ Start AI service and test endpoints
python -m uvicorn ai_service.app:app --reload
# Then access http://localhost:8000/docs for interactive API docs
```

## Performance Expectations

| Model | Expected Accuracy/R² | Training Time | Model Size |
|-------|----------------------|---------------|-----------| 
| Crop Recommendation | ~92% | 30-60s | 0.4-0.8 MB |
| Yield Prediction | ~0.85 R² | 20-40s | 0.8-1.2 MB |
| Price Prediction | ~0.80 R² | 15-30s | 0.7-1.0 MB |
| Fertilizer Recommendation | ~94% | 20-45s | 0.4-0.7 MB |
| Plant Disease Detection | ~88% | 2-5 min | 40-50 MB |
| Animal Disease Detection | ~85% | 2-5 min | 40-50 MB |

*Note: Times vary based on hardware (CPU vs GPU), batch size, and dataset size*

## Next Steps

1. ✓ Run training pipeline: `python train_all_models.py --datasets ./datasets`
2. ✓ Verify model files are created
3. ✓ Start AI service: `python -m uvicorn ai_service.app:app --reload`
4. ✓ Test API endpoints (use curl or Postman)
5. ✓ Integrate with backend (Java/Spring Boot)
6. ✓ Connect frontend to API endpoints
7. ✓ Deploy with Docker Compose

## Support

For detailed model information, see individual training scripts:
- `ai/training/train_crop_recommendation.py`
- `ai/training/train_yield_prediction.py`
- `ai/training/train_price_prediction.py`
- `ai/training/train_fertilizer_recommendation.py`
- `ai/training/train_image_classifier.py`

Model loading logic:
- `ai_service/model_loader.py`
- `ai_service/app.py`

---
**Last Updated:** 2026-06-29
**Status:** Ready for training
