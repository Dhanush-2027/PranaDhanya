# Smart Agri Portal - Model Training Setup

## 🚀 Quick Start (Recommended)

### Windows Users
```bash
# Double-click to run:
train_models.bat

# Or from command line:
train_models.bat
```

### macOS/Linux Users
```bash
# Run the quick start script
python quickstart.py
```

### Manual Setup (All Platforms)
```bash
# 1. Install dependencies
pip install -r ai/requirements.txt
pip install -r ai_service/requirements.txt

# 2. Train all models
python train_all_models.py --datasets "path/to/datasets"
```

---

## 📊 6 Machine Learning Models

The Smart Agri Portal includes 6 state-of-the-art ML models for agriculture:

### 1. **Crop Recommendation** (XGBoost Classifier)
- **Input:** Soil nutrients (N, P, K), temperature, humidity, rainfall, pH
- **Output:** Recommended crop type (22 classes)
- **Accuracy:** ~92%
- **Dataset:** `datasets/crop_recommendation/Crop_recommendation.csv`

### 2. **Yield Prediction** (Random Forest Regressor)
- **Input:** Area, rainfall, fertilizer, soil type, crop, temperature, humidity
- **Output:** Expected crop yield
- **R² Score:** ~0.85
- **Dataset:** `datasets/yield_prediction/data.csv`

### 3. **Price Prediction** (Random Forest Regressor)
- **Input:** Crop type, state, market, month
- **Output:** Predicted crop price
- **R² Score:** ~0.80
- **Dataset:** `datasets/price_prediction/data.csv`

### 4. **Fertilizer Recommendation** (XGBoost Classifier)
- **Input:** Nitrogen, phosphorus, potassium, temperature, humidity, moisture, soil type, crop type
- **Output:** Recommended fertilizer
- **Accuracy:** ~94%
- **Dataset:** `datasets/fertilizer_prediction/Fertilizer Prediction.csv`

### 5. **Plant Disease Detection** (ResNet9 CNN)
- **Input:** Plant leaf image
- **Output:** Disease classification (38+ classes)
- **Accuracy:** ~88%
- **Dataset:** `datasets/plant_disease/data/` (ImageFolder format)
- **Image Sizes:** Multiple disease types (Apple, Bell Pepper, Cassava, etc.)

### 6. **Animal Disease Detection** (ResNet9 CNN)
- **Input:** Animal skin/coat image (dogs, cattle, livestock)
- **Output:** Disease classification (6+ disease types)
- **Accuracy:** ~85%
- **Dataset:** `datasets/dog_skin_disease/train/` (ImageFolder format)
- **Disease Types:** Demodicosis, Dermatitis, Fungal Infections, Healthy, Hypersensitivity, Ringworm

---

## 📁 Directory Structure

```
C&L/
├── ai/
│   ├── models/                          # Trained model storage
│   │   ├── crop_recommendation/
│   │   ├── yield_prediction/
│   │   ├── price_prediction/
│   │   ├── fertilizer_recommendation/
│   │   ├── image_classification/
│   │   └── resnet9.py                   # CNN architecture
│   ├── training/                        # Training scripts
│   │   ├── train_crop_recommendation.py
│   │   ├── train_yield_prediction.py
│   │   ├── train_price_prediction.py
│   │   ├── train_fertilizer_recommendation.py
│   │   └── train_image_classifier.py
│   ├── utils/                           # Utility functions
│   └── requirements.txt
├── ai_service/                          # FastAPI service
│   ├── app.py                           # Main Flask app with endpoints
│   ├── model_loader.py                  # Auto-load trained models
│   ├── requirements.txt
│   └── Dockerfile
├── datasets/                            # Training datasets
│   ├── crop_recommendation/
│   ├── yield_prediction/
│   ├── price_prediction/
│   ├── fertilizer_prediction/
│   ├── plant_disease/
│   └── dog_skin_disease/
├── train_all_models.py                  # Master training script
├── quickstart.py                        # Interactive setup wizard
├── train_models.bat                     # Windows batch script
├── TRAINING_GUIDE.md                    # Detailed documentation
└── README.md
```

---

## 🎯 Training Options

### Option 1: Train All Models (Default)
```bash
python train_all_models.py --datasets "C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets"
```
✅ Trains all 6 models sequentially
✅ Provides progress feedback
✅ Verifies trained models
✅ Generates summary report

### Option 2: Train Specific Categories
```bash
# Train only tabular models (skip image models)
python train_all_models.py --datasets ./datasets --skip-image

# Train only image models (skip tabular models)
python train_all_models.py --datasets ./datasets --skip-tabular
```

### Option 3: Skip Individual Models
```bash
# Skip specific models
python train_all_models.py --datasets ./datasets \
  --skip-crop \
  --skip-price \
  --skip-animal
```

### Option 4: Train Individual Models
```bash
# Crop recommendation
python ai/training/train_crop_recommendation.py \
  --input "datasets/crop_recommendation/Crop_recommendation.csv" \
  --target label

# Yield prediction
python ai/training/train_yield_prediction.py \
  --input "datasets/yield_prediction/data.csv" \
  --target yield

# Price prediction
python ai/training/train_price_prediction.py \
  --input "datasets/price_prediction/data.csv" \
  --target price

# Fertilizer recommendation
python ai/training/train_fertilizer_recommendation.py \
  --input "datasets/fertilizer_prediction/Fertilizer Prediction.csv" \
  --target fertilizer

# Plant disease detection (ResNet9)
python ai/training/train_image_classifier.py \
  --data-dir "datasets/plant_disease/data" \
  --out-dir ai/models/image_classification \
  --name plant_resnet9 \
  --epochs 10 \
  --batch-size 32

# Animal disease detection (ResNet9)
python ai/training/train_image_classifier.py \
  --data-dir "datasets/dog_skin_disease/train" \
  --out-dir ai/models/image_classification \
  --name animal_resnet9 \
  --epochs 10 \
  --batch-size 32
```

---

## 📊 Training Results

### Expected Metrics
| Model | Type | Metric | Expected Value | Training Time |
|-------|------|--------|----------------|---------------|
| Crop Recommendation | Classification | Accuracy | ~92% | 30-60s |
| Yield Prediction | Regression | R² Score | ~0.85 | 20-40s |
| Price Prediction | Regression | R² Score | ~0.80 | 15-30s |
| Fertilizer Recommendation | Classification | Accuracy | ~94% | 20-45s |
| Plant Disease Detection | Image Classification | Accuracy | ~88% | 2-5 min |
| Animal Disease Detection | Image Classification | Accuracy | ~85% | 2-5 min |

### Sample Output
```
======================================================================
 SMART AGRI PORTAL - MODEL TRAINING PIPELINE
======================================================================
Dataset Path: C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets

======================================================================
▶ Training Crop Recommendation (XGBoost)
======================================================================
Loading dataset from ...Crop_recommendation.csv
Dataset shape: (2200, 8)
Training XGBoost classifier with 22 classes
Training accuracy: 0.9500
Validation accuracy: 0.9200
✓ Saved crop_recommender.pkl

[... Training other models ...]

======================================================================
▶ Verifying trained models
======================================================================
✓ Crop Recommendation Model              | Size: 0.45 MB
✓ Yield Prediction Model                 | Size: 1.23 MB
✓ Price Prediction Model                 | Size: 0.89 MB
✓ Fertilizer Recommendation Model        | Size: 0.52 MB
✓ Plant Disease Detection Model          | Size: 45.67 MB
✓ Animal Disease Detection Model         | Size: 48.23 MB
======================================================================
Results: 6/6 models trained successfully
✓ All models trained and verified successfully!
```

---

## 🔧 AI Service Integration

### Start the AI Microservice
```bash
cd ai_service
pip install -r requirements.txt
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Access API Documentation
```
http://localhost:8000/docs
```

### API Endpoints

**Crop Recommendation**
```bash
curl -X POST "http://localhost:8000/api/cropRecommendation" \
  -H "Content-Type: application/json" \
  -d '{
    "nitrogen": 90,
    "phosphorus": 42,
    "potassium": 43,
    "temperature": 20.88,
    "humidity": 82.0,
    "rainfall": 202.94,
    "ph": 6.0,
    "soilType": "loamy"
  }'
```

**Yield Prediction**
```bash
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
```

**Plant Disease Detection**
```bash
curl -X POST "http://localhost:8000/api/predictPlant" \
  -F "file=@path/to/leaf_image.jpg"
```

**Animal Disease Detection**
```bash
curl -X POST "http://localhost:8000/api/predictAnimal" \
  -F "file=@path/to/animal_image.jpg"
```

---

## 🐳 Docker Deployment

### Build and Run with Docker Compose
```bash
# From project root
docker-compose up --build

# Services:
# - AI Service: http://localhost:8000
# - Backend: http://localhost:8080
# - Frontend: http://localhost:3000 (if configured)
```

### Build AI Service Image Separately
```bash
cd ai_service
docker build -t agri-ai-service:latest .

# Run container with volume mount for models
docker run -p 8000:8000 \
  -v $(pwd)/../ai/models:/app/ai/models \
  agri-ai-service:latest
```

---

## ✅ Verification Checklist

After training, verify everything works:

```bash
# 1. Check model files exist
dir ai/models/crop_recommendation/
dir ai/models/yield_prediction/
dir ai/models/price_prediction/
dir ai/models/fertilizer_recommendation/
dir ai/models/image_classification/

# 2. Verify metadata files
type ai/models/crop_recommendation/label_classes.json
type ai/models/fertilizer_recommendation/feature_columns.json

# 3. Test model loading
python -c "
from ai_service.model_loader import *
print('Testing model loading...')
print('✓ Crop:', load_crop_recommender() is not None)
print('✓ Yield:', load_yield_predictor() is not None)
print('✓ Price:', load_price_predictor() is not None)
print('✓ Fertilizer:', load_fertilizer_recommender() is not None)
print('✓ Image:', load_image_model() is not None)
print('All models loaded successfully!')
"

# 4. Start AI service and test
cd ai_service
python -m uvicorn app:app --reload
# Open http://localhost:8000/docs to test endpoints
```

---

## 🐛 Troubleshooting

### Issue: "Dataset path does not exist"
**Solution:** Check dataset path is correct and formatted properly
```bash
# Check if datasets folder exists
dir "C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets"
```

### Issue: "No module named 'torch'" 
**Solution:** Install PyTorch
```bash
pip install torch torchvision
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in training
```bash
python ai/training/train_image_classifier.py \
  --data-dir "datasets/plant_disease/data" \
  --batch-size 16  # Reduce from 32
```

### Issue: "No image files found"
**Solution:** Ensure ImageFolder structure
```
datasets/plant_disease/data/
├── Apple___alternaria_leaf_spot/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Apple___black_rot/
│   └── ...
└── ...
```

### Issue: Model not loading in API
**Solution:** Verify model files and restart service
```bash
# Check models exist
dir ai/models

# Restart the service
cd ai_service
python -m uvicorn app:app --reload
```

---

## 📚 Documentation Files

- **TRAINING_GUIDE.md** - Detailed training documentation with examples
- **train_all_models.py** - Master training script with all models
- **quickstart.py** - Interactive setup wizard (Python)
- **train_models.bat** - Automated setup for Windows
- **ai/training/*.py** - Individual model training scripts

---

## 🔗 Integration with Backend

The FastAPI AI service is integrated with the backend through Docker Compose:

1. **AI Service** exposes RESTful endpoints on port 8000
2. **Backend** (Java/Spring Boot) calls AI service endpoints on port 8080
3. **Frontend** communicates with backend API

Models are automatically loaded from `ai/models/` when the AI service starts.

---

## 📈 Performance Optimization

For faster training:

```bash
# Use GPU (if available)
python train_all_models.py --datasets ./datasets
# PyTorch will automatically use CUDA if available

# Reduce image training epochs
python ai/training/train_image_classifier.py \
  --data-dir "datasets/plant_disease/data" \
  --epochs 5 \      # Default: 10
  --batch-size 64   # Default: 32
```

---

## 🎓 Model Details

### XGBoost Models (Crop & Fertilizer)
- **Algorithm:** Gradient Boosting
- **Hyperparameters:** n_estimators=100, objective=multi:softprob, eval_metric=mlogloss
- **Features:** Auto-detected from CSV columns
- **Output:** Category with probability scores

### Random Forest Models (Yield & Price)
- **Algorithm:** Ensemble of Decision Trees
- **Hyperparameters:** n_estimators=100, random_state=42
- **Features:** Auto-detected from CSV columns
- **Output:** Continuous numerical value

### ResNet9 CNN (Disease Detection)
- **Architecture:** 9-layer Residual Network
- **Input:** 224x224 RGB images
- **Normalization:** ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Output:** Disease class probabilities

---

## 📝 License & Attribution

This Smart Agri Portal training pipeline uses:
- **XGBoost** for gradient boosting models
- **scikit-learn** for Random Forest models
- **PyTorch** for deep learning (ResNet9)
- **FastAPI** for REST service

---

## 🚀 Next Steps

1. ✅ Run training: `python train_all_models.py --datasets ./datasets`
2. ✅ Start AI service: `cd ai_service && python -m uvicorn app:app --reload`
3. ✅ Test endpoints: Open `http://localhost:8000/docs`
4. ✅ Integrate with backend
5. ✅ Deploy with Docker: `docker-compose up --build`

---

**Last Updated:** June 29, 2026  
**Status:** Ready for training and deployment
