# Enhanced Plant Disease Detection - Implementation Guide

## Overview

This enhanced system significantly improves plant disease detection accuracy through:

1. **Advanced Model Architecture**: EfficientNet-B3 (pre-trained on ImageNet)
2. **Enhanced Data Augmentation**: Albumentations with plant-specific transformations
3. **Training Improvements**: Mixup, label smoothing, class weighting, warmup + cosine annealing
4. **Test-Time Augmentation (TTA)**: Averages predictions across multiple augmentations for better accuracy

## Key Improvements

### 1. Model Architecture
- **Previous**: ResNet9 (custom architecture)
- **Enhanced**: EfficientNet-B3 (state-of-the-art, pre-trained on ImageNet)
- **Benefit**: Better feature extraction, higher accuracy, proven architecture

### 2. Data Augmentation
- **Previous**: Basic flips and color jitter
- **Enhanced**: Advanced Albumentations including:
  - Random crops, rotations, flips
  - Color/brightness/contrast adjustments
  - Blur and noise simulation
  - Weather effects (rain, fog)
  - Geometric transformations
- **Benefit**: Better generalization to real-world conditions

### 3. Training Techniques
- **Mixup Augmentation**: Blends images and labels for better generalization
- **Label Smoothing**: Reduces overconfidence, improves calibration
- **Class Weighting**: Handles imbalanced datasets
- **Warmup + Cosine Annealing**: Optimal learning rate schedule
- **Early Stopping**: Prevents overfitting
- **Benefit**: More robust model with better generalization

### 4. Test-Time Augmentation (TTA)
- **Feature**: Runs prediction on 6 augmented versions of each image
- **Averages**: Predictions are averaged for final result
- **Benefit**: 2-5% accuracy improvement, especially on edge cases

## Files Created

### Training Pipeline
- `ai/augmentation.py` - Advanced augmentation transforms
- `ai/models/efficientnet_model.py` - EfficientNet-B3 model
- `ai/training/train_plant_enhanced.py` - Enhanced training script
- `ai/predict_enhanced.py` - Enhanced inference with TTA

### Utility Scripts
- `train_enhanced_model.py` - Quick training launcher
- `test_enhanced_model.py` - Model testing script

### Service Integration
- `ai_service/app.py` - Updated to use enhanced model with TTA
- `ai_service/requirements.txt` - Updated dependencies
- `ai/requirements.txt` - Updated dependencies

## How to Use

### Step 1: Train the Enhanced Model

Run the training script:

```bash
python train_enhanced_model.py
```

**Expected Output:**
- Training progress with accuracy metrics
- Best model saved to: `ai/models/image_classification/plant_disease_enhanced.pth`
- Class labels saved to: `ai/models/image_classification/plant_classes_enhanced.json`
- Test metrics saved to: `ai/models/image_classification/plant_metrics_enhanced.json`

**Training Parameters:**
- Epochs: 20 (with early stopping)
- Batch size: 32
- Learning rate: 1e-4 with warmup
- Samples per class: 100 (adjust based on dataset)

**Training Time:**
- CPU: 30-60 minutes
- GPU: 10-20 minutes

### Step 2: Test the Model

Verify the model works correctly:

```bash
python test_enhanced_model.py
```

This will:
- Load the enhanced model
- Test prediction on a sample image
- Show predictions with and without TTA
- Display top 3 predictions

### Step 3: Restart AI Service

The AI service automatically uses the enhanced model when available:

```bash
# Stop the service if running
# Then restart
cd ai_service
python app.py
```

The service will:
- Detect the enhanced model at startup
- Load it with TTA support
- Use it for all plant disease predictions
- Fall back to standard model if enhanced model not found

### Step 4: Verify API Response

Upload a plant disease image via the API:

```bash
curl -X POST "http://localhost:8000/api/predictPlant" \
  -F "file=@test_image.jpg"
```

**Response will include:**
- `"source": "enhanced_model"` - indicates enhanced model is being used
- Higher confidence scores due to TTA
- More accurate predictions

## Configuration

### Adjust Training Parameters

Edit `train_enhanced_model.py`:

```python
metrics = train_plant_disease_enhanced(
    data_dir=r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\plant_disease\data",
    output_dir="ai/models/image_classification",
    samples_per_class=100,  # Increase for more data
    epochs=20,              # Increase for longer training
    batch_size=32,          # Decrease if out of memory
    lr=1e-4                 # Learning rate
)
```

### Adjust TTA Transforms

Edit `ai/predict_enhanced.py` in `_get_tta_transforms()` to add/remove augmentations.

### Disable TTA for Faster Inference

If you need faster predictions and can accept slightly lower accuracy:

In `ai_service/app.py`, change:
```python
result = PLANT_ENHANCED_PREDICTOR.predict_single(contents, use_tta=True)
```
to:
```python
result = PLANT_ENHANCED_PREDICTOR.predict_single(contents, use_tta=False)
```

## Expected Accuracy Improvements

Based on similar implementations:
- **Baseline (ResNet9)**: ~85-90% accuracy
- **Enhanced (EfficientNet-B3)**: ~92-95% accuracy
- **With TTA**: ~94-97% accuracy

Actual results depend on:
- Dataset quality and size
- Training duration
- Hardware availability

## Troubleshooting

### Model Not Loading
- Ensure `plant_disease_enhanced.pth` exists in `ai/models/image_classification/`
- Check that dependencies are installed: `pip install -r ai/requirements.txt`

### Out of Memory During Training
- Reduce `batch_size` in training script
- Reduce `samples_per_class`
- Use a smaller model (change `efficientnet_b3` to `efficientnet_b0`)

### Service Falls Back to Standard Model
- Check service logs for enhanced model loading errors
- Verify model file exists and is valid
- Ensure dependencies are installed in service environment

### Slow Inference with TTA
- Disable TTA (see above)
- Reduce number of TTA transforms
- Use GPU for inference

## Performance Tips

### For Maximum Accuracy
1. Train with more epochs (30-50)
2. Use all available data (increase `samples_per_class`)
3. Enable TTA for inference
4. Use GPU for training and inference

### For Faster Inference
1. Disable TTA
2. Use smaller model (EfficientNet-B0)
3. Batch predictions if processing multiple images

### For Better Generalization
1. Add more augmentation types
4. Increase mixup alpha
5. Use more label smoothing

## Next Steps

### Further Improvements
1. **Ensemble Methods**: Combine predictions from multiple models
2. **Vision Transformers**: Add ViT for additional model diversity
3. **Transfer Learning**: Fine-tune on domain-specific data
4. **Active Learning**: Continuously improve with user feedback

### Monitoring
1. Track prediction confidence distribution
2. Log uncertain predictions for review
3. Monitor model drift over time
4. Collect misclassified examples for analysis

## Support

For issues or questions:
1. Check logs in the AI service console
2. Verify model file integrity
3. Ensure all dependencies are installed
4. Test with known-good images

## Summary

This enhanced system provides:
- ✅ State-of-the-art EfficientNet-B3 architecture
- ✅ Advanced data augmentation for better generalization
- ✅ Training improvements (mixup, label smoothing, class weighting)
- ✅ Test-Time Augmentation for 2-5% accuracy boost
- ✅ Automatic integration with existing AI service
- ✅ Fallback to standard model if needed

**Result**: More accurate plant disease detection with better real-world performance.
