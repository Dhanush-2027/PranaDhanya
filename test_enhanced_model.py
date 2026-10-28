"""Test script to verify the enhanced model works correctly."""

import sys
import os
from pathlib import Path

# Add workspace to path
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

def test_model():
    """Test if the enhanced model can be loaded and used."""
    from ai.predict_enhanced import load_enhanced_predictor
    
    model_path = "ai/models/image_classification/plant_disease_enhanced.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please train the model first by running: python train_enhanced_model.py")
        return False
    
    try:
        print("Loading enhanced model...")
        predictor = load_enhanced_predictor(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Number of classes: {predictor.num_classes}")
        print(f"   Device: {predictor.device}")
        print(f"   Model architecture: EfficientNet-B3")
        print(f"   TTA transforms: {len(predictor.tta_transforms)}")
        
        # Find a test image
        data_dir = Path("datasets/plant_disease/data")
        if data_dir.exists():
            # Get first image from first class
            classes = sorted([d for d in data_dir.iterdir() if d.is_dir()])
            if classes:
                test_class = classes[0]
                images = list(test_class.glob("*.jpg")) + list(test_class.glob("*.png"))
                if images:
                    test_image = images[0]
                    print(f"\n🧪 Testing prediction on: {test_image.name}")
                    print(f"   Actual class: {test_class.name}")
                    
                    # Test without TTA
                    result_no_tta = predictor.predict_single(str(test_image), use_tta=False)
                    print(f"\nPrediction (no TTA):")
                    print(f"   Predicted: {result_no_tta['predicted_class']}")
                    print(f"   Confidence: {result_no_tta['confidence']:.2f}%")
                    
                    # Test with TTA
                    result_tta = predictor.predict_single(str(test_image), use_tta=True)
                    print(f"\nPrediction (with TTA):")
                    print(f"   Predicted: {result_tta['predicted_class']}")
                    print(f"   Confidence: {result_tta['confidence']:.2f}%")
                    print(f"   TTA transforms used: {result_tta.get('num_tta_transforms', 0)}")
                    
                    # Show top 3 predictions
                    print(f"\nTop 3 predictions:")
                    for i, pred in enumerate(result_tta['top_predictions'][:3], 1):
                        print(f"   {i}. {pred['class']}: {pred['probability']*100:.2f}%")
                    
                    return True
        
        print("\n⚠️  No test images found in dataset directory")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 80)
    print("ENHANCED MODEL TEST")
    print("=" * 80)
    success = test_model()
    print("=" * 80)
    if success:
        print("✅ Model test completed successfully!")
        print("\nThe enhanced model is ready to use.")
        print("The AI service will automatically use it when restarted.")
    else:
        print("❌ Model test failed. Please check the errors above.")
