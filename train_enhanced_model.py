"""Quick script to train the enhanced plant disease model."""

import sys
import os

# Add workspace to path
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from ai.training.train_plant_enhanced import train_plant_disease_enhanced

if __name__ == '__main__':
    print("=" * 80)
    print("ENHANCED PLANT DISEASE MODEL TRAINING")
    print("=" * 80)
    print("\nThis script will train an EfficientNet-B3 model with:")
    print("  - Advanced Albumentations augmentation")
    print("  - Mixup training for better generalization")
    print("  - Label smoothing + class weighting")
    print("  - Warmup + cosine annealing learning rate schedule")
    print("  - Early stopping and checkpointing")
    print("\nTraining may take 30-60 minutes depending on your hardware.")
    print("=" * 80)
    
    # Run training
    metrics = train_plant_disease_enhanced(
        data_dir=r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\plant_disease\data",
        output_dir="ai/models/image_classification",
        samples_per_class=100,  # Adjust based on your dataset size
        epochs=20,
        batch_size=32,
        lr=1e-4
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Final Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Final Test F1-Score: {metrics['f1']*100:.2f}%")
    print("\nThe enhanced model is now saved to: ai/models/image_classification/plant_disease_enhanced.pth")
    print("Restart the AI service to use the new model with TTA.")
