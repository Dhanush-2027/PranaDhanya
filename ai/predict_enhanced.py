"""Enhanced inference with Test-Time Augmentation (TTA) for improved accuracy."""

import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Dict, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add workspace to sys.path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in os.path.dirname(os.path.abspath(__file__)):
    import sys
    sys.path.insert(0, WORKSPACE_DIR)

from ai.models.efficientnet_model import build_efficientnet_model


class PlantDiseasePredictor:
    """Enhanced predictor with TTA support."""
    
    def __init__(self, model_path: str, device=None):
        """
        Load model and setup TTA transforms.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract classes and model state
        self.classes = checkpoint.get('classes', [])
        self.num_classes = len(self.classes)
        
        # Build model
        model_name = checkpoint.get('config', {}).get('model_name', 'efficientnet_b3')
        self.model = build_efficientnet_model(
            num_classes=self.num_classes,
            model_name=model_name,
            pretrained=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup TTA transforms
        self.img_size = checkpoint.get('config', {}).get('img_size', 224)
        self.tta_transforms = self._get_tta_transforms()
        self.base_transform = self._get_base_transform()
        
        print(f"Model loaded: {model_name}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Device: {self.device}")
    
    def _get_base_transform(self):
        """Base transform without augmentation."""
        return A.Compose([
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _get_tta_transforms(self):
        """Test-Time Augmentation transforms."""
        transforms = [
            # Original
            A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Horizontal flip
            A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Vertical flip
            A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Both flips
            A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Small rotation +5
            A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Rotate(limit=5, border_mode=0, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Small rotation -5
            A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Rotate(limit=-5, border_mode=0, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
        ]
        return transforms
    
    def _load_image(self, image_input: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        """Load image from various input types."""
        if isinstance(image_input, (str, Path)):
            image = np.array(Image.open(image_input).convert('RGB'))
        elif isinstance(image_input, bytes):
            import io
            image = np.array(Image.open(io.BytesIO(image_input)).convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        return image
    
    def predict_single(self, image_input: Union[str, Path, bytes, np.ndarray], use_tta: bool = True) -> Dict:
        """
        Predict disease for a single image.
        
        Args:
            image_input: Image path, bytes, or numpy array
            use_tta: Whether to use test-time augmentation
        
        Returns:
            Dictionary with prediction results
        """
        image = self._load_image(image_input)
        
        if use_tta:
            return self._predict_with_tta(image)
        else:
            return self._predict_single_transform(image, self.base_transform)
    
    def _predict_single_transform(self, image: np.ndarray, transform) -> Dict:
        """Predict with a single transform."""
        augmented = transform(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        pred_idx = int(np.argmax(probs))
        predicted_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx] * 100.0)
        
        distribution = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        top_predictions = sorted(
            [{"class": self.classes[i], "probability": float(probs[i])} for i in range(len(self.classes))],
            key=lambda x: x["probability"],
            reverse=True
        )[:5]  # Top 5
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probability_distribution": distribution,
            "top_predictions": top_predictions
        }
    
    def _predict_with_tta(self, image: np.ndarray) -> Dict:
        """Predict with test-time augmentation (average over multiple augmentations)."""
        all_probs = []
        
        for transform in self.tta_transforms:
            result = self._predict_single_transform(image, transform)
            probs = result['probability_distribution']
            all_probs.append([probs[cls] for cls in self.classes])
        
        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        
        pred_idx = int(np.argmax(avg_probs))
        predicted_class = self.classes[pred_idx]
        confidence = float(avg_probs[pred_idx] * 100.0)
        
        distribution = {self.classes[i]: float(avg_probs[i]) for i in range(len(self.classes))}
        top_predictions = sorted(
            [{"class": self.classes[i], "probability": float(avg_probs[i])} for i in range(len(self.classes))],
            key=lambda x: x["probability"],
            reverse=True
        )[:5]
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probability_distribution": distribution,
            "top_predictions": top_predictions,
            "tta_used": True,
            "num_tta_transforms": len(self.tta_transforms)
        }
    
    def predict_batch(self, image_inputs: List[Union[str, Path, bytes]], use_tta: bool = True) -> List[Dict]:
        """
        Predict multiple images.
        
        Args:
            image_inputs: List of image paths or bytes
            use_tta: Whether to use test-time augmentation
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_input in image_inputs:
            result = self.predict_single(img_input, use_tta=use_tta)
            results.append(result)
        return results


def load_enhanced_predictor(model_path: str, device=None) -> PlantDiseasePredictor:
    """
    Factory function to load enhanced predictor.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to use (cuda/cpu)
    
    Returns:
        PlantDiseasePredictor instance
    """
    return PlantDiseasePredictor(model_path, device)


if __name__ == '__main__':
    # Test the predictor
    model_path = "ai/models/image_classification/plant_disease_enhanced.pth"
    
    if os.path.exists(model_path):
        predictor = load_enhanced_predictor(model_path)
        print("Predictor loaded successfully!")
        
        # Test with a dummy prediction (will fail without actual image)
        # image_path = "path/to/test/image.jpg"
        # result = predictor.predict_single(image_path, use_tta=True)
        # print(f"Prediction: {result}")
    else:
        print(f"Model not found at {model_path}")
        print("Please train the model first using train_plant_enhanced.py")
