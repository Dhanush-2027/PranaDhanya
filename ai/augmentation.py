"""Enhanced data augmentation pipeline using Albumentations for plant disease detection."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transforms(img_size=224):
    """
    Advanced training augmentation pipeline for plant disease images.
    Includes geometric and color augmentations simulating real-world conditions.
    """
    return A.Compose([
        # Geometric transformations
        A.Resize(height=img_size, width=img_size),
        A.RandomCrop(height=img_size, width=img_size, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
        
        # Color and lighting augmentations (important for disease detection)
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        
        # Blur and noise to handle image quality variations
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Normalize
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=224):
    """
    Validation/test augmentation - minimal preprocessing.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size=224):
    """
    Test-Time Augmentation transforms for improved inference accuracy.
    Returns multiple augmentation variants to average predictions.
    """
    base_transforms = [
        # Original
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Horizontal flip
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Vertical flip
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Both flips
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Small rotation +5
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Rotate(limit=5, border_mode=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Small rotation -5
        A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Rotate(limit=-5, border_mode=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
    return base_transforms
