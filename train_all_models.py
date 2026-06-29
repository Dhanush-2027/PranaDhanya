#!/usr/bin/env python
"""
Master training script for all models in the Smart Agri Portal
Trains:
1. Crop Recommendation - XGBoost Classifier
2. Yield Prediction - Random Forest Regressor
3. Price Prediction - Random Forest Regressor
4. Fertilizer Recommendation - XGBoost Classifier
5. Plant Disease Detection - ResNet9 CNN
6. Animal Disease Detection - ResNet9 CNN
"""

import os
import sys

# Reconfigure stdout/stderr to use UTF-8 on Windows to avoid encoding issues
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import argparse
import subprocess
from pathlib import Path

# Add ai module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai'))

def run_command(cmd, description):
    """Execute command and report status"""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    try:
        # Add current directory to PYTHONPATH so subprocesses can find 'ai' module
        env = os.environ.copy()
        root_dir = os.path.dirname(os.path.abspath(__file__))
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = root_dir + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = root_dir
        
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        result = subprocess.run(cmd, shell=True, check=True, env=env)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False

def train_crop_recommendation(dataset_path):
    """Train XGBoost Classifier for crop recommendation"""
    csv_path = os.path.join(dataset_path, 'crop_recommendation', 'Crop_recommendation.csv')
    if not os.path.exists(csv_path):
        print(f"✗ Crop recommendation CSV not found at {csv_path}")
        return False
    
    cmd = f'python ai/training/train_crop_recommendation.py --input "{csv_path}" --target label --out-dir ai/models/crop_recommendation'
    return run_command(cmd, "Training Crop Recommendation (XGBoost)")

def train_yield_prediction(dataset_path):
    """Train Random Forest Regressor for yield prediction"""
    csv_path = os.path.join(dataset_path, 'yield_prediction', 'data.csv')
    if not os.path.exists(csv_path):
        print(f"✗ Yield prediction CSV not found at {csv_path}")
        return False
    
    cmd = f'python ai/training/train_yield_prediction.py --input "{csv_path}" --target yield --out-dir ai/models/yield_prediction'
    return run_command(cmd, "Training Yield Prediction (Random Forest)")

def train_price_prediction(dataset_path):
    """Train Random Forest Regressor for price prediction"""
    csv_path = os.path.join(dataset_path, 'price_prediction', 'data.csv')
    if not os.path.exists(csv_path):
        print(f"✗ Price prediction CSV not found at {csv_path}")
        return False
    
    cmd = f'python ai/training/train_price_prediction.py --input "{csv_path}" --target price --out-dir ai/models/price_prediction'
    return run_command(cmd, "Training Price Prediction (Random Forest)")

def train_fertilizer_recommendation(dataset_path):
    """Train XGBoost Classifier for fertilizer recommendation"""
    csv_path = os.path.join(dataset_path, 'fertilizer_prediction', 'Fertilizer Prediction.csv')
    if not os.path.exists(csv_path):
        print(f"✗ Fertilizer recommendation CSV not found at {csv_path}")
        return False
    
    cmd = f'python ai/training/train_fertilizer_recommendation.py --input "{csv_path}" --target fertilizer_name --out-dir ai/models/fertilizer_recommendation'
    return run_command(cmd, "Training Fertilizer Recommendation (XGBoost)")

def train_plant_disease_detection(dataset_path):
    """Train ResNet9 CNN for plant disease detection"""
    data_dir = os.path.join(dataset_path, 'plant_disease', 'data')
    if not os.path.exists(data_dir):
        print(f"✗ Plant disease data directory not found at {data_dir}")
        return False
    
    cmd = f'python ai/training/train_image_classifier.py --data-dir "{data_dir}" --out-dir ai/models/image_classification --name plant_resnet9 --epochs 10 --batch-size 32'
    return run_command(cmd, "Training Plant Disease Detection (ResNet9 CNN)")

def train_animal_disease_detection(dataset_path):
    """Train ResNet9 CNN for animal disease detection (dog skin disease as primary)"""
    # Using dog_skin_disease as the main animal disease dataset
    data_dir = os.path.join(dataset_path, 'dog_skin_disease', 'train')
    if not os.path.exists(data_dir):
        print(f"✗ Dog skin disease data directory not found at {data_dir}")
        return False
    
    cmd = f'python ai/training/train_image_classifier.py --data-dir "{data_dir}" --out-dir ai/models/image_classification --name animal_resnet9 --epochs 10 --batch-size 32'
    return run_command(cmd, "Training Animal Disease Detection (ResNet9 CNN)")

def verify_models_trained(models_dir='ai/models'):
    """Verify all models were trained successfully"""
    print(f"\n{'='*70}")
    print("▶ Verifying trained models")
    print(f"{'='*70}")
    
    expected_files = {
        'crop_recommendation/crop_recommender.pkl': 'Crop Recommendation Model',
        'yield_prediction/yield_predictor.pkl': 'Yield Prediction Model',
        'price_prediction/price_predictor.pkl': 'Price Prediction Model',
        'fertilizer_recommendation/fertilizer_recommender.pkl': 'Fertilizer Recommendation Model',
        'image_classification/plant_resnet9.pt': 'Plant Disease Detection Model',
        'image_classification/animal_resnet9.pt': 'Animal Disease Detection Model',
    }
    
    all_present = True
    for file_path, description in expected_files.items():
        full_path = os.path.join(models_dir, file_path)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"✓ {description:45} | Size: {size_mb:.2f} MB")
        else:
            print(f"✗ {description:45} | NOT FOUND")
            all_present = False
    
    return all_present

def main():
    parser = argparse.ArgumentParser(
        description='Train all models for Smart Agri Portal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_all_models.py --datasets "C:\\Users\\Dhanush\\OneDrive\\Desktop\\C&L\\datasets"
  python train_all_models.py --datasets ./datasets --skip-plant --skip-animal
        """
    )
    
    parser.add_argument('--datasets', type=str, required=True, 
                        help='Path to datasets directory')
    parser.add_argument('--skip-tabular', action='store_true',
                        help='Skip training tabular models (crop, yield, price, fertilizer)')
    parser.add_argument('--skip-image', action='store_true',
                        help='Skip training image models')
    parser.add_argument('--skip-crop', action='store_true',
                        help='Skip crop recommendation model')
    parser.add_argument('--skip-yield', action='store_true',
                        help='Skip yield prediction model')
    parser.add_argument('--skip-price', action='store_true',
                        help='Skip price prediction model')
    parser.add_argument('--skip-fertilizer', action='store_true',
                        help='Skip fertilizer recommendation model')
    parser.add_argument('--skip-plant', action='store_true',
                        help='Skip plant disease detection model')
    parser.add_argument('--skip-animal', action='store_true',
                        help='Skip animal disease detection model')
    
    args = parser.parse_args()
    
    # Normalize dataset path
    dataset_path = os.path.normpath(os.path.abspath(args.datasets))
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print(" SMART AGRI PORTAL - MODEL TRAINING PIPELINE")
    print("="*70)
    print(f"Dataset Path: {dataset_path}")
    print("="*70)
    
    results = {}
    
    # Tabular models
    if not args.skip_tabular:
        if not args.skip_crop:
            results['crop_recommendation'] = train_crop_recommendation(dataset_path)
        
        if not args.skip_yield:
            results['yield_prediction'] = train_yield_prediction(dataset_path)
        
        if not args.skip_price:
            results['price_prediction'] = train_price_prediction(dataset_path)
        
        if not args.skip_fertilizer:
            results['fertilizer_recommendation'] = train_fertilizer_recommendation(dataset_path)
    
    # Image models
    if not args.skip_image:
        if not args.skip_plant:
            results['plant_disease'] = train_plant_disease_detection(dataset_path)
        
        if not args.skip_animal:
            results['animal_disease'] = train_animal_disease_detection(dataset_path)
    
    # Verify all models
    all_trained = verify_models_trained()
    
    # Summary
    print(f"\n{'='*70}")
    print(" TRAINING SUMMARY")
    print(f"{'='*70}")
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    for model_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status:10} | {model_name}")
    
    print(f"{'='*70}")
    print(f"Results: {successful}/{total} models trained successfully")
    
    if all_trained and successful == total:
        print("\n✓ All models trained and verified successfully!")
        print("\nModels are ready for integration with the AI service.")
        print("The FastAPI service will auto-load these models from ai/models/")
        return 0
    else:
        print("\n✗ Some models failed training or verification")
        return 1

if __name__ == '__main__':
    sys.exit(main())
