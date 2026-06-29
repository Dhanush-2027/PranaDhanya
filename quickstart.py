#!/usr/bin/env python
"""
Quick Start: Install dependencies and train all models
Run this script once to set up and train everything
"""

import subprocess
import sys
import os
from pathlib import Path

def run_cmd(cmd, description):
    """Run command with error handling"""
    print(f"\n{'='*70}")
    print(f"► {description}")
    print(f"{'='*70}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         SMART AGRI PORTAL - QUICK START & TRAINING SETUP            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get dataset path
    default_dataset = r"C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets"
    dataset_path = input(f"Enter dataset path (default: {default_dataset}): ").strip()
    if not dataset_path:
        dataset_path = default_dataset
    
    dataset_path = os.path.normpath(os.path.abspath(dataset_path))
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    print(f"✓ Using dataset path: {dataset_path}")
    
    # Step 1: Install dependencies
    print("\n" + "="*70)
    print("STEP 1: Installing Dependencies")
    print("="*70)
    
    deps_cmd = f'{sys.executable} -m pip install --upgrade pip'
    run_cmd(deps_cmd, "Upgrading pip")
    
    # Install AI module deps
    ai_deps_cmd = f'{sys.executable} -m pip install -r ai/requirements.txt'
    if not run_cmd(ai_deps_cmd, "Installing AI module dependencies"):
        print("✗ Failed to install AI dependencies")
        sys.exit(1)
    
    # Install AI service deps
    service_deps_cmd = f'{sys.executable} -m pip install -r ai_service/requirements.txt'
    if not run_cmd(service_deps_cmd, "Installing AI service dependencies"):
        print("✗ Failed to install service dependencies")
        sys.exit(1)
    
    # Step 2: Verify datasets
    print("\n" + "="*70)
    print("STEP 2: Verifying Datasets")
    print("="*70)
    
    required_datasets = {
        'crop_recommendation/Crop_recommendation.csv': 'Crop Recommendation',
        'yield_prediction/data.csv': 'Yield Prediction',
        'price_prediction/data.csv': 'Price Prediction',
        'fertilizer_prediction/Fertilizer Prediction.csv': 'Fertilizer Recommendation',
        'plant_disease/data': 'Plant Disease Images',
        'dog_skin_disease/train': 'Animal Disease Images',
    }
    
    all_found = True
    for rel_path, description in required_datasets.items():
        full_path = os.path.join(dataset_path, rel_path)
        if os.path.exists(full_path):
            print(f"✓ {description:35} | {rel_path}")
        else:
            print(f"✗ {description:35} | NOT FOUND: {rel_path}")
            all_found = False
    
    if not all_found:
        print("\n⚠ Some datasets are missing. Training may fail for missing datasets.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    # Step 3: Train models
    print("\n" + "="*70)
    print("STEP 3: Training All Models")
    print("="*70)
    
    print("\nAvailable training options:")
    print("  1. Train all models")
    print("  2. Train only tabular models (crop, yield, price, fertilizer)")
    print("  3. Train only image models (plant disease, animal disease)")
    print("  4. Custom selection")
    
    choice = input("\nSelect option (1-4, default 1): ").strip() or "1"
    
    if choice == "1":
        # Train all
        train_cmd = f'{sys.executable} train_all_models.py --datasets "{dataset_path}"'
    elif choice == "2":
        # Train tabular only
        train_cmd = f'{sys.executable} train_all_models.py --datasets "{dataset_path}" --skip-image'
    elif choice == "3":
        # Train image only
        train_cmd = f'{sys.executable} train_all_models.py --datasets "{dataset_path}" --skip-tabular'
    elif choice == "4":
        # Custom
        print("\nSkip which models?")
        skip_flags = ""
        if input("Skip crop recommendation? (y/n): ").strip().lower() == 'y':
            skip_flags += " --skip-crop"
        if input("Skip yield prediction? (y/n): ").strip().lower() == 'y':
            skip_flags += " --skip-yield"
        if input("Skip price prediction? (y/n): ").strip().lower() == 'y':
            skip_flags += " --skip-price"
        if input("Skip fertilizer recommendation? (y/n): ").strip().lower() == 'y':
            skip_flags += " --skip-fertilizer"
        if input("Skip plant disease detection? (y/n): ").strip().lower() == 'y':
            skip_flags += " --skip-plant"
        if input("Skip animal disease detection? (y/n): ").strip().lower() == 'y':
            skip_flags += " --skip-animal"
        train_cmd = f'{sys.executable} train_all_models.py --datasets "{dataset_path}"{skip_flags}'
    else:
        print("Invalid choice")
        sys.exit(1)
    
    if not run_cmd(train_cmd, "Training models"):
        print("\n✗ Model training failed")
        sys.exit(1)
    
    # Step 4: Success
    print("\n" + "="*70)
    print("✓ SETUP AND TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("""
Next steps:

1. Start the AI service:
   cd ai_service
   python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

2. Test the API:
   Open http://localhost:8000/docs in your browser

3. To run the full stack with Docker:
   docker-compose up --build

4. See TRAINING_GUIDE.md for detailed documentation
    """)

if __name__ == '__main__':
    main()
