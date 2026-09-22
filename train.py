"""
train.py
========
Unified Training CLI for Plant & Animal Disease ResNet9 CNN Models
Usage:
    python train.py --task plant
    python train.py --task animal
    python train.py --task all
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from train_plant_model import train_plant_model
from train_animal_model import train_animal_model


def main():
    parser = argparse.ArgumentParser(description="Unified ResNet9 CNN Training CLI")
    parser.add_argument("--task", type=str, required=True, choices=["plant", "animal", "all"],
                        help="Which model to train: plant, animal, or all")
    parser.add_argument("--plant-dataset", type=str,
                        default=r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\plant_disease\data",
                        help="Path to plant disease dataset data folder")
    parser.add_argument("--cattle-path", type=str,
                        default=r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\cattle_diseases",
                        help="Path to cattle diseases dataset")
    parser.add_argument("--dog-path", type=str,
                        default=r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\dog_skin_disease",
                        help="Path to dog skin disease dataset")
    parser.add_argument("--goat-path", type=str,
                        default=r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\livestock",
                        help="Path to goat/livestock dataset")
    parser.add_argument("--output-dir", type=str, default="ai/models/image_classification",
                        help="Directory to save checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--img-size", type=int, default=84, help="Image input resolution (e.g. 84, 112, 224)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for optimizer/scheduler")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (epochs)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Execution device: auto (detect GPU), cuda (require GPU), or cpu")
    parser.add_argument("--allow-cpu", action="store_true", default=True,
                        help="Allow CPU execution if CUDA GPU is not available")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f" RESNET9 CNN MODEL TRAINING PIPELINE - TASK: {args.task.upper()}")
    print("="*70)
    
    results = {}
    
    if args.task in ["plant", "all"]:
        print("\n>>> STARTING PLANT DISEASE DETECTION MODEL TRAINING <<<\n")
        plant_res = train_plant_model(
            dataset_dir=args.plant_dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            patience=args.patience,
            device_mode=args.device,
            allow_cpu=args.allow_cpu,
            num_workers=args.num_workers,
            seed=args.seed
        )
        results["plant"] = plant_res
        
    if args.task in ["animal", "all"]:
        print("\n>>> STARTING ANIMAL DISEASE DETECTION MODEL TRAINING <<<\n")
        animal_res = train_animal_model(
            cattle_path=args.cattle_path,
            dog_path=args.dog_path,
            goat_path=args.goat_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            patience=args.patience,
            device_mode=args.device,
            allow_cpu=args.allow_cpu,
            num_workers=args.num_workers,
            seed=args.seed
        )
        results["animal"] = animal_res

    print("\n" + "="*70)
    print(" TRAINING PIPELINE SUMMARY")
    print("="*70)
    for task_name, res in results.items():
        print(f"[{task_name.upper()} MODEL]")
        print(f"  - Classes: {res['num_classes']}")
        print(f"  - Best Validation Accuracy: {res['best_val_accuracy']*100:.2f}% (Epoch {res['best_epoch']})")
        print(f"  - Epochs Run: {res['epochs_trained']}")
        print(f"  - Training Time: {res['training_time_seconds']/60:.2f} mins")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
