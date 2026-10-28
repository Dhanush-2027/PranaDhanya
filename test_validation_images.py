import asyncio
import io
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi import UploadFile
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from ai.plant_inference import (
    PLANT_MODEL_PATH,
    inspect_plant_dataset,
    load_plant_model,
    predict,
)


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "datasets" / "plant_disease" / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def select_sample_images(class_names):
    samples = []
    for class_name in class_names:
        class_dir = DATA_DIR / class_name
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_paths:
            samples.append((image_paths[0], class_name))
    return samples


def main():
    if not PLANT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Plant checkpoint not found: {PLANT_MODEL_PATH}")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = load_plant_model(device="cpu")
    classes = checkpoint["classes"]
    dataset_info = inspect_plant_dataset(DATA_DIR, classes)
    if dataset_info["missing_classes"]:
        raise ValueError(f"Dataset is missing supported classes: {dataset_info['missing_classes']}")

    samples = select_sample_images(classes)
    if not samples:
        raise FileNotFoundError(f"No sample images found under {DATA_DIR}")

    print(f"Checkpoint: {checkpoint['_resolved_path']}")
    print(f"Architecture: {checkpoint['architecture']}")
    print(f"Checkpoint output neurons: {checkpoint['output_count']}")
    print(f"Training classes recovered from checkpoint: {checkpoint['training_class_count']}")
    print(f"Class mapping: {checkpoint['class_mapping_source']}")
    print(f"Dataset path: {dataset_info['path']}")
    print(f"Dataset directories: {dataset_info['directory_class_count']}")
    print(f"Dataset image files: {dataset_info['image_count']}")
    print(f"Supported classes: {len(classes)}")
    print(f"Missing classes: {dataset_info['missing_classes']}")
    print(f"Unsupported directories: {dataset_info['unsupported_directories']}")
    print(
        "Evaluation protocol: first available sorted image per supported class; "
        "diagnostic sample, not a held-out validation split."
    )
    print(f"Images evaluated: {len(samples)}")

    targets = []
    predicted_indices = []
    invalid_predictions = 0
    api_agreements = 0
    api_labels = []
    model = checkpoint["model"]

    with patch("ai_service.app.PLANT_CHECKPOINT", checkpoint):
        from ai_service.app import predict_plant

        for sample_index, (image_path, expected_label) in enumerate(samples, start=1):
            image_bytes = image_path.read_bytes()
            direct_prediction = predict(
                model,
                classes,
                image_bytes,
                image_size=checkpoint["img_size"],
            )
            predicted_index = int(direct_prediction["class_index"])
            predicted_id = str(direct_prediction["class_id"])
            if (
                predicted_index < 0
                or predicted_index >= len(classes)
                or classes[predicted_index] != predicted_id
            ):
                invalid_predictions += 1
                continue

            upload = UploadFile(
                filename=image_path.name,
                file=io.BytesIO(image_bytes),
            )
            api_response = asyncio.run(predict_plant(upload))
            api_match = (
                api_response["classIndex"] == predicted_index
                and api_response["classId"] == predicted_id
                and api_response["diseaseName"] == direct_prediction["label"]
            )
            api_agreements += api_match
            api_labels.append(api_response["diseaseName"])
            targets.append(checkpoint["class_to_idx"][expected_label])
            predicted_indices.append(predicted_index)
            print(
                f"{sample_index:02d}/{len(samples)} {image_path.name}: "
                f"true={expected_label} predicted={predicted_id} "
                f"confidence={direct_prediction['confidence']:.4f} "
                f"API={'match' if api_match else 'mismatch'}"
            )

    evaluated = len(targets)
    correct = sum(target == predicted for target, predicted in zip(targets, predicted_indices))
    match_rate = api_agreements / evaluated if evaluated else 0.0
    accuracy = correct / evaluated if evaluated else 0.0
    print(f"\nNumber of images evaluated: {evaluated}")
    print(f"Number of valid ground-truth labels: {len(targets)}")
    print(f"Number of supported classes: {len(classes)}")
    print(f"Top-1 accuracy against folder labels: {accuracy:.4%} ({correct}/{evaluated})")
    print(
        "API-versus-direct prediction agreement: "
        f"{match_rate:.4%} ({api_agreements}/{evaluated})"
    )
    print(f"Invalid or unmapped predictions: {invalid_predictions}")
    print(f"Distinct API display labels: {len(set(api_labels))}")
    print("Per-class precision, recall, and F1:")
    print(
        classification_report(
            targets,
            predicted_indices,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0,
            digits=4,
        )
    )
    print("Confusion matrix (rows=true class index, columns=predicted class index):")
    print(confusion_matrix(targets, predicted_indices, labels=list(range(len(classes)))))

    metrics_path = ROOT_DIR / "ai" / "models" / "image_classification" / "plant_metrics.json"
    if metrics_path.exists():
        import json

        saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(
            "Stored checkpoint metrics (not recomputed in this run): "
            f"validation_accuracy={saved_metrics.get('val_accuracy')}, "
            f"test_accuracy={saved_metrics.get('test_accuracy')}, "
            f"num_classes={saved_metrics.get('num_classes')}"
        )


if __name__ == "__main__":
    main()
