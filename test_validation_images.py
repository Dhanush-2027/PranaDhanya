from pathlib import Path

import torch

from ai.plant_inference import (
    PLANT_MODEL_PATH,
    load_plant_model,
    predict_logits,
    preprocess_plant_image_path,
)


DATA_DIR = Path("datasets/plant_disease/data")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def select_sample_images(class_names, sample_count=10):
    indices = torch.linspace(0, len(class_names) - 1, steps=min(sample_count, len(class_names)))
    samples = []
    for index in indices.round().to(torch.int64).tolist():
        class_name = class_names[index]
        class_dir = DATA_DIR / class_name
        image_paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_paths:
            samples.append((image_paths[0], class_name))
    return samples


def main():
    if not PLANT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Plant checkpoint not found: {PLANT_MODEL_PATH}")

    checkpoint = load_plant_model()
    model = checkpoint["model"]
    classes = checkpoint["classes"]
    image_size = checkpoint["img_size"]
    samples = select_sample_images(classes)
    if not samples:
        raise FileNotFoundError(f"No sample images found under {DATA_DIR}")

    print(f"Checkpoint: {checkpoint['_resolved_path']}")
    print(f"Classes: {len(classes)} | Image size: {image_size}x{image_size}")
    correct = 0

    for index, (image_path, expected_label) in enumerate(samples, start=1):
        tensor = preprocess_plant_image_path(image_path, image_size).unsqueeze(0)
        logits = predict_logits(model, tensor)
        top_values, top_indices = torch.topk(logits[0], k=min(3, len(classes)))
        predicted_label = classes[int(top_indices[0])]
        correct += predicted_label == expected_label

        print(f"\nSample {index}: {image_path}")
        print(
            "Tensor: "
            f"shape={tuple(tensor.shape)}, mean={tensor.mean().item():.6f}, "
            f"std={tensor.std(unbiased=False).item():.6f}, "
            f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}"
        )
        print(f"Ground truth: {expected_label}")
        print(f"Prediction:   {predicted_label}")
        print("Top 3 logits:")
        for value, class_index in zip(top_values.tolist(), top_indices.tolist()):
            print(f"  {classes[class_index]}: {value:.6f}")

    print(f"\nDirect folder-label accuracy: {correct / len(samples):.1%} ({correct}/{len(samples)})")


if __name__ == "__main__":
    main()
