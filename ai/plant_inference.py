"""Shared Plant Disease ResNet9 model loading and image preprocessing."""

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ai.models.resnet9 import ResNet9, build_resnet9_from_state_dict


ROOT_DIR = Path(__file__).resolve().parents[1]
PLANT_MODEL_PATH = ROOT_DIR / "ai" / "models" / "image_classification" / "plant_resnet9_best.pt"
PLANT_CLASS_NAMES_PATH = ROOT_DIR / "ai" / "models" / "image_classification" / "class_names.json"
DEFAULT_IMAGE_SIZE = 256
CONFIDENCE_THRESHOLD = 0.60
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


class PlantImageInputError(ValueError):
    """Raised when an uploaded or local image cannot be decoded for inference."""


def save_class_names(classes: Sequence[str], output_dir: Union[str, Path]) -> Path:
    class_names = list(classes)
    if not class_names or any(not isinstance(name, str) or not name for name in class_names):
        raise ValueError("Class names must be a non-empty sequence of non-empty strings")
    if len(class_names) != len(set(class_names)):
        raise ValueError("Class names must be unique")

    path = Path(output_dir) / "class_names.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)
        file.write("\n")
    return path


def load_class_names(
    path: Union[str, Path] = PLANT_CLASS_NAMES_PATH,
    checkpoint_classes: Optional[Sequence[str]] = None,
) -> List[str]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    with path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Expected a non-empty class-name list in {path}")
    if any(not isinstance(name, str) or not name for name in class_names):
        raise ValueError(f"Invalid class name found in {path}")
    if len(class_names) != len(set(class_names)):
        raise ValueError(f"Duplicate class names found in {path}")
    if checkpoint_classes is not None and list(checkpoint_classes) != class_names:
        raise ValueError(
            f"Class order in {path} does not match the model checkpoint"
        )
    return class_names


def plant_display_name(class_id: str) -> str:
    if "___" not in class_id:
        return class_id.replace("_", " ").title()
    crop_name, disease_name = class_id.split("___", 1)
    crop_name = crop_name.replace("_", " ").title()
    disease_name = disease_name.replace("_", " ").title()
    if disease_name.lower() == "healthy":
        return f"{crop_name} Healthy Leaf"
    return f"{crop_name} {disease_name}"


def validate_class_mapping(classes: Sequence[str]) -> List[str]:
    class_names = list(classes)
    if not class_names or any(
        not isinstance(name, str) or not name for name in class_names
    ):
        raise ValueError("The plant class mapping must contain non-empty class identifiers")
    if len(class_names) != len(set(class_names)):
        raise ValueError("The plant class mapping contains duplicate class identifiers")
    display_names = [plant_display_name(class_id) for class_id in class_names]
    if len(display_names) != len(set(display_names)):
        raise ValueError("Plant display-name conversion is not one-to-one")
    return class_names


def inspect_plant_dataset(
    dataset_dir: Union[str, Path],
    classes: Sequence[str],
) -> Dict[str, Any]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        return {
            "available": False,
            "path": str(dataset_path),
            "directory_class_count": None,
            "image_count": None,
            "missing_classes": list(classes),
            "unsupported_directories": [],
        }

    directories = sorted(path.name for path in dataset_path.iterdir() if path.is_dir())
    supported = set(classes)
    class_directories = set(directories)
    image_count = sum(
        1
        for class_name in directories
        for image_path in (dataset_path / class_name).iterdir()
        if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )
    return {
        "available": True,
        "path": str(dataset_path.resolve()),
        "directory_class_count": len(directories),
        "image_count": image_count,
        "missing_classes": sorted(supported - class_directories),
        "unsupported_directories": sorted(class_directories - supported),
    }


def TRANSFORM(image: Image.Image, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Convert a PIL image to normalized RGB CHW float32 tensor."""
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero")

    image = image.convert("RGB").resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(np.ascontiguousarray(pixels))


preprocess_plant_image = TRANSFORM


def _decode_image(image: Union[Image.Image, bytes, bytearray, str, Path]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        if not image:
            raise PlantImageInputError("No image data was provided")
        try:
            with Image.open(io.BytesIO(image)) as source:
                return source.convert("RGB")
        except (OSError, ValueError) as error:
            raise PlantImageInputError("The uploaded file is not a supported image") from error
    if isinstance(image, (str, Path)):
        try:
            with Image.open(image) as source:
                return source.convert("RGB")
        except (OSError, ValueError) as error:
            raise PlantImageInputError("The image file could not be opened") from error
    raise PlantImageInputError(
        f"Unsupported plant image input: {type(image).__name__}"
    )


def decode_and_preprocess_plant_image(
    image_bytes: bytes,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Tuple[Image.Image, torch.Tensor]:
    image = _decode_image(image_bytes)
    return image, TRANSFORM(image, image_size)


def preprocess_plant_image_path(
    image_path: Union[str, Path],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    return TRANSFORM(_decode_image(image_path), image_size)


def load_model(
    weights_path: Union[str, Path] = PLANT_MODEL_PATH,
    classes_path: Union[str, Path] = PLANT_CLASS_NAMES_PATH,
    model_cls: Any = ResNet9,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    checkpoint_path = Path(weights_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT_DIR / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a checkpoint dictionary at {checkpoint_path}")

    checkpoint_classes = checkpoint.get("classes")
    if not isinstance(checkpoint_classes, list):
        raise ValueError(f"Checkpoint has no ordered class list: {checkpoint_path}")
    class_names = load_class_names(classes_path, checkpoint_classes)
    class_names = validate_class_mapping(class_names)
    model_state = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint has no model state: {checkpoint_path}")

    model = build_resnet9_from_state_dict(model_state, len(class_names))
    if not isinstance(model, model_cls):
        raise TypeError(
            f"Checkpoint architecture {type(model).__name__} is not an instance of "
            f"the requested model class {model_cls.__name__}"
        )
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()

    image_size = checkpoint.get("img_size", checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    if isinstance(image_size, (list, tuple)):
        if len(image_size) != 2 or image_size[0] != image_size[1]:
            raise ValueError(f"Only square plant model inputs are supported: {image_size}")
        image_size = image_size[0]
    image_size = int(image_size)
    if image_size <= 0:
        raise ValueError(f"Invalid checkpoint image size: {image_size}")
    with torch.no_grad():
        output = model(torch.zeros(1, 3, image_size, image_size, device=device))
    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] != len(class_names):
        raise ValueError(
            f"Model output shape {tuple(output.shape)} does not match "
            f"one sample and {len(class_names)} class names"
        )
    display_names = [plant_display_name(class_name) for class_name in class_names]

    return {
        "model": model,
        "classes": class_names,
        "display_names": display_names,
        "class_to_idx": {name: index for index, name in enumerate(class_names)},
        "output_count": int(output.shape[1]),
        "architecture": type(model).__name__,
        "training_class_count": len(checkpoint_classes),
        "class_mapping_source": "checkpoint classes validated against class_names.json",
        "model_state": model_state,
        "img_size": image_size,
        "_resolved_path": str(checkpoint_path.resolve()),
    }


def load_plant_model(
    checkpoint_path: Union[str, Path] = PLANT_MODEL_PATH,
    class_names_path: Union[str, Path] = PLANT_CLASS_NAMES_PATH,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    return load_model(
        weights_path=checkpoint_path,
        classes_path=class_names_path,
        model_cls=ResNet9,
        device=device,
    )


def predict_logits(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
    if logits.ndim != 2:
        raise ValueError(f"Expected [batch, classes] logits, got shape {tuple(logits.shape)}")
    return logits


def _predict(
    model: torch.nn.Module,
    classes: Sequence[str],
    image: Union[Image.Image, bytes, bytearray, str, Path],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Dict[str, Any]:
    classes = validate_class_mapping(classes)
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    tensor = TRANSFORM(_decode_image(image), image_size).unsqueeze(0).to(device)
    logits = predict_logits(model, tensor)
    if logits.shape[1] != len(classes):
        raise ValueError(
            f"Model output count ({logits.shape[1]}) does not match class list ({len(classes)})"
        )
    probabilities = torch.softmax(logits, dim=1)[0]
    index = int(torch.argmax(probabilities).item())
    result: Dict[str, Any] = {
        "class_id": classes[index],
        "class_index": index,
        "label": plant_display_name(classes[index]),
        "index": index,
        "confidence": float(probabilities[index].item()),
        "uncertain": float(probabilities[index].item()) < CONFIDENCE_THRESHOLD,
        "tensor_shape": tuple(tensor.shape),
        "probabilities": probabilities.tolist(),
    }
    top_k = min(3, len(classes))
    top_indices = torch.topk(probabilities, k=top_k).indices.tolist()
    result["top_k"] = [
        {
            "class_id": classes[top_index],
            "class_index": top_index,
            "label": plant_display_name(classes[top_index]),
            "confidence": float(probabilities[top_index].item()),
        }
        for top_index in top_indices
    ]
    return result


def predict(
    model: torch.nn.Module,
    classes: Sequence[str],
    image: Union[Image.Image, bytes, bytearray, str, Path],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Dict[str, Any]:
    return _predict(model, classes, image, image_size)


def predict_with_top3(
    model: torch.nn.Module,
    classes: Sequence[str],
    image: Union[Image.Image, bytes, bytearray, str, Path],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    result = _predict(model, classes, image, image_size)
    return result, result["top_k"]
