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
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


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
        with Image.open(io.BytesIO(image)) as source:
            return source.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return source.convert("RGB")
    raise TypeError(f"Unsupported plant image input: {type(image).__name__}")


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

    return {
        "model": model,
        "classes": class_names,
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
) -> Tuple[Dict[str, Union[str, int, float]], List[Dict[str, Union[str, float]]]]:
    if not classes:
        raise ValueError("The plant class list must not be empty")
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
    result: Dict[str, Union[str, int, float]] = {
        "label": classes[index],
        "index": index,
        "confidence": float(probabilities[index].item()),
    }
    top_k = min(3, len(classes))
    top_indices = torch.topk(probabilities, k=top_k).indices.tolist()
    top_predictions = [
        {
            "class": classes[top_index],
            "confidence": float(probabilities[top_index].item()),
        }
        for top_index in top_indices
    ]
    return result, top_predictions


def predict(
    model: torch.nn.Module,
    classes: Sequence[str],
    image: Union[Image.Image, bytes, bytearray, str, Path],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Dict[str, Union[str, int, float]]:
    result, _ = _predict(model, classes, image, image_size)
    return result


def predict_with_top3(
    model: torch.nn.Module,
    classes: Sequence[str],
    image: Union[Image.Image, bytes, bytearray, str, Path],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Tuple[Dict[str, Union[str, int, float]], List[Dict[str, Union[str, float]]]]:
    return _predict(model, classes, image, image_size)
