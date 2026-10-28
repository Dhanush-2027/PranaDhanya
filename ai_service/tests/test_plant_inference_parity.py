import asyncio
import io
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi import UploadFile
from PIL import Image

from ai.plant_inference import (
    PLANT_CLASS_NAMES_PATH,
    PLANT_MODEL_PATH,
    TRANSFORM,
    load_plant_model,
    predict,
    preprocess_plant_image_path,
)
from ai_service.app import format_plant_display_name, predict_plant


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "datasets" / "plant_disease" / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_COUNT = 20
RANDOM_SEED = 42


def select_random_images(class_names, sample_count=SAMPLE_COUNT):
    images = []
    for class_name in class_names:
        class_dir = DATA_DIR / class_name
        if not class_dir.is_dir():
            continue
        images.extend(
            (path, class_name)
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if len(images) < sample_count:
        return []
    return random.Random(RANDOM_SEED).sample(images, sample_count)


class PlantInferenceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required_paths = (PLANT_MODEL_PATH, PLANT_CLASS_NAMES_PATH, DATA_DIR)
        if not all(path.exists() for path in required_paths):
            raise unittest.SkipTest("Plant checkpoint, class list, or dataset is unavailable")

        torch.set_num_threads(min(4, torch.get_num_threads()))
        cls.api_checkpoint = load_plant_model(device="cpu")
        cls.direct_checkpoint = load_plant_model(device="cpu")
        if cls.api_checkpoint["classes"] != cls.direct_checkpoint["classes"]:
            raise AssertionError("API and direct-inference class lists differ")
        cls.samples = select_random_images(cls.api_checkpoint["classes"])
        if len(cls.samples) != SAMPLE_COUNT:
            raise unittest.SkipTest(
                f"Expected at least {SAMPLE_COUNT} image files, found fewer"
            )

    def test_api_and_direct_inference_match_for_twenty_random_images(self):
        class_names = self.api_checkpoint["classes"]
        model = self.direct_checkpoint["model"]
        api_labels = []
        direct_correct = 0
        endpoint_matches = 0

        print(f"Loaded classes={len(class_names)}")
        self.assertEqual(len(class_names), len(self.direct_checkpoint["classes"]))

        with patch("ai_service.app.PLANT_CHECKPOINT", self.api_checkpoint):
            for sample_index, (image_path, expected_label) in enumerate(
                self.samples, start=1
            ):
                image_bytes = image_path.read_bytes()
                direct_prediction = predict(
                    model,
                    class_names,
                    image_bytes,
                    image_size=self.direct_checkpoint["img_size"],
                )
                direct_label = str(direct_prediction["label"])
                direct_correct += direct_label == expected_label

                with Image.open(io.BytesIO(image_bytes)) as image:
                    direct_tensor = TRANSFORM(
                        image.convert("RGB"),
                        self.direct_checkpoint["img_size"],
                    ).unsqueeze(0)
                path_tensor = preprocess_plant_image_path(
                    image_path,
                    self.direct_checkpoint["img_size"],
                ).unsqueeze(0)
                self.assertTrue(
                    torch.allclose(direct_tensor, path_tensor, rtol=0, atol=1e-6),
                    f"Preprocessing differs for {image_path}",
                )

                upload = UploadFile(
                    filename=image_path.name,
                    file=io.BytesIO(image_bytes),
                )
                response = asyncio.run(predict_plant(upload))
                endpoint_label = response["diseaseName"]
                expected_api_label = format_plant_display_name(direct_label)
                self.assertEqual(
                    endpoint_label,
                    expected_api_label,
                    f"API and direct predictions differ for {image_path}",
                )
                endpoint_matches += endpoint_label == expected_api_label
                api_labels.append(endpoint_label)

                print(
                    f"{sample_index:02d}: {image_path.name} | "
                    f"direct={direct_label} | api={endpoint_label} | "
                    f"ground_truth={expected_label}"
                )

        match_rate = endpoint_matches / len(self.samples)
        accuracy = direct_correct / len(self.samples)
        distinct_api_labels = len(set(api_labels))
        print(
            f"API Match Rate={match_rate * 100:.1f}% "
            f"({endpoint_matches}/{len(self.samples)})"
        )
        print(
            f"Direct folder-label accuracy={accuracy * 100:.1f}% "
            f"({direct_correct}/{len(self.samples)})"
        )
        print(f"Distinct API labels={distinct_api_labels}")

        self.assertEqual(endpoint_matches, SAMPLE_COUNT)
        self.assertGreater(distinct_api_labels, 1)


if __name__ == "__main__":
    unittest.main()
