import asyncio
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi import HTTPException, UploadFile
from PIL import Image

from ai.plant_inference import (
    CONFIDENCE_THRESHOLD,
    PLANT_CLASS_NAMES_PATH,
    PLANT_MODEL_PATH,
    PlantImageInputError,
    inspect_plant_dataset,
    load_plant_model,
    plant_display_name,
    predict,
    preprocess_plant_image,
    preprocess_plant_image_path,
)
from ai_service.app import predict_plant
from ai_service.model_loader import load_image_model


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


class PlantInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required_paths = (PLANT_MODEL_PATH, PLANT_CLASS_NAMES_PATH, DATA_DIR)
        if not all(path.exists() for path in required_paths):
            raise unittest.SkipTest("Plant checkpoint, class list, or dataset is unavailable")

        torch.set_num_threads(min(4, torch.get_num_threads()))
        cls.api_checkpoint = load_plant_model(device="cpu")
        cls.direct_checkpoint = load_plant_model(device="cpu")
        cls.checkpoint = cls.api_checkpoint
        cls.samples = select_random_images(cls.checkpoint["classes"])
        if len(cls.samples) != SAMPLE_COUNT:
            raise unittest.SkipTest(
                f"Expected at least {SAMPLE_COUNT} image files, found fewer"
            )

    def test_api_and_direct_inference_match_for_twenty_random_images(self):
        classes = self.checkpoint["classes"]
        model = self.direct_checkpoint["model"]
        targets = []
        direct_indices = []
        api_indices = []
        api_labels = []

        self.assertEqual(len(classes), self.checkpoint["output_count"])
        self.assertEqual(classes, self.checkpoint["classes"])

        with patch("ai_service.app.PLANT_CHECKPOINT", self.checkpoint):
            for sample_index, (image_path, expected_id) in enumerate(
                self.samples, start=1
            ):
                image_bytes = image_path.read_bytes()
                direct_prediction = predict(
                    model,
                    classes,
                    image_bytes,
                    image_size=self.checkpoint["img_size"],
                )
                direct_tensor = preprocess_plant_image_path(
                    image_path,
                    self.checkpoint["img_size"],
                ).unsqueeze(0)
                self.assertEqual(
                    tuple(direct_prediction["tensor_shape"]),
                    tuple(direct_tensor.shape),
                )

                with Image.open(io.BytesIO(image_bytes)) as image:
                    same_upload_tensor = preprocess_plant_image(
                        image,
                        self.checkpoint["img_size"],
                    ).unsqueeze(0)
                self.assertTrue(
                    torch.allclose(direct_tensor, same_upload_tensor, rtol=0, atol=1e-6)
                )

                upload = UploadFile(
                    filename=image_path.name,
                    file=io.BytesIO(image_bytes),
                )
                api_response = asyncio.run(predict_plant(upload))

                self.assertEqual(api_response["classIndex"], direct_prediction["class_index"])
                self.assertEqual(api_response["classId"], direct_prediction["class_id"])
                self.assertEqual(api_response["diseaseName"], direct_prediction["label"])
                self.assertAlmostEqual(
                    api_response["confidence"] / 100.0,
                    direct_prediction["confidence"],
                    places=6,
                )
                self.assertEqual(
                    api_response["uncertain"],
                    direct_prediction["uncertain"],
                )
                self.assertEqual(
                    api_response["top3Predictions"][0]["classId"],
                    direct_prediction["class_id"],
                )

                expected_index = self.checkpoint["class_to_idx"][expected_id]
                targets.append(expected_index)
                direct_indices.append(int(direct_prediction["class_index"]))
                api_indices.append(int(api_response["classIndex"]))
                api_labels.append(api_response["classId"])
                print(
                    f"{sample_index:02d}: {image_path.name} "
                    f"true={expected_id} direct={direct_prediction['class_id']} "
                    f"API={api_response['classId']}"
                )

        matches = sum(api == direct for api, direct in zip(api_indices, direct_indices))
        correct = sum(target == predicted for target, predicted in zip(targets, direct_indices))
        print(f"Loaded classes={len(classes)}")
        print(
            f"API Match Rate={matches / SAMPLE_COUNT:.1%} "
            f"({matches}/{SAMPLE_COUNT})"
        )
        print(
            f"Folder-label accuracy={correct / SAMPLE_COUNT:.1%} "
            f"({correct}/{SAMPLE_COUNT})"
        )
        self.assertEqual(matches, SAMPLE_COUNT)
        self.assertGreater(
            len(set(api_labels)),
            1,
            "The API must produce diverse labels across the random image sample",
        )

    def test_label_mapping_and_checkpoint_integrity(self):
        classes = self.checkpoint["classes"]
        self.assertEqual(self.api_checkpoint["classes"], self.direct_checkpoint["classes"])
        self.assertEqual(
            self.api_checkpoint["_resolved_path"],
            self.direct_checkpoint["_resolved_path"],
        )
        for key, api_value in self.api_checkpoint["model"].state_dict().items():
            self.assertTrue(
                torch.equal(api_value, self.direct_checkpoint["model"].state_dict()[key]),
                f"Loaded checkpoint state differs at {key}",
            )
        self.assertEqual(len(classes), len(set(classes)))
        self.assertEqual(set(self.checkpoint["class_to_idx"].values()), set(range(len(classes))))
        self.assertEqual(
            [self.checkpoint["class_to_idx"][name] for name in classes],
            list(range(len(classes))),
        )
        self.assertEqual(
            self.checkpoint["display_names"],
            [plant_display_name(class_id) for class_id in classes],
        )
        self.assertEqual(len(self.checkpoint["display_names"]), len(set(self.checkpoint["display_names"])))
        self.assertFalse(self.checkpoint["model"].training)
        self.assertEqual(self.checkpoint["architecture"], type(self.checkpoint["model"]).__name__)
        self.assertEqual(self.checkpoint["output_count"], len(classes))
        self.assertEqual(self.checkpoint["training_class_count"], len(classes))
        self.assertIn("plant_resnet9_best.pt", self.checkpoint["_resolved_path"])

    def test_dataset_mapping_diagnostics(self):
        info = inspect_plant_dataset(DATA_DIR, self.checkpoint["classes"])
        self.assertTrue(info["available"])
        self.assertEqual(info["directory_class_count"], len(self.checkpoint["classes"]))
        self.assertGreater(info["image_count"], 0)
        self.assertEqual(info["missing_classes"], [])
        self.assertEqual(info["unsupported_directories"], [])

    def test_class_mapping_mismatch_fails_loudly(self):
        wrong_classes = self.checkpoint["classes"][:-1]
        with tempfile.TemporaryDirectory() as temp_dir:
            classes_path = Path(temp_dir) / "classes.json"
            classes_path.write_text(json.dumps(wrong_classes), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match the model checkpoint"):
                load_plant_model(class_names_path=classes_path, device="cpu")

    def test_incompatible_checkpoint_state_fails_strict_loading(self):
        state = dict(self.checkpoint["model_state"])
        state.pop("classifier.bias")
        broken_checkpoint = {
            "classes": self.checkpoint["classes"],
            "model_state": state,
            "img_size": self.checkpoint["img_size"],
        }
        with patch("ai.plant_inference.torch.load", return_value=broken_checkpoint):
            with self.assertRaises(RuntimeError):
                load_plant_model(device="cpu")

    def test_legacy_plant_checkpoints_are_not_silently_substituted(self):
        legacy_checkpoint = (
            ROOT_DIR
            / "ai"
            / "models"
            / "image_classification"
            / "plant-disease-resnet9.pth"
        )
        if not legacy_checkpoint.is_file():
            self.skipTest("No legacy 38-output plant checkpoint is present")

        with self.assertRaisesRegex(ValueError, "legacy plant checkpoint"):
            load_image_model(str(legacy_checkpoint), device="cpu")

    def test_invalid_images_and_missing_files_do_not_predict_a_label(self):
        with self.assertRaises(PlantImageInputError):
            predict(self.checkpoint["model"], self.checkpoint["classes"], b"not an image")
        with self.assertRaises(PlantImageInputError):
            predict(self.checkpoint["model"], self.checkpoint["classes"], b"")
        with self.assertRaises(PlantImageInputError):
            predict(
                self.checkpoint["model"],
                self.checkpoint["classes"],
                DATA_DIR / "missing-image.jpg",
            )

        with patch("ai_service.app.PLANT_CHECKPOINT", self.checkpoint):
            upload = UploadFile(filename="corrupt.jpg", file=io.BytesIO(b"broken"))
            with self.assertRaises(HTTPException) as error:
                asyncio.run(predict_plant(upload))
        self.assertEqual(error.exception.status_code, 400)
        self.assertNotIn("diseaseName", error.exception.detail)
        with patch("ai_service.app.PLANT_CHECKPOINT", self.checkpoint):
            upload = UploadFile(filename="empty.jpg", file=io.BytesIO(b""))
            with self.assertRaises(HTTPException) as empty_error:
                asyncio.run(predict_plant(upload))
        self.assertEqual(empty_error.exception.status_code, 400)

    def test_confidence_uses_top_probability_and_uncertainty_threshold(self):
        image_path, _ = self.samples[0]
        prediction = predict(
            self.checkpoint["model"],
            self.checkpoint["classes"],
            image_path,
            image_size=self.checkpoint["img_size"],
        )
        self.assertAlmostEqual(
            prediction["confidence"],
            prediction["top_k"][0]["confidence"],
            places=7,
        )
        self.assertEqual(
            prediction["uncertain"],
            prediction["confidence"] < CONFIDENCE_THRESHOLD,
        )
        self.assertEqual(prediction["class_id"], self.checkpoint["classes"][prediction["class_index"]])


if __name__ == "__main__":
    unittest.main()
