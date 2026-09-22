import asyncio
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_service.model_loader import (
    load_crop_recommender,
    load_yield_predictor,
    load_price_predictor,
    load_fertilizer_recommender,
    load_image_model,
)
from ai_service.app import crop_recommendation, CropRecInputs


class ModelLoaderTests(unittest.TestCase):
    def test_loaders_work_from_ai_service_directory(self):
        original_cwd = os.getcwd()
        os.chdir(Path(__file__).resolve().parents[1])
        try:
            self.assertIsNotNone(load_crop_recommender())
            self.assertIsNotNone(load_yield_predictor())
            self.assertIsNotNone(load_price_predictor())
            self.assertIsNotNone(load_fertilizer_recommender())
            self.assertIsNotNone(load_image_model('ai/models/image_classification/plant_resnet9.pt'))
        finally:
            os.chdir(original_cwd)

    def test_crop_model_accepts_dashboard_feature_names(self):
        model = load_crop_recommender()
        self.assertIsNotNone(model)

        df = pd.DataFrame([{
            'N': 80,
            'P': 40,
            'K': 120,
            'temperature': 28,
            'humidity': 70,
            'ph': 6.8,
            'rainfall': 180,
        }])

        prediction = model.predict(df)
        self.assertEqual(len(prediction), 1)

    def test_crop_recommendation_returns_readable_label(self):
        async def run_prediction():
            result = await crop_recommendation(CropRecInputs(
                state='Karnataka',
                district='Mandya',
                season='Kharif',
                soilType='Clay',
                nitrogen=80,
                phosphorus=40,
                potassium=120,
                temperature=28,
                humidity=70,
                rainfall=180,
            ))
            self.assertIn('recommendedCrop', result)
            self.assertTrue(isinstance(result['recommendedCrop'], str))
            self.assertNotEqual(result['recommendedCrop'], '5')

        asyncio.run(run_prediction())


if __name__ == '__main__':
    unittest.main()
