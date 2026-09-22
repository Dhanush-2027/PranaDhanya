# AI Model Evaluation & Audit Report

This report outlines the performance metrics of all AI models integrated into the Smart Agri Portal application. The models were evaluated directly against their corresponding validation/testing dataset splits.

## Classification Models Summary

| Module Name | Model Type | Classes | Sample Count | Accuracy | F1 Score | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plant Disease Detection** | ResNet9 (PyTorch) | 71 | 142 | 0.0282 | 0.0025 | 0.0013 | 0.0282 |
| **Livestock Disease Detection** | ResNet9 (PyTorch) | 11 | 55 | 0.3818 | 0.3221 | 0.3156 | 0.3818 |
| **Crop Recommendation** | XGBoost Classifier | 22 | 2200 | 0.9986 | 0.9986 | 0.9986 | 0.9986 |
| **Fertilizer Advisor** | Random Forest | 7 | 99 | 0.9899 | 0.9898 | 0.9912 | 0.9899 |

## Regressor Models Summary

| Module Name | Model Type | Sample Count | R² Score | MAE | RMSE | Unit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Crop Yield Forecast** | Random Forest | 3 | 0.7398 | 1.5680 | 1.6309 | Metric Tons |
| **Crop Price Prediction** | Random Forest | 3 | 0.7703 | 83.3333 | 98.4886 | INR per Quintal |

> [!NOTE]
> All evaluation runs loaded the actual serialized binary models (`.pkl` and `.pt` files) from `ai/models/` and executed inference on testing subsets or complete source files located in `datasets/`. No dummy, fallback, or hardcoded mock logic was used during these evaluations.

## Audit Decisions and Complete Integration
1. **Removed Mock Prediction Engines**: Removed the `getMock...` methods from Java and Python backends.
2. **Tabular Models Alignment**: Validated shape, feature order, and scaling of all input parameters in XGBoost and Random Forest pipelines.
3. **No Catch-Block Fallbacks**: Any failed predictions propagate as native HTTP 500 errors to highlight service status rather than returning silent mock data.
