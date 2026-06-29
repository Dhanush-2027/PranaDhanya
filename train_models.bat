@echo off
REM Smart Agri Portal - Model Training Script for Windows
REM This script installs dependencies and trains all ML models

setlocal enabledelayedexpansion

cls
echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║         SMART AGRI PORTAL - MODEL TRAINING FOR WINDOWS              ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ✗ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✓ Python found:
python --version

REM Set default dataset path
set "DATASET_PATH=C:\Users\Dhanush\OneDrive\Desktop\C&L\datasets"

REM Ask for dataset path
set /p "DATASET_PATH=Enter dataset path (press Enter for default): "

if not exist "!DATASET_PATH!" (
    echo.
    echo ✗ Dataset path does not exist: !DATASET_PATH!
    pause
    exit /b 1
)

echo ✓ Using dataset path: !DATASET_PATH!
echo.

REM Step 1: Upgrade pip
echo ======================================================================
echo STEP 1: Upgrading pip
echo ======================================================================
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ✗ Failed to upgrade pip
    pause
    exit /b 1
)
echo ✓ pip upgraded successfully
echo.

REM Step 2: Install AI dependencies
echo ======================================================================
echo STEP 2: Installing AI module dependencies
echo ======================================================================
python -m pip install -r ai/requirements.txt
if errorlevel 1 (
    echo ✗ Failed to install AI dependencies
    pause
    exit /b 1
)
echo ✓ AI dependencies installed
echo.

REM Step 3: Install service dependencies
echo ======================================================================
echo STEP 3: Installing AI service dependencies
echo ======================================================================
python -m pip install -r ai_service/requirements.txt
if errorlevel 1 (
    echo ✗ Failed to install service dependencies
    pause
    exit /b 1
)
echo ✓ Service dependencies installed
echo.

REM Step 4: Verify datasets
echo ======================================================================
echo STEP 4: Verifying Datasets
echo ======================================================================

set "MISSING=0"

if not exist "!DATASET_PATH!\crop_recommendation\Crop_recommendation.csv" (
    echo ✗ Crop recommendation dataset NOT FOUND
    set "MISSING=1"
) else (
    echo ✓ Crop recommendation dataset found
)

if not exist "!DATASET_PATH!\yield_prediction\data.csv" (
    echo ✗ Yield prediction dataset NOT FOUND
    set "MISSING=1"
) else (
    echo ✓ Yield prediction dataset found
)

if not exist "!DATASET_PATH!\price_prediction\data.csv" (
    echo ✗ Price prediction dataset NOT FOUND
    set "MISSING=1"
) else (
    echo ✓ Price prediction dataset found
)

if not exist "!DATASET_PATH!\fertilizer_prediction\Fertilizer Prediction.csv" (
    echo ✗ Fertilizer recommendation dataset NOT FOUND
    set "MISSING=1"
) else (
    echo ✓ Fertilizer recommendation dataset found
)

if not exist "!DATASET_PATH!\plant_disease\data" (
    echo ✗ Plant disease images NOT FOUND
    set "MISSING=1"
) else (
    echo ✓ Plant disease images found
)

if not exist "!DATASET_PATH!\dog_skin_disease\train" (
    echo ✗ Animal disease images NOT FOUND
    set "MISSING=1"
) else (
    echo ✓ Animal disease images found
)

if "!MISSING!"=="1" (
    echo.
    echo ⚠ Some datasets are missing. Training may fail for missing datasets.
    set /p "CONTINUE=Continue anyway? (y/n): "
    if /i not "!CONTINUE!"=="y" (
        exit /b 1
    )
)

echo.

REM Step 5: Train models
echo ======================================================================
echo STEP 5: Training All Models
echo ======================================================================
echo.
echo Available training options:
echo   1. Train all models (recommended)
echo   2. Train only tabular models (crop, yield, price, fertilizer)
echo   3. Train only image models (plant disease, animal disease)
echo.

set /p "TRAIN_OPTION=Select option (1-3, default 1): "
if "!TRAIN_OPTION!"=="" set "TRAIN_OPTION=1"

if "!TRAIN_OPTION!"=="1" (
    python train_all_models.py --datasets "!DATASET_PATH!"
) else if "!TRAIN_OPTION!"=="2" (
    python train_all_models.py --datasets "!DATASET_PATH!" --skip-image
) else if "!TRAIN_OPTION!"=="3" (
    python train_all_models.py --datasets "!DATASET_PATH!" --skip-tabular
) else (
    echo Invalid option
    pause
    exit /b 1
)

if errorlevel 1 (
    echo.
    echo ✗ Model training failed
    pause
    exit /b 1
)

REM Step 6: Success
echo.
echo ======================================================================
echo ✓ TRAINING COMPLETED SUCCESSFULLY!
echo ======================================================================
echo.
echo Next steps:
echo.
echo 1. Start the AI service (in a new terminal):
echo    cd ai_service
echo    python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
echo.
echo 2. Test the API:
echo    Open http://localhost:8000/docs in your browser
echo.
echo 3. Run full stack with Docker:
echo    docker-compose up --build
echo.
echo 4. For detailed documentation:
echo    See TRAINING_GUIDE.md
echo.

pause
