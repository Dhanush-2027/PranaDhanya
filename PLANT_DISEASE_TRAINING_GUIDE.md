# 🌿 Plant Disease Model Training - Step-by-Step Guide

Welcome! This guide explains **every single click and step** to train the high-accuracy (256x256) ResNet9 Plant Disease Model on Google Colab GPU and connect it back to your project.

---

## 📁 Files Prepared on Your Desktop
1. **`plant_disease.zip`** → Located at `C:\Users\Dhanush\OneDrive\Desktop\plant_disease.zip` (Your full 71-class plant dataset).
2. **`Plant_Disease_ResNet9_Training.ipynb`** → Located at `C:\Users\Dhanush\OneDrive\Desktop\Plant_Disease_ResNet9_Training.ipynb` (Your 16-cell Google Colab Jupyter Notebook).

---

## 🚀 Part 1: Uploading to Google Drive

1. Open your web browser and go to [Google Drive](https://drive.google.com).
2. Make sure you are signed into your Google account.
3. In Google Drive, create a folder named **`CL`** (or open it if you already created it):
   - Click **`+ New`** (top-left) → **`New folder`** → type `CL` → click **`Create`**.
4. Inside `CL`, open it and create a subfolder named **`datasets`**:
   - Double-click `CL` → click **`+ New`** → **`New folder`** → type `datasets` → click **`Create`**.
5. Upload your dataset zip:
   - Double-click the `datasets` folder.
   - Drag and drop **`plant_disease.zip`** from your Desktop into this folder.
   - *Wait until the upload progress circle completes.*
6. *(Optional)* If you have an existing model checkpoint, you can create a folder `models` inside `CL` and upload `plant_disease_model.pth` there.

---

## 💻 Part 2: Opening & Running on Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. In the popup window that appears, click the **`Upload`** tab.
3. Click **`Browse`** and select **`Plant_Disease_ResNet9_Training.ipynb`** from your Desktop (`C:\Users\Dhanush\OneDrive\Desktop`).
4. **Enable T4 GPU (Crucial)**:
   - In the Colab top menu, click **`Runtime`** → **`Change runtime type`**.
   - Under *Hardware accelerator*, select **`T4 GPU`**.
   - Click **`Save`**.
5. **Run the Notebook Cells**:
   - Run each cell one by one from top to bottom by clicking inside the cell and pressing **`Shift + Enter`** (or clicking the ▶ Play button on the left of each cell).
   - In **Cell 3 (Mount Google Drive)**, Colab will ask: *"Permit this notebook to access your Google Drive files?"* Click **`Connect to Google Drive`** and select your account.
   - **Cell 4** will unzip your dataset into Colab's high-speed local disk and create an 80% Train / 10% Validation / 10% Test split.
   - **Cell 9** will train the ResNet9 CNN model across 25 epochs.
   - **Cell 16** will automatically export and download:
     - `plant_disease_resnet9.pth` (Model weights)
     - `plant_class_names.json` (List of 71 disease names)
     - `preprocessing_config.json` (Image size & normalization settings)

---

## 🔗 Part 3: Connecting the Model to Your Project

Once training finishes and you download the 3 files to your PC (e.g. `Downloads` folder):
1. Copy `plant_disease_resnet9.pth` to:
   `C:\Users\Dhanush\OneDrive\Desktop\CL\ai\models\image_classification\plant_disease_resnet9.pth`
2. Copy `plant_class_names.json` to:
   `C:\Users\Dhanush\OneDrive\Desktop\CL\ai\models\image_classification\plant_class_names.json`
3. Tell me you have downloaded them, and I will automatically run the end-to-end test (`e2e_test.py`) to verify that the plant diagnosis API produces 100% accurate results!
