import os
import shutil
from pathlib import Path

def main():
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed in the current environment.")
        print("Installing kagglehub...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub

    print("Downloading abdallahalidev/plantvillage-dataset via kagglehub...")
    downloaded_path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
    print(f"Dataset downloaded by kagglehub to: {downloaded_path}")

    target_dir = Path(r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\plantvillage-dataset")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying dataset files to: {target_dir} ...")
    src_path = Path(downloaded_path)
    
    # Copy all items in the downloaded directory to target_dir
    for item in src_path.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
            print(f"Copied directory: {item.name}")
        else:
            shutil.copy2(item, dest)
            print(f"Copied file: {item.name}")

    print(f"\nSuccessfully placed PlantVillage dataset in: {target_dir}")

if __name__ == "__main__":
    main()
