import os
from PIL import Image

def scan_plant_disease(path):
    print("Scanning plant disease dataset at:", path)
    if not os.path.exists(path):
        print("Path does not exist!")
        return
    classes = os.listdir(path)
    total_imgs = 0
    corrupted = 0
    for cls in classes:
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            files = os.listdir(cls_path)
            total_imgs += len(files)
            for f in files[:5]: # sample check
                try:
                    with Image.open(os.path.join(cls_path, f)) as img:
                        img.verify()
                except Exception:
                    corrupted += 1
    print(f"Total plant disease classes: {len(classes)}, total images: {total_imgs}, sample check corrupted: {corrupted}")

scan_plant_disease(r"C:\Users\Dhanush\OneDrive\Desktop\CL\datasets\plant_disease\data")
