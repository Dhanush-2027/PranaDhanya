import torch

for model_name, path in [
    ("Plant Best", "ai/models/image_classification/plant_resnet9_best.pt"),
    ("Plant Final", "ai/models/image_classification/plant_resnet9.pt"),
    ("Animal Best", "ai/models/image_classification/animal_resnet9_best.pt"),
    ("Animal Final", "ai/models/image_classification/animal_resnet9.pt")
]:
    try:
        checkpoint = torch.load(path, map_location='cpu')
        print(f"\n=== {model_name} Checkpoint ===")
        print("Keys:", list(checkpoint.keys()))
        if 'epoch' in checkpoint:
            print("Epoch:", checkpoint['epoch'])
        if 'epochs_trained' in checkpoint:
            print("Epochs trained:", checkpoint['epochs_trained'])
        if 'accuracy' in checkpoint:
            print("Accuracy:", checkpoint['accuracy'])
        if 'best_accuracy' in checkpoint:
            print("Best Accuracy:", checkpoint['best_accuracy'])
        if 'classes' in checkpoint:
            print("Num Classes:", len(checkpoint['classes']))
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
