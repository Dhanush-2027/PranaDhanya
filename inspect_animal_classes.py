import torch
checkpoint = torch.load('ai/models/image_classification/animal_resnet9_best.pt', map_location='cpu', weights_only=False)
print("Classes:", checkpoint.get('classes'))
