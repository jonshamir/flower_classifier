import torch
from torchvision import models

def create_model(n_classes):
    model = models.resnet34(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    n_features = model.fc.in_features
    model.fc = torch.nn.Linear(n_features, 256)
    model.fc2 = torch.nn.Linear(256, 256)
    model.fc3 = torch.nn.Linear(256, n_classes)
    return model
