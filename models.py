# models.py

import torch.nn as nn
from torchvision.models import resnet50, resnet101, resnet18, resnet34
import torch

def build_model(model_name: str, num_classes: int):
    if model_name == "resnet50":
        model = resnet50(weights="IMAGENET1K_V1")
    elif model_name == "resnet101":
        model = resnet101(weights="IMAGENET1K_V1")
    elif model_name == "resnet18":
        model = resnet18(weights="IMAGENET1K_V1")
    elif model_name == "resnet34":
        model = resnet34(weights="IMAGENET1K_V1")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # CIFAR-10 用調整
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model



def _cifarize_resnet(model, num_classes):
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_teacher(name, num_classes, load_from=None):
    if name == "resnet34":
        model = resnet34(weights="IMAGENET1K_V1")
        # ./checkpoints/resnet34_epoch10.pth をload
    else:
        raise ValueError(name)
    model = _cifarize_resnet(model, num_classes)
    if load_from is not None:
        print(f"Loading teacher model from {load_from}")
        model.load_state_dict(torch.load(load_from))

    return model

def build_student(name, num_classes):
    if name == "resnet18":
        model = resnet18(weights="IMAGENET1K_V1")
    else:
        raise ValueError(name)
    return _cifarize_resnet(model, num_classes)


