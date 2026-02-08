# train.py

import torch
import torch.nn as nn
import torch.optim as optim

import config
from models import build_model
from datasets import build_dataloaders
from crosskd_hooks import VisualizeFeatureHook
from engine import train_one_epoch, evaluate
from utils import get_device, save_checkpoint
from utils import get_all_layer_shapes

def main():
    device = get_device(config.DEVICE)

    model = build_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
    )
    # .to(device)

    shapes = get_all_layer_shapes(model, input_shape=(1, 3, 32, 32))
    for name, shape in shapes.items():
        print(f"{name:30s} -> {shape}")

    
if __name__ == "__main__":
    main()
