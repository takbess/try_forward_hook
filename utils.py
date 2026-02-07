# utils.py

import torch

import os
import torch


def get_device(device_str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")
