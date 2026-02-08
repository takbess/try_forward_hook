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

def log_feat_stats(feat, epoch, step, path="feat_stats.log"):
    mean = feat.mean().item()
    var  = feat.var(unbiased=False).item()
    with open(path, "a") as f:
        f.write(f"{epoch},{step},{mean},{var}\n")
