# utils.py

import torch

import os
import torch
import torch.nn as nn

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


def get_all_layer_shapes(model, input_shape=(1, 3, 32, 32)):
    """
    model 内のすべての named_modules に forward hook を仕掛け、
    出力 tensor の shape を取得する。

    Returns:
        shapes: dict {layer_name: output_shape}
    """
    shapes = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                shapes[name] = tuple(out.shape)
            elif isinstance(out, (list, tuple)):
                shapes[name] = [tuple(o.shape) for o in out if isinstance(o, torch.Tensor)]
        return hook

    for name, module in model.named_modules():
        # 空 module（model 自身）はスキップ
        if name == "":
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(*input_shape)
        model(dummy)

    for h in hooks:
        h.remove()

    return shapes
