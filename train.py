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


def main():
    device = get_device(config.DEVICE)

    model = build_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
    ).to(device)

    # hook = VisualizeFeatureHook(path="log_feat_stats/resnet18.log")
    hook = VisualizeFeatureHook(path="log_feat_stats/resnet34.log")
    getattr(model, config.STUDENT_HOOK_LAYER)\
        .register_forward_hook(hook)


    trainloader, testloader = build_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )
    best_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device
        )
        test_acc = evaluate(model, testloader, device)
        scheduler.step()

        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     save_checkpoint(
        #         model,
        #         config.SAVE_DIR,
        #         config.SAVE_BEST_NAME,
        #     )


        print(
            f"Epoch [{epoch}/{config.EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Test Acc: {test_acc:.2f}%"
        )

    save_checkpoint(
        model,
        config.SAVE_DIR,
        config.SAVE_LAST_NAME,
    )
    
if __name__ == "__main__":
    main()
