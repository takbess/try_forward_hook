# train_crosskd.py

import torch
import torch.optim as optim

import configcrosskd as config
from models import build_teacher, build_student
from datasets import build_dataloaders
from crosskd_hooks import SaveFeatureHook
from engine_crosskd import train_one_epoch_crosskd, evaluate
from utils import get_device, save_checkpoint
from utils import log_feat_stats

def main():
    device = get_device(config.DEVICE)

    student = build_student(
        config.STUDENT_MODEL, config.NUM_CLASSES
    ).to(device)

    teacher = build_teacher(
        config.TEACHER_MODEL, config.NUM_CLASSES
    ).to(device)

    # freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # hooks
    s_hook = SaveFeatureHook()

    getattr(student, config.STUDENT_HOOK_LAYER)\
        .register_forward_hook(s_hook)

    def teacher_pre_hook(module, input):
        log_feat_stats(s_hook.feat, epoch, -1, path="log_feat_stats/crosskd.log")

        return (s_hook.feat,)

    getattr(teacher, config.TEACHER_INJECT_LAYER)\
        .register_forward_pre_hook(teacher_pre_hook)

    trainloader, testloader = build_dataloaders(
        config.BATCH_SIZE, config.NUM_WORKERS
    )

    optimizer = optim.SGD(
        student.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )
    best_acc = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        loss, acc = train_one_epoch_crosskd(
            student,
            teacher,
            s_hook,
            trainloader,
            optimizer,
            device,
            config.LAMBDA_CROSSKD,
        )

        test_acc = evaluate(student, testloader, device)
        scheduler.step()

        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     save_checkpoint(
        #         student,
        #         config.SAVE_DIR,
        #         config.SAVE_BEST_NAME,
        #     )


        print(
            f"Epoch [{epoch}/{config.EPOCHS}] "
            f"Loss {loss:.4f} "
            f"Train Acc {acc:.2f}% "
            f"Test Acc {test_acc:.2f}%"
        )
        # log
        with open(f"log/{config.MODEL_NAME}.log","a") as f:
            f.write(
                f"{epoch},{loss:.4f},{acc:.2f},{test_acc:.2f}\n"
            )

    save_checkpoint(
        student,
        config.SAVE_DIR,
        config.SAVE_LAST,
    )


if __name__ == "__main__":
    main()
