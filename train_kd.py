# train_kd.py

import torch
import torch.optim as optim

import configkd as config
from models import build_teacher, build_student
from datasets import build_dataloaders
from engine_kd import train_one_epoch_kd, evaluate
from kd_hooks import FeatureHook
from kd_hooks import FeatureVisualizeHook
from utils import get_device, save_checkpoint


def main():
    device = get_device(config.DEVICE)

    teacher = build_teacher(
        config.TEACHER_MODEL, config.NUM_CLASSES, config.teacher_load_from
    ).to(device)
    student = build_student(
        config.STUDENT_MODEL, config.NUM_CLASSES
    ).to(device)

    # freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # hooks
    t_hook = FeatureHook()
    s_hook = FeatureVisualizeHook(path="log_feat_stats/kd.log")

    getattr(teacher, config.KD_LAYER).register_forward_hook(t_hook)
    getattr(student, config.KD_LAYER).register_forward_hook(s_hook)

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
        loss, acc = train_one_epoch_kd(
            teacher,
            student,
            t_hook,
            s_hook,
            trainloader,
            optimizer,
            device,
            config.LAMBDA_KD,
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

    save_checkpoint(
        student,
        config.SAVE_DIR,
        config.SAVE_LAST_NAME,
    )
    
if __name__ == "__main__":
    main()
