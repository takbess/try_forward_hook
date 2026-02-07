# engine_kd.py

import torch
import torch.nn.functional as F


def train_one_epoch_kd(
    teacher,
    student,
    t_hook,
    s_hook,
    loader,
    optimizer,
    device,
    lambda_kd,
):
    student.train()
    teacher.eval()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            _ = teacher(x)

        logits = student(x)

        loss_ce = F.cross_entropy(logits, y)
        loss_kd = F.mse_loss(
            s_hook.feature,
            t_hook.feature.detach()
        )
        loss = loss_ce + lambda_kd * loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        total += y.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(student, loader, device):
    student.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = student(x)
        correct += logits.argmax(1).eq(y).sum().item()
        total += y.size(0)

    return 100.0 * correct / total
