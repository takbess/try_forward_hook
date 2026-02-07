# engine_crosskd.py

import torch
import torch.nn.functional as F


def train_one_epoch_crosskd(
    student,
    teacher,
    s_hook,
    loader,
    optimizer,
    device,
    lambda_ckd,
):
    student.train()
    teacher.eval()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # ① student forward
        logits_s = student(x)
        loss_student = F.cross_entropy(logits_s, y)

        # ② teacher forward（student feat が注入される）
        logits_t = teacher(x)
        loss_crosskd = F.cross_entropy(logits_t, y)

        loss = loss_student + lambda_ckd * loss_crosskd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += logits_s.argmax(1).eq(y).sum().item()
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
