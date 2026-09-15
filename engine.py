"""Training/validation loops (plain cross-entropy and logit-based KD) and the shared
epoch loop with early stopping + checkpointing used by every train_*.py script."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from common import save_json


def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    device = next(model.parameters()).device

    with tqdm(loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            running_loss += loss.item() * bs
            _, predicted = torch.max(outputs, dim=1)
            total += bs
            correct += (predicted == labels).sum().item()
            tepoch.set_postfix(loss=running_loss / max(1, total), accuracy=100.0 * correct / max(1, total))

    return running_loss / max(1, len(loader.dataset)), 100.0 * correct / max(1, total)


def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / max(1, len(loader.dataset)), 100.0 * correct / max(1, total)


def train_epoch_kd(student, teacher, loader, criterion_ce, criterion_kl, optimizer,
                    kd_alpha, kd_temperature, epoch):
    """Logit-based KD: student sees the RGB slice of the teacher's input; loss =
    (1 - alpha) * CE(student, label) + alpha * T^2 * KL(student/T || teacher/T)."""
    student.train()
    running_loss, correct, total = 0.0, 0, 0
    device = next(student.parameters()).device

    with tqdm(loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            tepoch.set_description(f"[KD] Epoch {epoch}")
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs[:, :, :3, :, :])
            loss_ce = criterion_ce(student_logits, labels)
            loss_kld = criterion_kl(
                F.log_softmax(student_logits / kd_temperature, dim=1),
                F.softmax(teacher_logits / kd_temperature, dim=1),
            ) * (kd_temperature ** 2)
            loss = (1.0 - kd_alpha) * loss_ce + kd_alpha * loss_kld

            loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            running_loss += loss.item() * bs
            _, predicted = torch.max(student_logits, dim=1)
            total += bs
            correct += (predicted == labels).sum().item()
            tepoch.set_postfix(loss=running_loss / max(1, total), accuracy=100.0 * correct / max(1, total))

    return running_loss / max(1, total), 100.0 * correct / max(1, total)


def validate_epoch_kd(student, loader, criterion_ce):
    student.eval()
    running_loss, correct, total = 0.0, 0, 0
    device = next(student.parameters()).device

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            logits = student(inputs[:, :, :3, :, :])
            loss = criterion_ce(logits, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / max(1, total), 100.0 * correct / max(1, total)


def run_training_loop(model, train_loader, val_loader, optimizer, log_dir, cfg,
                       num_epochs, esp, kd=False, teacher=None, kd_alpha=0.7, kd_temperature=3.0):
    """Shared epoch loop: trains, validates, logs to TensorBoard + training_log.txt,
    checkpoints (latest + best-by-val-loss), and applies early stopping. Updates and
    persists `cfg` (config.json) with the final training outcome."""
    criterion = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean") if kd else None
    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_txt = os.path.join(log_dir, "training_log.txt")

    def save_checkpoint(state, is_best):
        torch.save(state, os.path.join(checkpoint_dir, "checkpoint_latest.pth"))
        if is_best:
            torch.save(state, os.path.join(checkpoint_dir, "best_model.pth"))

    best_val_loss = float("inf")
    early_stopping_counter = 0
    epoch = 0

    for epoch in range(num_epochs):
        if kd:
            train_loss, train_acc = train_epoch_kd(model, teacher, train_loader, criterion, criterion_kl,
                                                     optimizer, kd_alpha, kd_temperature, epoch)
            val_loss, val_acc = validate_epoch_kd(model, val_loader, criterion)
        else:
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        with open(log_txt, "a") as f:
            f.write(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Train Acc: {train_acc}\n")
            f.write(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Acc: {val_acc}\n\n")

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        save_checkpoint({
            "epoch": epoch + 1, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc,
        }, is_best=is_best)

        if early_stopping_counter >= esp:
            print("Early stopping triggered")
            with open(log_txt, "a") as f:
                f.write("Early stopping triggered\n")
            break

    writer.close()
    cfg.update({
        "final_epoch": epoch + 1,
        "best_val_loss": best_val_loss,
        "stopped_early": early_stopping_counter >= esp,
    })
    save_json(os.path.join(log_dir, "config.json"), cfg)
    return cfg
