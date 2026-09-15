"""Distill the RGB-only KD student from a trained joint (multi-dataset) IFD teacher run."""
import os
import argparse
from datetime import datetime

import torch
import torch.optim as optim

from common import (build_transforms, build_joint_train_val_datasets, build_model,
                     save_json, load_json, set_data_root, parse_dataset_list)
from utils import collate_fn, create_log_directory
from engine import run_training_loop


def parse_args():
    p = argparse.ArgumentParser(description="Train a joint KD student from a joint IFD teacher run")
    p.add_argument("--teacher_log_dir", type=str, required=True,
                    help="Path to a completed train_joint.py run")
    p.add_argument("--teacher_ckpt_name", type=str, default="best_model.pth")
    p.add_argument("--datasets", type=str, default=None,
                    help="Comma-separated dataset list. Default: the teacher's own dataset list")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=None, help="Default: teacher's batch_size")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--esp", type=int, default=50)
    p.add_argument("--kd_alpha", type=float, default=0.7)
    p.add_argument("--kd_temperature", type=float, default=3.0)
    p.add_argument("--student_pretrained", action="store_true", default=True)
    p.add_argument("--no-student_pretrained", dest="student_pretrained", action="store_false")
    p.add_argument("--eval", action="store_true", default=True)
    p.add_argument("--no-eval", dest="eval", action="store_false")
    p.add_argument("--log_base_dir", type=str, default="./logs_joint/student_kd")
    return p.parse_args()


def main():
    args = parse_args()
    set_data_root(args.data_root)

    tcfg = load_json(os.path.join(args.teacher_log_dir, "config.json"))
    datasets = parse_dataset_list(args.datasets or tcfg.get("datasets", tcfg.get("dataset", "RA,RM,RY")))
    dataset_tag = "_".join(datasets)

    img_size = int(tcfg.get("img_size", 224))
    num_frames = int(tcfg.get("num_frames", 1))
    flow_channels = int(tcfg.get("flow_channels", 3))
    flow_representation = tcfg.get("flow_representation", "hsv")
    flow_offset = int(tcfg.get("flow_offset", 0))
    protocol = tcfg.get("protocol", "all")
    n_split = tcfg.get("n_split", "NA")
    batch_size = args.batch_size if args.batch_size is not None else int(tcfg.get("batch_size", 16))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log_dir = create_log_directory(base_dir=os.path.join(args.log_base_dir, dataset_tag))
    print(f"[KD] Log dir: {log_dir}")

    transform = build_transforms(img_size, flow_channels, flow_representation)
    train_ds, val_ds = build_joint_train_val_datasets(
        datasets, img_size=img_size, num_frames=num_frames, flow_channels=flow_channels,
        flow_representation=flow_representation, flow_offset=flow_offset,
        protocol=protocol, n_split=n_split, transform=transform,
    )
    print(f"[KD] Datasets: {datasets} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                                collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate_fn, pin_memory=True)

    teacher = build_model(tcfg, device)
    ckpt = torch.load(os.path.join(args.teacher_log_dir, "checkpoints", args.teacher_ckpt_name), map_location=device)
    teacher.load_state_dict(ckpt["model_state_dict"], strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[KD] Loaded teacher from: {args.teacher_log_dir}")

    cfg = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "dataset_tag": dataset_tag,
        "dataset": datasets[0],
        "kd": True,
        "kd_alpha": args.kd_alpha,
        "kd_temperature": args.kd_temperature,
        "teacher_ref": args.teacher_log_dir,
        "teacher_ckpt_name": args.teacher_ckpt_name,
        "student_pretrained": bool(args.student_pretrained),
        "student_projector_dim": 0,
        "img_size": img_size,
        "num_frames": num_frames,
        "flow_channels": flow_channels,
        "flow_representation": flow_representation,
        "flow_offset": flow_offset,
        "protocol": protocol,
        "n_split": n_split,
        "batch_size": batch_size,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "esp": args.esp,
        "gpu": args.gpu,
        "FLOW_T": 20.0,
        "classes_order": tcfg.get("classes_order", ["attack", "real"]),
        "log_dir": log_dir,
        "checkpoint_dir": os.path.join(log_dir, "checkpoints"),
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "device": str(device),
    }
    save_json(os.path.join(log_dir, "config.json"), cfg)

    student = build_model(cfg, device)
    cfg["model_name"] = student.__class__.__name__
    cfg["num_params"] = sum(p.numel() for p in student.parameters())
    save_json(os.path.join(log_dir, "config.json"), cfg)

    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    run_training_loop(student, train_loader, val_loader, optimizer, log_dir, cfg,
                       num_epochs=args.num_epochs, esp=args.esp,
                       kd=True, teacher=teacher, kd_alpha=args.kd_alpha, kd_temperature=args.kd_temperature)

    if args.eval:
        print("\n[Eval] Running evaluate_joint.py on the KD student run...")
        from evaluate_joint import evaluate_joint_from_config
        evaluate_joint_from_config(log_dir, device=device)
        print("[Eval] Done.")

    print(f"\n[KD] Finished. Log dir: {log_dir}")


if __name__ == "__main__":
    main()
