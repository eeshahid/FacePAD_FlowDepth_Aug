"""Distill the RGB-only KD student from a trained OCIM leave-one-out IFD teacher run.

The student is trained on the same LOO train/val split as its teacher (same seed,
same val_source) and evaluated on the same held-out target's TEST split.
"""
import os
import glob
import argparse
from datetime import datetime

import torch
import torch.optim as optim

from common import build_transforms, build_model, save_json, load_json, set_data_root
from utils import collate_fn, create_log_directory
from engine import run_training_loop
from datasets_ocim import DOMAIN_NAMES, LOO_SOURCES, VAL_SOURCES, build_sources_train_val


def _eval_hter(eval_json_path: str):
    try:
        d = load_json(eval_json_path)
    except Exception:
        return None
    m = d.get("metrics")
    return float(m["hter"]) if isinstance(m, dict) and isinstance(m.get("hter"), (int, float)) else None


def auto_select_teacher(target: str, base: str = "logs_loo_ocim") -> str:
    """Lowest-HTER completed teacher run for `target` (must have best_model.pth + eval json)."""
    pattern = os.path.join(base, "imgflowdepth", target.upper(), "*", "log_*")
    candidates = []
    for d in sorted(glob.glob(pattern)):
        ckpt = os.path.join(d, "checkpoints", "best_model.pth")
        eval_json = os.path.join(d, "loo_ocim_eval_test.json")
        if os.path.isfile(ckpt) and os.path.isfile(eval_json):
            hter = _eval_hter(eval_json)
            if hter is not None:
                candidates.append((hter, d))
    if not candidates:
        raise FileNotFoundError(
            f"No completed teacher run for target {target} under {base}/imgflowdepth/{target}/. "
            "Pass --teacher_log_dir explicitly."
        )
    candidates.sort(key=lambda x: x[0])
    print(f"[KD] selected teacher (lowest HTER {candidates[0][0]*100:.2f}%): {candidates[0][1]}")
    return candidates[0][1]


def parse_args():
    p = argparse.ArgumentParser(description="KD student for OCIM leave-one-out FacePAD")
    p.add_argument("--target", type=str, required=True, choices=["O", "C", "I", "M"])
    p.add_argument("--teacher_log_dir", type=str, default=None,
                    help="Default: auto-select the lowest-HTER teacher run for --target")
    p.add_argument("--kd_alpha", type=float, default=0.7)
    p.add_argument("--kd_temperature", type=float, default=3.0)
    p.add_argument("--val_source", type=str, default=None, choices=list(VAL_SOURCES),
                    help="Default: inherit the teacher's val_source")
    p.add_argument("--seed", type=int, default=None, help="Default: inherit the teacher's seed")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--esp", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=None, help="Default: teacher's batch_size")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--student_pretrained", action="store_true", default=True)
    p.add_argument("--no-student_pretrained", dest="student_pretrained", action="store_false")
    p.add_argument("--eval", action="store_true", default=True)
    p.add_argument("--no-eval", dest="eval", action="store_false")
    p.add_argument("--log_base_dir", type=str, default="./logs_loo_ocim_kd")
    return p.parse_args()


def _seed_everything(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_data_root(args.data_root)

    target = args.target.upper()
    sources = LOO_SOURCES[target]
    teacher_log_dir = args.teacher_log_dir or auto_select_teacher(target)
    teacher_hter = _eval_hter(os.path.join(teacher_log_dir, "loo_ocim_eval_test.json"))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tcfg = load_json(os.path.join(teacher_log_dir, "config.json"))
    teacher = build_model(tcfg, device)
    ckpt = torch.load(os.path.join(teacher_log_dir, "checkpoints", "best_model.pth"), map_location=device)
    teacher.load_state_dict(ckpt["model_state_dict"], strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[KD] teacher loaded: {teacher_log_dir}")

    img_size = int(tcfg.get("img_size", 224))
    num_frames = int(tcfg.get("num_frames", 1))
    flow_channels = int(tcfg.get("flow_channels", 3))
    flow_representation = tcfg.get("flow_representation", "hsv")
    flow_offset = int(tcfg.get("flow_offset", 0))
    val_source = args.val_source or tcfg.get("val_source", "target")
    seed = int(args.seed if args.seed is not None else tcfg.get("seed", 1234))
    batch_size = args.batch_size if args.batch_size is not None else int(tcfg.get("batch_size", 16))
    _seed_everything(seed)

    log_dir = create_log_directory(base_dir=os.path.join(args.log_base_dir, "imgflowdepth", target, val_source))
    print(f"[KD] OCIM {'&'.join(sources)} -> {target} ({DOMAIN_NAMES[target]}) | log dir: {log_dir}")

    transform = build_transforms(img_size, flow_channels, flow_representation)
    train_ds, val_ds = build_sources_train_val(
        sources, val_source, img_size=img_size, num_frames=num_frames, flow_channels=flow_channels,
        flow_representation=flow_representation, flow_offset=flow_offset, transform=transform,
        val_targets=[target], seed=seed,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                                collate_fn=collate_fn, pin_memory=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate_fn, pin_memory=True, num_workers=args.num_workers)

    cfg = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "kd": True,
        "kd_alpha": args.kd_alpha,
        "kd_temperature": args.kd_temperature,
        "target": target,
        "sources": sources,
        "eval_targets": [target],
        "val_source": val_source,
        "teacher_ref": teacher_log_dir,
        "teacher_hter": teacher_hter,
        "student_pretrained": bool(args.student_pretrained),
        "student_projector_dim": 0,
        "img_size": img_size,
        "num_frames": num_frames,
        "flow_channels": flow_channels,
        "flow_representation": flow_representation,
        "flow_offset": flow_offset,
        "batch_size": batch_size,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "esp": args.esp,
        "num_workers": args.num_workers,
        "seed": seed,
        "gpu": args.gpu,
        "FLOW_T": 20.0,
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
        from evaluate_ocim import evaluate_loo_from_config
        summary = evaluate_loo_from_config(log_dir, device=device)
        student_hter = (summary.get("metrics") or {}).get("hter")
        if student_hter is not None and teacher_hter is not None:
            print(f"[KD] teacher HTER {teacher_hter * 100:.2f}% -> student HTER {student_hter * 100:.2f}%")


if __name__ == "__main__":
    main()
