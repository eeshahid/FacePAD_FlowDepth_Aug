"""Train the IFD teacher jointly on multiple datasets (default: RA+RM+RY)."""
import os
import argparse
from datetime import datetime

import torch
import torch.optim as optim

from common import build_transforms, build_joint_train_val_datasets, build_model, save_json, set_data_root, parse_dataset_list
from utils import collate_fn, create_log_directory
from engine import run_training_loop


def parse_args():
    p = argparse.ArgumentParser(description="Joint IFD (RGB+Flow+Depth) teacher training on multiple datasets")
    p.add_argument("--datasets", type=str, default="RA,RM,RY", help="Comma-separated dataset list")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_frames", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--esp", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--flow_channels", type=int, default=3, choices=[2, 3])
    p.add_argument("--flow_representation", type=str, default="hsv", choices=["uv", "uvm", "hsv"])
    p.add_argument("--flow_offset", type=int, default=0)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--protocol", type=str, default="all")
    p.add_argument("--n_split", type=str, default="NA")
    p.add_argument("--eval", action="store_true", default=True)
    p.add_argument("--no-eval", dest="eval", action="store_false")
    p.add_argument("--log_base_dir", type=str, default="./logs_joint")
    p.add_argument("--exp_description", type=str, default="Joint IFD training on multiple datasets")
    return p.parse_args()


def main():
    args = parse_args()
    set_data_root(args.data_root)
    datasets = parse_dataset_list(args.datasets)
    dataset_tag = "_".join(datasets)

    log_dir = create_log_directory(base_dir=os.path.join(args.log_base_dir, "imgflowdepth", args.datasets))
    print(f"[INFO] Created log directory: {log_dir}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    transform = build_transforms(args.img_size, args.flow_channels, args.flow_representation)
    train_ds, val_ds = build_joint_train_val_datasets(
        datasets, img_size=args.img_size, num_frames=args.num_frames, flow_channels=args.flow_channels,
        flow_representation=args.flow_representation, flow_offset=args.flow_offset,
        protocol=args.protocol, n_split=args.n_split, transform=transform,
    )
    print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                                collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                              collate_fn=collate_fn, pin_memory=True)

    cfg = vars(args).copy()
    cfg.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "modality": "imgflowdepth",
        "datasets": datasets,
        "dataset_tag": dataset_tag,
        "checkpoint_dir": os.path.join(log_dir, "checkpoints"),
        "FLOW_T": 20.0,
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "device": str(device),
        "log_dir": log_dir,
    })
    save_json(os.path.join(log_dir, "config.json"), cfg)

    model = build_model(cfg, device)
    cfg["model_name"] = model.__class__.__name__
    cfg["num_params"] = sum(p.numel() for p in model.parameters())
    save_json(os.path.join(log_dir, "config.json"), cfg)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    run_training_loop(model, train_loader, val_loader, optimizer, log_dir, cfg,
                       num_epochs=args.num_epochs, esp=args.esp)

    if args.eval:
        from evaluate_joint import evaluate_joint_from_config
        evaluate_joint_from_config(log_dir, device=device)


if __name__ == "__main__":
    main()
