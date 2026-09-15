"""Train the RGB+Flow+Depth (IFD) teacher on a single dataset: RA, RM, RY, OULU, or SIW."""
import os
import argparse
from datetime import datetime

import torch
import torch.optim as optim

from common import build_transforms, build_dataset, build_model, save_json, set_data_root
from utils import collate_fn, create_log_directory
from engine import run_training_loop


def parse_args():
    p = argparse.ArgumentParser(description="Train the IFD (RGB+Flow+Depth) teacher on a single dataset")
    p.add_argument("--dataset", type=str, required=True, choices=["RA", "RM", "RY", "OULU", "SIW"])
    p.add_argument("--data_root", type=str, default="data", help="Base directory for the data/ layout (see README)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_frames", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--esp", type=int, default=50, help="Early stopping patience (epochs)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--flow_channels", type=int, default=3, choices=[2, 3])
    p.add_argument("--flow_representation", type=str, default="hsv", choices=["uv", "uvm", "hsv"])
    p.add_argument("--flow_offset", type=int, default=0,
                    help="Pair RGB frame t with the flow at (t+flow_offset)->(t+flow_offset+1)")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--protocol", type=str, default="all", help="OULU/SIW protocol ('all', '1'-'4' for OULU; '1' for SIW)")
    p.add_argument("--n_split", type=str, default="NA", help="OULU protocol 3/4 split id")
    p.add_argument("--eval", action="store_true", default=True)
    p.add_argument("--no-eval", dest="eval", action="store_false")
    p.add_argument("--log_base_dir", type=str, default="./logs")
    p.add_argument("--exp_description", type=str, default="IFD teacher training")
    return p.parse_args()


def main():
    args = parse_args()
    set_data_root(args.data_root)

    log_root_dir = os.path.join(args.log_base_dir, args.dataset, "imgflowdepth")
    log_dir = create_log_directory(base_dir=log_root_dir)
    args.log_dir = log_dir
    print(f"[INFO] Created log directory: {log_dir}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    transform = build_transforms(args.img_size, args.flow_channels, args.flow_representation)
    ds_kwargs = dict(img_size=args.img_size, num_frames=args.num_frames, flow_channels=args.flow_channels,
                      flow_representation=args.flow_representation, flow_offset=args.flow_offset,
                      protocol=args.protocol, n_split=args.n_split, transform=transform)

    train_ds = build_dataset(args.dataset, "train", **ds_kwargs)
    val_ds = build_dataset(args.dataset, "devel", **ds_kwargs)
    print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                                collate_fn=collate_fn, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                              collate_fn=collate_fn, pin_memory=True)

    cfg = vars(args).copy()
    cfg.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "modality": "imgflowdepth",
        "model": "DB",
        "backbone": "mobilenet_v3_large",
        "checkpoint_dir": os.path.join(log_dir, "checkpoints"),
        "FLOW_T": 20.0,
        "classes_order": ["attack", "real"],
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "device": str(device),
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
        print("\n[Eval] Calling evaluate.py entry from config...")
        from evaluate import evaluate_from_config
        evaluate_from_config(log_dir, device=device)
        print("[Eval] Done.")


if __name__ == "__main__":
    main()
