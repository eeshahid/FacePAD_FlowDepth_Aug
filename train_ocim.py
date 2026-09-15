"""Leave-one-dataset-out (LOO) IFD teacher training on the OCIM benchmark.

Domains: O = OULU-NPU, C = CASIA-MFSD, I = Replay-Attack, M = MSU-MFSD.
Protocols (train on the other three, test on the held-out target's TEST split):
    C&I&M -> O   (--target O)
    O&I&M -> C   (--target C)
    O&C&M -> I   (--target I)
    O&C&I -> M   (--target M)
"""
import os
import argparse
from datetime import datetime

import torch
import torch.optim as optim

from common import build_transforms, build_model, save_json, set_data_root
from utils import collate_fn, create_log_directory
from engine import run_training_loop
from datasets_ocim import DOMAIN_NAMES, LOO_SOURCES, VAL_SOURCES, parse_domain_list, build_sources_train_val


def parse_args():
    p = argparse.ArgumentParser(description="LOO-OCIM IFD teacher training")
    p.add_argument("--target", type=str, required=True, choices=["O", "C", "I", "M"],
                    help="Leave-out domain: trained on the other three, evaluated on this one's test split")
    p.add_argument("--sources", type=str, default=None,
                    help="Override: comma list of domains to train on (default: the LOO complement of --target)")
    p.add_argument("--eval_targets", type=str, default=None,
                    help="Override: comma list of domains to evaluate on (default: just --target)")
    p.add_argument("--val_source", type=str, default="target", choices=list(VAL_SOURCES))
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
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval", action="store_true", default=True)
    p.add_argument("--no-eval", dest="eval", action="store_false")
    p.add_argument("--log_base_dir", type=str, default="./logs_loo_ocim")
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
    _seed_everything(args.seed)

    target = args.target.upper()
    sources = parse_domain_list(args.sources) if args.sources else LOO_SOURCES[target]
    eval_targets = parse_domain_list(args.eval_targets) if args.eval_targets else [target]
    protocol_tag = "".join(sources) + "_to_" + "".join(eval_targets)
    proto_dir = target if eval_targets == [target] and not args.sources else protocol_tag

    log_dir = create_log_directory(base_dir=os.path.join(args.log_base_dir, "imgflowdepth", proto_dir, args.val_source))
    print(f"[INFO] OCIM protocol: {'&'.join(sources)} -> {'&'.join(eval_targets)} "
          f"({', '.join(DOMAIN_NAMES[t] for t in eval_targets)})")
    print(f"[INFO] Created log directory: {log_dir}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    transform = build_transforms(args.img_size, args.flow_channels, args.flow_representation)
    train_ds, val_ds = build_sources_train_val(
        sources, args.val_source, img_size=args.img_size, num_frames=args.num_frames,
        flow_channels=args.flow_channels, flow_representation=args.flow_representation,
        flow_offset=args.flow_offset, transform=transform, val_targets=eval_targets, seed=args.seed,
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                                collate_fn=collate_fn, pin_memory=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                              collate_fn=collate_fn, pin_memory=True, num_workers=args.num_workers)

    cfg = vars(args).copy()
    cfg.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "modality": "imgflowdepth",
        "target": target,
        "sources": sources,
        "eval_targets": eval_targets,
        "protocol_tag": protocol_tag,
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
        from evaluate_ocim import evaluate_loo_from_config
        evaluate_loo_from_config(log_dir, device=device)


if __name__ == "__main__":
    main()
