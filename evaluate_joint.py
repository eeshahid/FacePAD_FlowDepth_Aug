"""Evaluate a joint (multi-dataset) run on each dataset's own test split, plus the mean."""
import os
import json
import argparse
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import build_transforms, build_joint_test_sets, build_model, load_json, save_json, parse_dataset_list, set_data_root
from utils import collate_fn
from metrics import evaluate_all


def _merge_numeric_mean(results_by_ds: Dict[str, dict]) -> dict:
    keys = ["test_loss", "test_acc", "auc_roc", "eer", "hter", "far", "frr", "youdens_index", "avg_inference_time"]
    out = {}
    for k in keys:
        vals = [float(res[k]) for res in results_by_ds.values() if isinstance(res.get(k), (int, float, np.number))]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def evaluate_joint_from_config(log_dir: str, device: torch.device = None, split: str = "test") -> dict:
    cfg = load_json(os.path.join(log_dir, "config.json"))
    if device is None:
        device = torch.device(f'cuda:{cfg.get("gpu", 0)}' if torch.cuda.is_available() else "cpu")

    datasets = parse_dataset_list(cfg.get("datasets", cfg.get("dataset", "RA,RM,RY")))
    img_size = int(cfg.get("img_size", 224))
    num_frames = int(cfg.get("num_frames", 1))
    flow_channels = int(cfg.get("flow_channels", 3))
    flow_representation = cfg.get("flow_representation", "hsv")
    flow_offset = int(cfg.get("flow_offset", 0))
    protocol = cfg.get("protocol", "all")
    n_split = cfg.get("n_split", "NA")
    kd = bool(cfg.get("kd", False))

    transform = build_transforms(img_size, flow_channels, flow_representation)
    test_sets = build_joint_test_sets(
        datasets, img_size=img_size, num_frames=num_frames, flow_channels=flow_channels,
        flow_representation=flow_representation, flow_offset=flow_offset,
        protocol=protocol, n_split=n_split, transform=transform,
    )

    model = build_model(cfg, device)
    ckpt = torch.load(os.path.join(log_dir, "checkpoints", "best_model.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    per_dataset = {}
    for ds_name, ds in test_sets.items():
        loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 256)), shuffle=False,
                             collate_fn=collate_fn, pin_memory=True)
        print(f"[EVAL] {ds_name} ({split}) samples: {len(ds)}")
        per_dataset[ds_name] = evaluate_all(model, loader, criterion, device, dataset=ds_name, kd_student=kd)

    mean_metrics = _merge_numeric_mean(per_dataset)
    summary = {
        "log_dir": log_dir, "datasets": datasets, "per_dataset": per_dataset,
        "mean_metrics": mean_metrics, "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(os.path.join(log_dir, f"joint_eval_{split}.json"), summary)
    with open(os.path.join(log_dir, f"joint_eval_{split}.txt"), "w") as f:
        f.write(json.dumps(summary, indent=2, sort_keys=True, default=str))

    print("\n===== Joint Evaluation Summary =====")
    for ds_name, res in per_dataset.items():
        print(f"{ds_name}: Acc={res['test_acc']:.2f}% | AUC={res['auc_roc']:.4f} | "
              f"EER={res['eer']:.4f} | HTER={res['hter'] * 100:.2f}%")
    if mean_metrics:
        print(f"Mean: Acc={mean_metrics['test_acc']:.2f}% | AUC={mean_metrics['auc_roc']:.4f} | "
              f"EER={mean_metrics['eer']:.4f} | HTER={mean_metrics['hter'] * 100:.2f}%")
    print("===================================\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a joint multi-dataset training run")
    p.add_argument("--log_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--data_root", type=str, default="data")
    args = p.parse_args()
    set_data_root(args.data_root)
    evaluate_joint_from_config(args.log_dir, device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"), split=args.split)
