"""Evaluate an OCIM run strictly on the official TEST split(s) of its eval target(s)."""
import os
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import build_transforms, build_model, load_json, save_json, set_data_root
from utils import collate_fn
from metrics import evaluate_all
from datasets_ocim import build_loo_test, DOMAIN_NAMES, LOO_SOURCES

_METRIC_KEYS = ("test_loss", "test_acc", "auc_roc", "eer", "hter", "far", "frr",
                "youdens_index", "optimal_threshold", "avg_inference_time")


def evaluate_loo_from_config(log_dir: str, device: torch.device = None, split: str = "test") -> dict:
    cfg = load_json(os.path.join(log_dir, "config.json"))
    if device is None:
        device = torch.device(f'cuda:{cfg.get("gpu", 0)}' if torch.cuda.is_available() else "cpu")

    target = str(cfg["target"]).upper()
    sources = [s.upper() for s in cfg.get("sources", LOO_SOURCES[target])]
    eval_targets = [t.upper() for t in cfg.get("eval_targets", [target])]
    kd = bool(cfg.get("kd", False))

    img_size = int(cfg.get("img_size", 224))
    num_frames = int(cfg.get("num_frames", 1))
    flow_channels = int(cfg.get("flow_channels", 3))
    flow_representation = cfg.get("flow_representation", "hsv")
    flow_offset = int(cfg.get("flow_offset", 0))
    seed = int(cfg.get("seed", 1234))

    transform = build_transforms(img_size, flow_channels, flow_representation)

    model = build_model(cfg, device)
    ckpt = torch.load(os.path.join(log_dir, "checkpoints", "best_model.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    per_target = {}
    for tgt in eval_targets:
        test_ds, eval_tag = build_loo_test(
            tgt, img_size=img_size, num_frames=num_frames, flow_channels=flow_channels,
            flow_representation=flow_representation, flow_offset=flow_offset, transform=transform, seed=seed,
        )
        loader = DataLoader(test_ds, batch_size=int(cfg.get("batch_size", 64)), shuffle=False,
                             collate_fn=collate_fn, pin_memory=True, num_workers=int(cfg.get("num_workers", 4)))
        print(f"[EVAL{' KD' if kd else ''}] {'&'.join(sources)} -> {tgt} ({DOMAIN_NAMES[tgt]}) | test samples: {len(test_ds)}")
        results = evaluate_all(model, loader, criterion, device, dataset=eval_tag, kd_student=kd)
        per_target[tgt] = {
            "n_test": len(test_ds), "eval_tag": eval_tag,
            "metrics": {k: v for k, v in results.items() if k not in ("fpr", "tpr", "labels", "probs")},
        }

    mean_metrics = {}
    if len(per_target) > 1:
        for k in _METRIC_KEYS:
            vals = [float(pt["metrics"][k]) for pt in per_target.values()
                    if isinstance(pt["metrics"].get(k), (int, float, np.number))]
            if vals:
                mean_metrics[k] = float(np.mean(vals))

    summary = {
        "log_dir": log_dir, "task": "loo_ocim_kd" if kd else "loo_ocim", "kd": kd,
        "target": target, "sources": sources, "eval_targets": eval_targets, "seed": seed,
        "per_target": per_target, "mean_metrics": mean_metrics,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if len(eval_targets) == 1:
        only = per_target[eval_targets[0]]
        summary.update({"eval_tag": only["eval_tag"], "n_test": only["n_test"], "metrics": only["metrics"]})

    save_json(os.path.join(log_dir, f"loo_ocim_eval_{split}.json"), summary)
    with open(os.path.join(log_dir, f"loo_ocim_eval_{split}.txt"), "w") as f:
        f.write(json.dumps(summary, indent=2, sort_keys=True, default=str))

    print("\n===== OCIM Evaluation Summary =====")
    print(f"Sources : {'&'.join(sources)}")
    for tgt, pt in per_target.items():
        m = pt["metrics"]
        print(f"  {tgt} ({DOMAIN_NAMES[tgt]:>5}) n={pt['n_test']:5d} | Acc {m['test_acc']:6.2f}% | "
              f"AUC {m['auc_roc']:.4f} | EER {m['eer']:.4f} | HTER {m['hter'] * 100:6.2f}%")
    if mean_metrics:
        print(f"  MEAN         | Acc {mean_metrics['test_acc']:6.2f}% | AUC {mean_metrics['auc_roc']:.4f} | "
              f"EER {mean_metrics['eer']:.4f} | HTER {mean_metrics['hter'] * 100:6.2f}%")
    print("==================================\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a LOO-OCIM run on the target TEST split")
    p.add_argument("--log_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--data_root", type=str, default="data")
    args = p.parse_args()
    set_data_root(args.data_root)
    evaluate_loo_from_config(args.log_dir,
                              device=torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"),
                              split=args.split)
