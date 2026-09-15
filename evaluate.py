"""Evaluate a single-dataset run (teacher or KD student) from its saved config.json."""
import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import build_transforms, build_model, save_json, load_json, set_data_root, DATASET_ROOTS, oulu_list_path
from datasets import OULU_Combined, Generic_Combined, SIW_Dataset
from utils import collate_fn
from metrics import evaluate_all


def _build_test_dataset(cfg: dict):
    dataset = cfg["dataset"]
    img_size = int(cfg.get("img_size", 224))
    num_frames = int(cfg.get("num_frames", 1))
    flow_channels = int(cfg.get("flow_channels", 3))
    flow_representation = cfg.get("flow_representation", "hsv")
    flow_offset = int(cfg.get("flow_offset", 0))
    protocol = cfg.get("protocol", "all")
    n_split = cfg.get("n_split", "NA")

    transform = build_transforms(img_size, flow_channels, flow_representation)
    roots = DATASET_ROOTS[dataset]

    if dataset == "OULU":
        flist = oulu_list_path(roots["orig"], protocol, "test", n_split) if protocol not in (None, "all") else None
        return OULU_Combined(
            orig_root_dir=os.path.join(roots["orig"], "Test_files"), file_list_path=flist, protocol=protocol,
            flow_root_dir=os.path.join(roots["flow"], "Test_files"),
            depth_root_dir=os.path.join(roots["depth"], "Test_files"),
            flow_channels=flow_channels, flow_representation=flow_representation,
            transform=transform, num_frames=num_frames, is_train=False, flow_offset=flow_offset,
            return_access_type=True,
        )
    if dataset == "SIW":
        return SIW_Dataset(
            orig_root_dir=roots["orig"], flow_root_dir=roots["flow"], depth_root_dir=roots["depth"],
            transform=transform, num_frames=num_frames, is_train=False,
            flow_channels=flow_channels, flow_representation=flow_representation, flow_offset=flow_offset,
            protocol=protocol, split="test",
        )
    return Generic_Combined(
        orig_root_dir=os.path.join(roots["orig"], "test"),
        flow_root_dir=os.path.join(roots["flow"], "test"), depth_root_dir=os.path.join(roots["depth"], "test"),
        flow_channels=flow_channels, flow_representation=flow_representation,
        transform=transform, num_frames=num_frames, is_train=False, flow_offset=flow_offset,
    )


def evaluate_from_config(log_dir: str, device: torch.device = None) -> dict:
    cfg = load_json(os.path.join(log_dir, "config.json"))
    if device is None:
        device = torch.device(f'cuda:{cfg.get("gpu", 0)}' if torch.cuda.is_available() else "cpu")

    test_ds = _build_test_dataset(cfg)
    if len(test_ds) == 0:
        raise ValueError("Test dataset is empty. Check --data_root and the dataset layout.")

    test_loader = DataLoader(test_ds, batch_size=max(1, int(cfg.get("batch_size", 256))), shuffle=False,
                              collate_fn=collate_fn, pin_memory=True)

    model = build_model(cfg, device)
    ckpt_path = os.path.join(log_dir, "checkpoints", "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    kd = bool(cfg.get("kd", False))
    criterion = nn.CrossEntropyLoss()
    results = evaluate_all(model, test_loader, criterion, device, dataset=cfg["dataset"], kd_student=kd)

    with open(os.path.join(log_dir, "evaluation_log.txt"), "a") as f:
        f.write("\n--- Evaluation (from config) ---\n")
        f.write(f"Evaluated at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Test samples: {len(test_ds)}\n")
        f.write(f"Test Loss: {results['test_loss']:.8f}\n")
        f.write(f"Test Accuracy (%): {results['test_acc']:.8f}%\n")
        f.write(f"AUC-ROC: {results['auc_roc']:.8f}\n")
        f.write(f"EER: {results['eer']:.8f}\n")
        f.write(f"HTER (%): {results['hter'] * 100:.8f}\n")
        if cfg["dataset"] in ("OULU", "SIW") and results.get("acer") is not None:
            f.write(f"APCER (%): {results['apcer'] * 100:.8f}\n")
            f.write(f"BPCER (%): {results['bpcer'] * 100:.8f}\n")
            f.write(f"ACER (%): {results['acer'] * 100:.8f}\n")

    cfg.update({
        "test_len": len(test_ds),
        "post_eval": {k: float(v) for k, v in results.items() if isinstance(v, (int, float))},
        "last_evaluated_at": datetime.now().isoformat(timespec="seconds"),
    })
    save_json(os.path.join(log_dir, "config.json"), cfg)

    print("\n===== Evaluation Summary =====")
    print(f"Test samples: {len(test_ds)}")
    print(f"Loss: {results['test_loss']:.6f} | Acc: {results['test_acc']:.2f}% | "
          f"AUC: {results['auc_roc']:.4f} | EER: {results['eer']:.4f} | HTER: {results['hter'] * 100:.2f}%")
    if cfg["dataset"] in ("OULU", "SIW") and results.get("acer") is not None:
        print(f"APCER(%): {results['apcer'] * 100:.4f} | BPCER(%): {results['bpcer'] * 100:.4f} | "
              f"ACER(%): {results['acer'] * 100:.4f}%")
    print("================================\n")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a single-dataset run from its config.json")
    p.add_argument("--log_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data_root", type=str, default="data")
    args = p.parse_args()
    set_data_root(args.data_root)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    evaluate_from_config(args.log_dir, device=dev)
