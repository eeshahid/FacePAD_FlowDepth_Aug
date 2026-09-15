"""Shared dataset-path registry, transform/dataset/model builders used by every
training and evaluation script (individual-dataset, joint, and OCIM)."""
import os
import json
from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

from models import TripleBranchMobileNetV3LImgFlowDepth, StudentMobileNetV3
from datasets import OULU_Combined, Generic_Combined, SIW_Dataset
from utils import make_combined_transform_with_depth

# --------------------------------------------------------------------------------------
# Dataset path registry. Defaults assume a `data/<NAME>[_flow|_depth]` layout under the
# current working directory (see README for the expected internal structure of each);
# point `--data_root` at a different base directory to reuse an existing layout.
# --------------------------------------------------------------------------------------
DATASET_ROOTS: Dict[str, Dict[str, object]] = {
    "RA": {"orig": "data/RA", "flow": "data/RA_flow", "depth": "data/RA_depth"},
    "RM": {"orig": "data/RM", "flow": "data/RM_flow", "depth": "data/RM_depth"},
    "RY": {"orig": "data/RY", "flow": "data/RY_flow", "depth": "data/RY_depth"},
    "OULU": {"orig": "data/OULU", "flow": "data/OULU_flow", "depth": "data/OULU_depth"},
    "SIW": {"orig": "data/SIW", "flow": ["data/SIW_flow"], "depth": "data/SIW_depth"},
    "CASIA": {"orig": "data/CASIA", "flow": "data/CASIA_flow", "depth": "data/CASIA_depth"},
    "MSU": {"orig": "data/MSU", "flow": "data/MSU_flow", "depth": "data/MSU_depth"},
}
_DEFAULT_ROOT = "data"


def set_data_root(new_root: str) -> None:
    """Rewrite every DATASET_ROOTS entry to live under `new_root` instead of './data'."""
    if new_root == _DEFAULT_ROOT:
        return
    for paths in DATASET_ROOTS.values():
        for key in ("orig", "flow", "depth"):
            v = paths[key]
            if isinstance(v, list):
                paths[key] = [os.path.join(new_root, os.path.relpath(p, _DEFAULT_ROOT)) for p in v]
            else:
                paths[key] = os.path.join(new_root, os.path.relpath(v, _DEFAULT_ROOT))


NAME_TO_JOINT_DATASETS = ("RA", "RM", "RY")  # datasets supported by the joint RA+RM+RY pipeline


def parse_dataset_list(datasets: Union[Sequence[str], str]) -> List[str]:
    if isinstance(datasets, str):
        items = [x.strip() for x in datasets.split(",") if x.strip()]
    else:
        items = [str(x).strip() for x in datasets if str(x).strip()]
    if not items:
        raise ValueError("datasets list cannot be empty")
    return items


def build_transforms(img_size: int, flow_channels: int, flow_representation: str, flow_T: float = 20.0):
    """Single transform for the RGB+Flow+Depth concatenated tensor (adaptive-crop is
    applied by the caller before this; this only handles resize + per-modality normalize)."""
    return make_combined_transform_with_depth(
        img_size=img_size, flow_channels=flow_channels, flow_representation=flow_representation,
        depth_norm="standardize", flow_T=flow_T,
    )


def _oulu_split_dir(split: str) -> str:
    return {"train": "Train_files", "devel": "Dev_files", "test": "Test_files"}[split]


def oulu_list_path(orig_root_dir: str, protocol: str, split: str, n_split: str) -> str:
    """Path to the OULU-NPU protocol file list for a given split ("train"/"devel"/"test")."""
    split_name = {"train": "Train", "devel": "Dev", "test": "Test"}[split]
    base = os.path.join(orig_root_dir, "protocols", f"Protocol_{protocol}")
    if protocol in {"3", "4"}:
        return os.path.join(base, f"{split_name}_{n_split}.txt")
    return os.path.join(base, f"{split_name}.txt")


def build_dataset(dataset: str, split: str, *, img_size: int, num_frames: int,
                   flow_channels: int, flow_representation: str, flow_offset: int,
                   protocol: str, n_split: str, transform):
    """One RGB+Flow+Depth dataset for a given dataset name and split
    (split in {"train", "devel"/"val", "test"})."""
    roots = DATASET_ROOTS[dataset]
    is_train = split == "train"

    if dataset == "OULU":
        subdir = _oulu_split_dir(split)
        flist = oulu_list_path(roots["orig"], protocol, split, n_split) if protocol not in (None, "all") else None
        return OULU_Combined(
            orig_root_dir=os.path.join(roots["orig"], subdir), file_list_path=flist, protocol=protocol,
            flow_root_dir=os.path.join(roots["flow"], subdir), depth_root_dir=os.path.join(roots["depth"], subdir),
            flow_channels=flow_channels, flow_representation=flow_representation,
            transform=transform, num_frames=num_frames, is_train=is_train, flow_offset=flow_offset,
        )

    if dataset == "SIW":
        return SIW_Dataset(
            orig_root_dir=roots["orig"], flow_root_dir=roots["flow"], depth_root_dir=roots["depth"],
            transform=transform, num_frames=num_frames, is_train=is_train,
            flow_channels=flow_channels, flow_representation=flow_representation, flow_offset=flow_offset,
            protocol=protocol, split=("val" if split == "devel" else split),
        )

    if dataset in ("RA", "RM", "RY"):
        return Generic_Combined(
            orig_root_dir=os.path.join(roots["orig"], split),
            flow_root_dir=os.path.join(roots["flow"], split), depth_root_dir=os.path.join(roots["depth"], split),
            flow_channels=flow_channels, flow_representation=flow_representation,
            transform=transform, num_frames=num_frames, is_train=is_train, flow_offset=flow_offset,
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_joint_train_val_datasets(datasets, *, img_size: int, num_frames: int, flow_channels: int,
                                    flow_representation: str, flow_offset: int, protocol: str, n_split: str,
                                    transform):
    dataset_list = parse_dataset_list(datasets)
    kwargs = dict(img_size=img_size, num_frames=num_frames, flow_channels=flow_channels,
                  flow_representation=flow_representation, flow_offset=flow_offset,
                  protocol=protocol, n_split=n_split, transform=transform)
    train_sets = [build_dataset(ds, "train", **kwargs) for ds in dataset_list]
    val_sets = [build_dataset(ds, "devel", **kwargs) for ds in dataset_list]
    return ConcatDataset(train_sets), ConcatDataset(val_sets)


def build_joint_test_sets(datasets, *, img_size: int, num_frames: int, flow_channels: int,
                           flow_representation: str, flow_offset: int, protocol: str, n_split: str, transform):
    dataset_list = parse_dataset_list(datasets)
    kwargs = dict(img_size=img_size, num_frames=num_frames, flow_channels=flow_channels,
                  flow_representation=flow_representation, flow_offset=flow_offset,
                  protocol=protocol, n_split=n_split, transform=transform)
    return {ds: build_dataset(ds, "test", **kwargs) for ds in dataset_list}


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """Build the IFD teacher, or (cfg["kd"]=True) the RGB-only KD student."""
    flow_channels = int(cfg.get("flow_channels", 3))
    pretrained = bool(cfg.get("pretrained", True))

    if bool(cfg.get("kd", False)):
        model = StudentMobileNetV3(
            num_classes=2, projector_dim=int(cfg.get("student_projector_dim") or 0), pretrained=True,
        )
        return model.to(device)

    model = TripleBranchMobileNetV3LImgFlowDepth(
        num_classes=2, flow_channels=flow_channels, depth_channels=1, pretrained=pretrained,
    )
    return model.to(device)


def _jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return _jsonable(obj.detach().cpu().numpy())
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        try:
            return _jsonable(obj.tolist())
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return str(obj)


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=True)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
