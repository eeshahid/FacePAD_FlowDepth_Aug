"""OCIM leave-one-dataset-out (LOO) dataset assembly for the RGB+Flow+Depth (IFD) pipeline.

Domain codes: O = OULU-NPU, C = CASIA-MFSD, I = Replay-Attack (a.k.a. "RA" elsewhere
in this repo), M = MSU-MFSD.

Reuses the dataset classes in `datasets.py`; only adds the MSU subject-list filtering
and the source/target assembly logic specific to the OCIM protocol.
"""
import os
import random
from typing import Dict, List, Optional, Tuple

from torch.utils.data import ConcatDataset, Subset

from datasets import Generic_Combined, OULU_Combined
from common import DATASET_ROOTS

DOMAIN_NAMES: Dict[str, str] = {"O": "OULU", "C": "CASIA", "I": "RA", "M": "MSU"}
ALL_DOMAINS: Tuple[str, ...] = ("O", "C", "I", "M")
LOO_SOURCES: Dict[str, List[str]] = {t: [d for d in ALL_DOMAINS if d != t] for t in ALL_DOMAINS}
HAS_NATIVE_VAL: Tuple[str, ...] = ("O", "I")
VAL_SOURCES: Tuple[str, ...] = ("source_split", "source_native", "target")
DEFAULT_SEED = 1234


def parse_domain_list(spec) -> List[str]:
    if isinstance(spec, str):
        items = [x.strip().upper() for x in spec.replace(",", " ").split() if x.strip()]
    else:
        items = [str(x).strip().upper() for x in spec if str(x).strip()]
    for d in items:
        if d not in ALL_DOMAINS:
            raise ValueError(f"Unknown OCIM domain '{d}'. Valid: {ALL_DOMAINS}")
    return items


def _stratified_80_20(samples, seed: int) -> Tuple[List[int], List[int]]:
    """Class-stratified 80/20 index split over `samples` (list of tuples, label at [1])."""
    by_cls: Dict[int, List[int]] = {}
    for i, s in enumerate(samples):
        by_cls.setdefault(int(s[1]), []).append(i)
    rng = random.Random(seed)
    tr, va = [], []
    for cls in sorted(by_cls):
        idxs = list(by_cls[cls])
        rng.shuffle(idxs)
        k = int(round(0.8 * len(idxs)))
        tr.extend(idxs[:k])
        va.extend(idxs[k:])
    return sorted(tr), sorted(va)


# --------------------------------------------------------------------------------------
# MSU-MFSD: subject-list filtering on top of Generic_Combined
# --------------------------------------------------------------------------------------

def _msu_scene_root(root: str) -> str:
    if os.path.isdir(os.path.join(root, "scene01")):
        return os.path.join(root, "scene01")
    return root


def _msu_subject_of(video_path: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    for part in stem.split("_"):
        if part.startswith("client"):
            try:
                return str(int(part.replace("client", "")))
            except ValueError:
                return None
    return None


def _msu_read_subjects(path: str) -> set:
    with open(path, "r") as f:
        return {str(int(line.strip().replace("client", ""))) for line in f if line.strip()}


class MSU_Combined(Generic_Combined):
    """MSU-MFSD RGB+Flow+Depth, filtered by the official train/test subject lists.

    msu_split in {"train", "test", "train80", "val20"}:
        train   -> all videos from train_sub_list subjects
        test    -> all videos from test_sub_list subjects
        train80 -> class-stratified 80% of the train-subject videos (seed controlled)
        val20   -> the remaining class-stratified 20% (MSU has no native validation split)
    """

    def __init__(self, msu_root: str, msu_split: str, msu_seed: int = DEFAULT_SEED, **kwargs):
        self._msu_root = msu_root
        self._msu_split = msu_split
        self._msu_seed = msu_seed
        kwargs.pop("orig_root_dir", None)
        super().__init__(orig_root_dir=_msu_scene_root(msu_root), **kwargs)

    def _load_samples(self):
        samples = super()._load_samples()
        train_subs = _msu_read_subjects(os.path.join(self._msu_root, "train_sub_list.txt"))
        test_subs = _msu_read_subjects(os.path.join(self._msu_root, "test_sub_list.txt"))

        allowed = test_subs if self._msu_split == "test" else train_subs
        selected = sorted(s for s in samples if _msu_subject_of(s[0]) in allowed)
        if self._msu_split in ("train", "test"):
            return selected

        tr_idx, va_idx = _stratified_80_20(selected, self._msu_seed)
        keep = tr_idx if self._msu_split == "train80" else va_idx
        return [selected[i] for i in keep]


# --------------------------------------------------------------------------------------
# per-domain dataset builder (imgflowdepth only)
# --------------------------------------------------------------------------------------

def build_ocim_domain_dataset(domain: str, split: str, *,
                               img_size: int, num_frames: int, flow_channels: int,
                               flow_representation: str, flow_offset: int, transform,
                               seed: int = DEFAULT_SEED):
    """One RGB+Flow+Depth dataset for `domain` / `split`.

    split: one of {train, val, test, train80, val20} (train80/val20 only meaningful
    for C and M; O/I use native train/val/test).
    """
    domain = domain.upper()
    is_train = split in ("train", "train80")

    if domain == "O":
        roots = DATASET_ROOTS["OULU"]
        subdir = {"train": "Train_files", "val": "Dev_files", "test": "Test_files"}[split]
        return OULU_Combined(
            orig_root_dir=os.path.join(roots["orig"], subdir),
            file_list_path=None, protocol="all",
            flow_root_dir=os.path.join(roots["flow"], subdir),
            depth_root_dir=os.path.join(roots["depth"], subdir),
            flow_channels=flow_channels, flow_representation=flow_representation,
            transform=transform, num_frames=num_frames, is_train=is_train,
            flow_offset=flow_offset,
        )

    if domain == "M":
        roots = DATASET_ROOTS["MSU"]
        return MSU_Combined(
            msu_root=roots["orig"], msu_split=split, msu_seed=seed,
            flow_root_dir=roots["flow"], depth_root_dir=roots["depth"],
            flow_channels=flow_channels, flow_representation=flow_representation,
            transform=transform, num_frames=num_frames, is_train=is_train,
            flow_offset=flow_offset,
        )

    if domain == "I":
        roots = DATASET_ROOTS["RA"]
        subdir = {"train": "train", "val": "devel", "test": "test"}[split]
    else:  # C (CASIA): only train/test exist; train80/val20 derive from train/
        roots = DATASET_ROOTS["CASIA"]
        if split == "val":
            raise ValueError("CASIA (C) has no native 'val' split; use 'train80'/'val20'.")
        subdir = "test" if split == "test" else "train"

    ds = Generic_Combined(
        orig_root_dir=os.path.join(roots["orig"], subdir),
        flow_root_dir=os.path.join(roots["flow"], subdir),
        depth_root_dir=os.path.join(roots["depth"], subdir),
        flow_channels=flow_channels, flow_representation=flow_representation,
        transform=transform, num_frames=num_frames, is_train=is_train,
        flow_offset=flow_offset,
    )
    if domain == "C" and split in ("train80", "val20"):
        tr_idx, va_idx = _stratified_80_20(ds.samples, seed)
        ds = Subset(ds, tr_idx if split == "train80" else va_idx)
    return ds


# --------------------------------------------------------------------------------------
# LOO train / val / test assembly
# --------------------------------------------------------------------------------------

def _domain_train_split(domain: str, val_source: str) -> str:
    if val_source == "source_split" and domain in ("C", "M"):
        return "train80"
    return "train"


def _held_out_split(domain: str) -> str:
    return "val" if domain in HAS_NATIVE_VAL else "train"


def build_sources_train_val(sources: List[str], val_source: str, *,
                             img_size: int, num_frames: int, flow_channels: int,
                             flow_representation: str, flow_offset: int, transform,
                             val_targets: Optional[List[str]] = None, seed: int = DEFAULT_SEED):
    """(train_dataset, val_dataset) over the given source domains.

    val_source:
      - "source_split"  : per source -> train80/val20 (C,M) or train/val (O,I)
      - "source_native" : per source -> train ; val = native val of sources that have one
      - "target"        : per source -> train ; val = held-out split of `val_targets`
                          (O/I -> native val, C/M -> train), concatenated if several
    """
    sources = [s.upper() for s in sources]
    if val_source not in VAL_SOURCES:
        raise ValueError(f"val_source must be one of {VAL_SOURCES}")

    def _build(domain, split):
        return build_ocim_domain_dataset(
            domain, split, img_size=img_size, num_frames=num_frames, flow_channels=flow_channels,
            flow_representation=flow_representation, flow_offset=flow_offset,
            transform=transform, seed=seed,
        )

    train_sets = [_build(d, _domain_train_split(d, val_source)) for d in sources]

    if val_source == "source_split":
        val_ds = ConcatDataset([_build(d, "val20" if d in ("C", "M") else "val") for d in sources])
    elif val_source == "source_native":
        val_sets = [_build(d, "val") for d in sources if d in HAS_NATIVE_VAL]
        if not val_sets:
            raise RuntimeError(f"val_source='source_native' but none of {sources} has a native val split.")
        val_ds = ConcatDataset(val_sets)
    else:  # "target"
        if not val_targets:
            raise ValueError("val_source='target' requires val_targets.")
        val_sets = [_build(t.upper(), _held_out_split(t.upper())) for t in val_targets]
        val_ds = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)

    return ConcatDataset(train_sets), val_ds


def build_loo_train_val(target: str, val_source: str, *,
                         img_size: int, num_frames: int, flow_channels: int,
                         flow_representation: str, flow_offset: int, transform,
                         seed: int = DEFAULT_SEED):
    """LOO wrapper: sources = the other three domains, validation target = `target`."""
    target = target.upper()
    return build_sources_train_val(
        LOO_SOURCES[target], val_source,
        img_size=img_size, num_frames=num_frames, flow_channels=flow_channels,
        flow_representation=flow_representation, flow_offset=flow_offset,
        transform=transform, val_targets=[target], seed=seed,
    )


def build_loo_test(target: str, *,
                    img_size: int, num_frames: int, flow_channels: int,
                    flow_representation: str, flow_offset: int, transform,
                    seed: int = DEFAULT_SEED):
    """(test_dataset, eval_tag) for the leave-out domain's official TEST split.

    `eval_tag` is the bare domain code ("O"/"C"/"I"/"M"); it is deliberately not
    "OULU"/"SIW" so `evaluate_all` uses the generic metric path (no access-type /
    video-id triplets, which these OCIM loaders don't produce).
    """
    target = target.upper()
    ds = build_ocim_domain_dataset(
        target, "test", img_size=img_size, num_frames=num_frames, flow_channels=flow_channels,
        flow_representation=flow_representation, flow_offset=flow_offset,
        transform=transform, seed=seed,
    )
    return ds, target
