"""Dataset loaders for the RGB+Flow+Depth (IFD) pipeline.

All three modalities are read from precomputed caches:
  - RGB frames are decoded directly from the source video.
  - Optical flow is read from a mirrored directory of per-frame-pair `.flo` files
    (produced upstream by an optical-flow model, e.g. UniMatch/GMFlow).
  - Depth is read from a mirrored grayscale "depth video" (produced upstream by a
    monocular depth model, e.g. Depth Anything V2).

Every dataset yields `(frames, label)` with `frames: (num_frames, 3+flow_channels+depth_channels, H, W)`
and `label` in {0=attack, 1=real}, except OULU_Combined with `return_access_type=True`,
which additionally returns the per-sample OULU attack-type code (needed for APCER/BPCER/ACER).
"""
import os
import cv2
import glob
import math
import random
import numpy as np
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torch.nn.functional as TF


# ================================
# Low-level IO / geometry helpers
# ================================

def read_flo(path: str) -> np.ndarray:
    """Read a Middlebury .flo file -> ndarray (H, W, 2) float32."""
    if not os.path.exists(path) or os.path.getsize(path) < 12:
        raise ValueError(f"{path}: file missing or too small to be a .flo")
    with open(path, 'rb') as f:
        magic = np.frombuffer(f.read(4), np.float32)[0]
        if magic != 202021.25:
            raise ValueError(f"{path}: wrong magic {magic}")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        expected = 4 + 4 + 4 + (w * h * 2 * 4)
        if os.path.getsize(path) != expected:
            raise ValueError(f"{path}: incorrect size ({os.path.getsize(path)} bytes), expected {expected}")
        data = np.fromfile(f, np.float32, count=2 * w * h)
    return data.reshape(h, w, 2)


def natural_sort(file_list: List[str]) -> List[str]:
    import re
    def key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]
    return sorted(file_list, key=key)


def rel_path_without_ext(full_path: str, root_dir: str) -> str:
    rel = os.path.relpath(full_path, root_dir)
    return os.path.splitext(rel)[0]


def flow_dir_for_video(video_full_path: str, orig_root_dir: str, flow_root_dir: str) -> str:
    return os.path.join(flow_root_dir, rel_path_without_ext(video_full_path, orig_root_dir))


def list_flo_files(flow_dir: str) -> List[str]:
    if not os.path.isdir(flow_dir):
        return []
    return natural_sort(glob.glob(os.path.join(flow_dir, '*.flo')))


def resize_like(t: torch.Tensor, size_hw: Tuple[int, int], is_flow: bool = False) -> torch.Tensor:
    """Resize (C,H,W) to size_hw; for flow, also rescale vector magnitudes by the resize ratio."""
    Ht, Wt = t.shape[-2:]
    H, W = size_hw
    if (Ht, Wt) == (H, W):
        return t
    out = TF.interpolate(t.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
    if is_flow:
        out[0] *= W / float(Wt)
        out[1] *= H / float(Ht)
    return out


def rotate_scale_flow_vectors(uv: torch.Tensor, angle_deg: float, scale: float) -> torch.Tensor:
    """Rotate/scale the (u,v) vector components themselves (the spatial grid is rotated separately)."""
    if angle_deg == 0 and scale == 1:
        return uv
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    u, v = uv[0], uv[1]
    u2 = c * u - s * v
    v2 = s * u + c * v
    if scale != 1:
        u2, v2 = u2 * scale, v2 * scale
    return torch.stack([u2, v2], dim=0)


def flow_uv_to_uvm(uv: torch.Tensor) -> torch.Tensor:
    """(2,H,W) -> (3,H,W): (u, v, |flow|)."""
    u, v = uv[0], uv[1]
    return torch.stack([u, v, torch.sqrt(u * u + v * v)], dim=0)


def flow_uv_to_hsv_rgb(uv: torch.Tensor, max_mag: float = 20.0) -> torch.Tensor:
    """(2,H,W) -> (3,H,W) RGB via an HSV color wheel: hue=angle, saturation=mag/max_mag, value=1."""
    u, v = uv[0], uv[1]
    angle = torch.atan2(v, u)
    hue = (angle + math.pi) / (2 * math.pi)
    mag = torch.sqrt(u * u + v * v)
    sat = torch.clamp(mag / max_mag, 0.0, 1.0)
    val = torch.ones_like(sat)

    h6 = hue * 6.0
    i = torch.floor(h6).to(torch.int32) % 6
    f = h6 - i.float()
    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)

    r = torch.where(i == 0, val, torch.where(i == 1, q, torch.where(i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, val)))))
    g = torch.where(i == 0, t, torch.where(i == 1, val, torch.where(i == 2, val, torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t, torch.where(i == 3, val, torch.where(i == 4, val, q)))))
    return torch.stack([r, g, b], dim=0)


def choose_start(total_frames: int, num_frames: int, is_train: bool) -> int:
    """Random start frame during training, deterministic (0) at inference."""
    if is_train and total_frames >= num_frames:
        return np.random.randint(0, max(1, total_frames - num_frames + 1))
    return 0


def frame_indices_from(start: int, num_frames: int) -> np.ndarray:
    return np.linspace(start, start + num_frames - 1, num_frames, dtype=int)


def load_rgb_frames(video_path: str, frame_indices: np.ndarray, fallback_hw=(224, 224)) -> List[torch.Tensor]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(F.to_tensor(frame))
    cap.release()
    if not frames:
        H, W = fallback_hw
        frames = [torch.zeros(3, H, W, dtype=torch.float32)]
    while len(frames) < len(frame_indices):
        frames.append(frames[-1])
    return frames


def depth_video_for_video(video_full_path: str, orig_root_dir: str, depth_root_dir: str) -> Optional[str]:
    """Map an RGB video path to its cached depth video by mirroring the relative path."""
    rel_wo_ext = rel_path_without_ext(video_full_path, orig_root_dir)
    for ext in ('.mp4', '.avi', '.mov', '.mkv'):
        cand = os.path.join(depth_root_dir, rel_wo_ext + ext)
        if os.path.exists(cand):
            return cand
    return None


def load_depth_frames(video_path: str, frame_indices: np.ndarray, fallback_hw=(224, 224)) -> List[torch.Tensor]:
    """Load grayscale depth frames from a cached depth video; returns (1,H,W) tensors in [0,1]."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening depth video: {video_path}")
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(torch.from_numpy(gray).float().unsqueeze(0) / 255.0)
    cap.release()
    if not frames:
        H, W = fallback_hw
        frames = [torch.zeros(1, H, W, dtype=torch.float32)]
    while len(frames) < len(frame_indices):
        frames.append(frames[-1])
    return frames


def _rand_aug():
    angle = random.uniform(-180, 180) if random.random() > 0.5 else 0.0
    scale = random.uniform(0.7, 1.3) if random.random() > 0.5 else 1.0
    return angle, scale


def _aug_img(x: torch.Tensor, angle: float, scale: float) -> torch.Tensor:
    if angle != 0:
        x = F.rotate(x, angle)
    if scale != 1:
        x = F.affine(x, angle=0, translate=(0, 0), scale=scale, shear=0)
    return x


def _flow_seq_to_channels(uv_seq: List[torch.Tensor], flow_channels: int, flow_representation: str,
                           hsv_max_mag: float) -> List[torch.Tensor]:
    if flow_channels == 2:
        return uv_seq
    if flow_representation == "uvm":
        return [flow_uv_to_uvm(uv) for uv in uv_seq]
    return [flow_uv_to_hsv_rgb(uv, hsv_max_mag) for uv in uv_seq]


# ================================
# OULU-NPU: RGB + Flow + Depth
# ================================

class OULU_Combined(Dataset):
    """OULU-NPU RGB+Flow+Depth. label: 0=attack, 1=real (from the filename access-type code)."""

    def __init__(self,
                 orig_root_dir: str,
                 file_list_path: Optional[str],
                 flow_root_dir: str,
                 depth_root_dir: str,
                 transform=None,
                 num_frames: int = 1,
                 is_train: bool = False,
                 protocol: Optional[str] = None,
                 flow_channels: int = 3,
                 flow_representation: str = "hsv",
                 hsv_max_mag: float = 20.0,
                 augment_flow: bool = True,
                 flow_offset: int = 0,
                 return_access_type: bool = False):
        self.orig_root_dir = orig_root_dir
        self.file_list_path = file_list_path
        self.flow_root_dir = flow_root_dir
        self.depth_root_dir = depth_root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.is_train = is_train
        self.protocol = protocol
        self.flow_channels = flow_channels
        self.flow_representation = flow_representation.lower()
        self.hsv_max_mag = hsv_max_mag
        self.augment_flow = augment_flow
        self.flow_offset = flow_offset
        self.return_access_type = return_access_type

        assert flow_channels in (2, 3)
        if flow_channels == 3:
            assert self.flow_representation in ("uvm", "hsv")

        self.samples = self._load_samples()

    def _load_samples(self):
        valid = None
        if self.protocol not in (None, 'all') and self.file_list_path is not None:
            valid = set()
            with open(self.file_list_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        valid.add(parts[1].strip())

        samples = []
        for fname in os.listdir(self.orig_root_dir):
            if not fname.lower().endswith(('.avi', '.mp4', '.mov')):
                continue
            if valid is not None:
                base = os.path.splitext(fname)[0]
                if fname not in valid and base not in valid:
                    continue
            try:
                access_type = int(fname.split('_')[-1].split('.')[0])
            except Exception:
                access_type = 2
            label = 1 if access_type == 1 else 0

            vpath = os.path.join(self.orig_root_dir, fname)
            if len(list_flo_files(flow_dir_for_video(vpath, self.orig_root_dir, self.flow_root_dir))) == 0:
                continue
            if depth_video_for_video(vpath, self.orig_root_dir, self.depth_root_dir) is None:
                continue

            samples.append((vpath, label, access_type) if self.return_access_type else (vpath, label))
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def _load_rgb_and_flow(self, video_path, start):
        fidx = frame_indices_from(start, self.num_frames)
        angle, scale = _rand_aug() if self.is_train else (0.0, 1.0)

        rgb_frames = load_rgb_frames(video_path, fidx, fallback_hw=(224, 224))
        if self.is_train:
            rgb_frames = [_aug_img(x, angle, scale) for x in rgb_frames]
        rh, rw = rgb_frames[0].shape[-2:]

        fdir = flow_dir_for_video(video_path, self.orig_root_dir, self.flow_root_dir)
        flo_files = list_flo_files(fdir)
        n_flows = len(flo_files)
        ffidx = np.arange(start + self.flow_offset, start + self.flow_offset + self.num_frames)
        if n_flows > 0:
            ffidx = np.clip(ffidx, 0, n_flows - 1)

        uv_seq = []
        for k in ffidx:
            if k >= n_flows:
                break
            try:
                uv_np = read_flo(flo_files[k]).astype(np.float32)
            except Exception:
                return False, None, None, None, None, None, None
            uv = torch.from_numpy(uv_np).permute(2, 0, 1)
            uv = resize_like(uv, (rh, rw), is_flow=True)
            if self.augment_flow and (angle != 0 or scale != 1):
                uv = rotate_scale_flow_vectors(uv, angle, scale)
                if angle != 0:
                    uv = F.rotate(uv, angle)
                if scale != 1:
                    uv = F.affine(uv, angle=0, translate=(0, 0), scale=scale, shear=0)
            uv_seq.append(uv)
        while len(uv_seq) < self.num_frames:
            uv_seq.append(uv_seq[-1] if uv_seq else torch.zeros(2, rh, rw))

        return True, rgb_frames, uv_seq, angle, scale, fidx, (rh, rw)

    def __getitem__(self, idx):
        if self.return_access_type:
            video_path, label, access_type = self.samples[idx]
        else:
            video_path, label = self.samples[idx]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        start = choose_start(total, self.num_frames, self.is_train)
        ok, rgb_frames, uv_seq, angle, scale, fidx, (rh, rw) = self._load_rgb_and_flow(video_path, start)
        if not ok:
            # A handful of OULU-NPU cached .flo files are 0KB; retry an adjacent start.
            max_start = max(0, total - self.num_frames)
            alt = start + 1 if start + 1 <= max_start else (start - 1 if start - 1 >= 0 else None)
            if alt is not None:
                ok, rgb_frames, uv_seq, angle, scale, fidx, (rh, rw) = self._load_rgb_and_flow(video_path, alt)

        flow_seq = _flow_seq_to_channels(uv_seq, self.flow_channels, self.flow_representation, self.hsv_max_mag)

        dpath = depth_video_for_video(video_path, self.orig_root_dir, self.depth_root_dir)
        if dpath is None:
            raise FileNotFoundError(f"No depth video for: {video_path}")
        depth_frames = load_depth_frames(dpath, fidx, fallback_hw=(rh, rw))
        depth_frames = [resize_like(x, (rh, rw)) for x in depth_frames]
        if self.is_train:
            depth_frames = [_aug_img(x, angle, scale) for x in depth_frames]

        combo = []
        for i in range(self.num_frames):
            cf = torch.cat([rgb_frames[i], flow_seq[i], depth_frames[i]], dim=0)
            if self.transform:
                cf = self.transform(cf)
            combo.append(cf)
        x = torch.stack(combo, 0)

        return (x, label, access_type) if self.return_access_type else (x, label)


# ================================
# RA / RM / RY: nested root/{attack,real}/**.mp4 layout
# ================================

class Generic_Combined(Dataset):
    """RGB+Flow+Depth for nested-class-directory datasets (Replay-Attack, Replay-Mobile, ROSE-Youtu, CASIA, MSU)."""

    def __init__(self,
                 orig_root_dir: str,
                 flow_root_dir: str,
                 depth_root_dir: str,
                 transform=None,
                 num_frames: int = 1,
                 is_train: bool = False,
                 classes: Tuple[str, str] = ('attack', 'real'),
                 flow_channels: int = 3,
                 flow_representation: str = "hsv",
                 hsv_max_mag: float = 20.0,
                 augment_flow: bool = True,
                 flow_offset: int = 0):
        self.orig_root_dir = orig_root_dir
        self.flow_root_dir = flow_root_dir
        self.depth_root_dir = depth_root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.is_train = is_train
        self.classes = list(classes)
        self.flow_channels = flow_channels
        self.flow_representation = flow_representation.lower()
        self.hsv_max_mag = hsv_max_mag
        self.augment_flow = augment_flow
        self.flow_offset = flow_offset

        assert flow_channels in (2, 3)
        if flow_channels == 3:
            assert self.flow_representation in ("uvm", "hsv")

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for ci, cls in enumerate(self.classes):
            base_dir = os.path.join(self.orig_root_dir, cls)
            for dirpath, _, filenames in os.walk(base_dir):
                for fname in filenames:
                    if not fname.lower().endswith(('.mp4', '.avi', '.mov')):
                        continue
                    vpath = os.path.join(dirpath, fname)
                    if len(list_flo_files(flow_dir_for_video(vpath, self.orig_root_dir, self.flow_root_dir))) == 0:
                        continue
                    if depth_video_for_video(vpath, self.orig_root_dir, self.depth_root_dir) is None:
                        continue
                    samples.append((vpath, ci))
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        start = choose_start(total, self.num_frames, self.is_train)
        fidx = frame_indices_from(start, self.num_frames)
        angle, scale = _rand_aug() if self.is_train else (0.0, 1.0)

        rgb_frames = load_rgb_frames(video_path, fidx, fallback_hw=(224, 224))
        if self.is_train:
            rgb_frames = [_aug_img(x, angle, scale) for x in rgb_frames]
        rh, rw = rgb_frames[0].shape[-2:]

        fdir = flow_dir_for_video(video_path, self.orig_root_dir, self.flow_root_dir)
        flo_files = list_flo_files(fdir)
        n_flows = len(flo_files)
        ffidx = np.arange(start + self.flow_offset, start + self.flow_offset + self.num_frames)
        if n_flows > 0:
            ffidx = np.clip(ffidx, 0, n_flows - 1)

        uv_seq = []
        for k in ffidx:
            if k >= n_flows:
                break
            uv_np = read_flo(flo_files[k]).astype(np.float32)
            uv = torch.from_numpy(uv_np).permute(2, 0, 1)
            uv = resize_like(uv, (rh, rw), is_flow=True)
            if self.augment_flow and (angle != 0 or scale != 1):
                uv = rotate_scale_flow_vectors(uv, angle, scale)
                if angle != 0:
                    uv = F.rotate(uv, angle)
                if scale != 1:
                    uv = F.affine(uv, angle=0, translate=(0, 0), scale=scale, shear=0)
            uv_seq.append(uv)
        while len(uv_seq) < self.num_frames:
            uv_seq.append(uv_seq[-1] if uv_seq else torch.zeros(2, rh, rw))

        flow_seq = _flow_seq_to_channels(uv_seq, self.flow_channels, self.flow_representation, self.hsv_max_mag)

        dpath = depth_video_for_video(video_path, self.orig_root_dir, self.depth_root_dir)
        if dpath is None:
            raise FileNotFoundError(f"No depth video for: {video_path}")
        depth_frames = load_depth_frames(dpath, fidx, fallback_hw=(rh, rw))
        depth_frames = [resize_like(x, (rh, rw)) for x in depth_frames]
        if self.is_train:
            depth_frames = [_aug_img(x, angle, scale) for x in depth_frames]

        combo = []
        for i in range(self.num_frames):
            cf = torch.cat([rgb_frames[i], flow_seq[i], depth_frames[i]], dim=0)
            if self.transform:
                cf = self.transform(cf)
            combo.append(cf)
        return torch.stack(combo, 0), label


# ================================
# SiW-Mv2: protocol text-list based
# ================================

def read_txt_list(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def split_train_val(items: List[str], is_train: bool, seed: int = 1234) -> List[str]:
    """Deterministic 80/20 split of a list of video-id stems."""
    rng = random.Random(seed)
    items = sorted(items)
    rng.shuffle(items)
    n_train = int(0.8 * len(items))
    return items[:n_train] if is_train else items[n_train:]


def find_video_by_stem(root: str, cls: str, stem: str) -> Optional[str]:
    cls_root = os.path.join(root, cls)
    for dirpath, _, filenames in os.walk(cls_root):
        for fn in filenames:
            if fn.lower().endswith(('.mov', '.mp4', '.avi')) and os.path.splitext(fn)[0] == stem:
                return os.path.join(dirpath, fn)
    return None


class SIW_Dataset(Dataset):
    """SiW-Mv2 Protocol-1, RGB+Flow+Depth. Classes: Spoof=0, Live=1.

    Layout: root/{Spoof,Live}/**.mov, root/protocol_files/{train,test}list_{all,live}.txt
    `flow_root_dir` may be a list of roots (tried in order; a video's flow dir need only
    exist under one of them).
    """

    def __init__(self,
                 orig_root_dir: str,
                 flow_root_dir,
                 depth_root_dir: str,
                 transform=None,
                 num_frames: int = 1,
                 is_train: bool = False,
                 flow_channels: int = 3,
                 flow_representation: str = "hsv",
                 hsv_max_mag: float = 20.0,
                 augment_flow: bool = True,
                 flow_offset: int = 0,
                 protocol: Optional[str] = '1',
                 split: str = "train",
                 seed: int = 1234):
        assert protocol == '1', "Only SIW Protocol-1 is implemented"
        assert split in ("train", "val", "test")

        self.orig_root_dir = orig_root_dir
        self.depth_root_dir = depth_root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.is_train = is_train
        self.flow_channels = flow_channels
        self.flow_representation = flow_representation.lower()
        self.hsv_max_mag = hsv_max_mag
        self.augment_flow = augment_flow
        self.flow_offset = flow_offset
        self.protocol = protocol
        self.split = split
        self.seed = seed

        self.flow_root_dirs = flow_root_dir if isinstance(flow_root_dir, list) else [flow_root_dir]
        self.classes = ("Spoof", "Live")
        self.samples = self._load_samples()

    def _load_samples(self):
        proto_dir = os.path.join(self.orig_root_dir, "protocol_files")
        train_spoof = read_txt_list(os.path.join(proto_dir, "trainlist_all.txt"))
        train_live = read_txt_list(os.path.join(proto_dir, "trainlist_live.txt"))
        test_spoof = read_txt_list(os.path.join(proto_dir, "testlist_all.txt"))
        test_live = read_txt_list(os.path.join(proto_dir, "testlist_live.txt"))

        if self.split == "test":
            spoof_ids, live_ids = test_spoof, test_live
        else:
            spoof_ids = split_train_val(train_spoof, self.is_train, self.seed)
            live_ids = split_train_val(train_live, self.is_train, self.seed)

        class_map = {"Spoof": spoof_ids, "Live": live_ids}
        samples = []
        for cls_idx, cls_name in enumerate(self.classes):
            for stem in class_map[cls_name]:
                vpath = find_video_by_stem(self.orig_root_dir, cls_name, stem)
                if vpath is None:
                    continue
                if not any(len(list_flo_files(flow_dir_for_video(vpath, self.orig_root_dir, fr))) > 0
                           for fr in self.flow_root_dirs):
                    continue
                if depth_video_for_video(vpath, self.orig_root_dir, self.depth_root_dir) is None:
                    continue
                spoof_type = os.path.basename(os.path.dirname(vpath)) if cls_name == "Spoof" else "Live"
                samples.append((vpath, cls_idx, spoof_type))
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label, spoof_type = self.samples[idx]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        start = choose_start(total, self.num_frames, self.is_train)
        fidx = frame_indices_from(start, self.num_frames)
        angle, scale = _rand_aug() if self.is_train else (0.0, 1.0)

        rgb_frames = load_rgb_frames(video_path, fidx, fallback_hw=(224, 224))
        if self.is_train:
            rgb_frames = [_aug_img(x, angle, scale) for x in rgb_frames]
        rh, rw = rgb_frames[0].shape[-2:]

        fdir = None
        for fr in self.flow_root_dirs:
            cand = flow_dir_for_video(video_path, self.orig_root_dir, fr)
            if cand and os.path.isdir(cand) and len(list_flo_files(cand)) > 0:
                fdir = cand
                break
        flo_files = list_flo_files(fdir)
        n_flows = len(flo_files)
        ffidx = np.clip(np.arange(start + self.flow_offset, start + self.flow_offset + self.num_frames), 0, n_flows - 1)

        uv_seq = []
        for k in ffidx:
            uv_np = read_flo(flo_files[k]).astype(np.float32)
            uv = torch.from_numpy(uv_np).permute(2, 0, 1)
            uv = resize_like(uv, (rh, rw), is_flow=True)
            if self.augment_flow and (angle != 0 or scale != 1):
                uv = rotate_scale_flow_vectors(uv, angle, scale)
                if angle != 0:
                    uv = F.rotate(uv, angle)
                if scale != 1:
                    uv = F.affine(uv, angle=0, translate=(0, 0), scale=scale, shear=0)
            uv_seq.append(uv)
        while len(uv_seq) < self.num_frames:
            uv_seq.append(uv_seq[-1])

        flow_seq = _flow_seq_to_channels(uv_seq, self.flow_channels, self.flow_representation, self.hsv_max_mag)

        dpath = depth_video_for_video(video_path, self.orig_root_dir, self.depth_root_dir)
        depth_frames = load_depth_frames(dpath, fidx, fallback_hw=(rh, rw))
        depth_frames = [resize_like(x, (rh, rw)) for x in depth_frames]
        if self.is_train:
            depth_frames = [_aug_img(x, angle, scale) for x in depth_frames]

        combo = []
        for i in range(self.num_frames):
            cf = torch.cat([rgb_frames[i], flow_seq[i], depth_frames[i]], dim=0)
            if self.transform:
                cf = self.transform(cf)
            combo.append(cf)

        if self.split == "test":
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            return torch.stack(combo, 0), label, (video_id, spoof_type)
        return torch.stack(combo, 0), label
