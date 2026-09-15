"""Shared low-level utilities: logging dirs, tensor transforms, batch collation."""
import os
import torch
import torchvision.transforms as T
from datetime import datetime


def create_log_directory(base_dir):
    """Create base_dir/log_NNN_<timestamp> with an auto-incrementing 3-digit index."""
    os.makedirs(base_dir, exist_ok=True)
    log_numbers = []
    for d in os.listdir(base_dir):
        if d.startswith('log_'):
            try:
                log_numbers.append(int(d.split('_')[1]))
            except ValueError:
                pass
    next_num = 1 if not log_numbers else max(log_numbers) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_log_dir = os.path.join(base_dir, f'log_{next_num:03d}_{timestamp}')
    os.makedirs(new_log_dir)
    return new_log_dir


def collate_fn(batch):
    """Pads variable-length (T,C,H,W) sequences and stacks the batch.

    Supports plain (frames, label) samples, OULU-style (frames, label, access_type),
    and SIW-style (frames, label, (video_id, spoof_type)) samples.
    """
    first = batch[0]
    has_third = isinstance(first, (list, tuple)) and len(first) == 3

    is_oulu = False
    is_siw = False
    if has_third:
        third = first[2]
        if isinstance(third, (int, torch.Tensor)):
            is_oulu = True
        elif isinstance(third, (tuple, list)) and len(third) == 2:
            is_siw = True
        else:
            raise ValueError(f"Unknown 3rd element type in batch: {type(third)}")

    max_len = max(sample[0].size(0) for sample in batch)

    padded_frames, labels = [], []
    access_types = []
    video_ids, spoof_types = [], []

    for sample in batch:
        if has_third:
            frames, label, meta = sample
        else:
            frames, label = sample

        T_, C, H, W = frames.shape
        if T_ < max_len:
            padding = torch.zeros((max_len - T_, C, H, W), dtype=frames.dtype, device=frames.device)
            frames = torch.cat([frames, padding], dim=0)
        padded_frames.append(frames)

        labels.append(int(label.item()) if isinstance(label, torch.Tensor) else int(label))

        if is_oulu:
            atype = meta
            access_types.append(int(atype.item()) if isinstance(atype, torch.Tensor) else int(atype))
        elif is_siw:
            vid, stype = meta
            video_ids.append(vid)
            spoof_types.append(stype)

    padded_frames = torch.stack(padded_frames, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    if is_oulu:
        return padded_frames, labels, torch.tensor(access_types, dtype=torch.long)
    if is_siw:
        return padded_frames, labels, (video_ids, spoof_types)
    return padded_frames, labels


# ---------- Depth / flow normalization ----------

FLOW_T = 20.0  # pixels; clip range for raw flow magnitude before scaling to [-1, 1]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def normalize_depth_tensor(x: torch.Tensor, mode: str = "standardize", D_T: float = 10.0) -> torch.Tensor:
    """x: (1,H,W) depth channel in [0,1]. mode: standardize -> [-1,1], zero_one, or clip_scale."""
    out = x.clone()
    if mode == "standardize":
        out = (out - 0.5) / 0.5
    elif mode == "zero_one":
        out = out.clamp_(0.0, 1.0)
    elif mode == "clip_scale":
        out = out.clamp_(0.0, D_T) / D_T
    else:
        raise ValueError(f"Unsupported depth normalization mode: {mode}")
    return out


def normalize_combined_with_depth_tensor(x: torch.Tensor,
                                          flow_channels: int,
                                          flow_representation: str,
                                          depth_norm: str = "standardize",
                                          flow_T: float = FLOW_T) -> torch.Tensor:
    """x: (3 + flow_channels + 1, H, W) = RGB | Flow | Depth. Normalizes each modality in place."""
    x[:3] = (x[:3] - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)

    if flow_channels == 2:
        x[3] = x[3].clamp_(-flow_T, flow_T).div_(flow_T)
        x[4] = x[4].clamp_(-flow_T, flow_T).div_(flow_T)
        depth_start = 5
    else:
        if flow_representation == 'uvm':
            x[3] = x[3].clamp_(-flow_T, flow_T).div_(flow_T)
            x[4] = x[4].clamp_(-flow_T, flow_T).div_(flow_T)
            x[5] = x[5].clamp_(0, flow_T).div_(flow_T)
        elif flow_representation == 'hsv':
            x[3:6] = (x[3:6] - 0.5) / 0.5
        else:
            raise ValueError(f"Unsupported flow_representation: {flow_representation}")
        depth_start = 6

    x[depth_start:] = normalize_depth_tensor(x[depth_start:], mode=depth_norm)
    return x


def make_combined_transform_with_depth(img_size: int,
                                        flow_channels: int,
                                        flow_representation: str,
                                        depth_norm: str = "standardize",
                                        flow_T: float = FLOW_T):
    """Transform for the RGB+Flow+Depth concatenated tensor used by the IFD teacher/KD pipeline."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.Lambda(lambda t: normalize_combined_with_depth_tensor(
            t, flow_channels=flow_channels, flow_representation=flow_representation,
            depth_norm=depth_norm, flow_T=flow_T)),
    ])
