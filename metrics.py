"""Evaluation metrics: Accuracy, AUC-ROC, EER, HTER, and (OULU/SiW) APCER/BPCER/ACER."""
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

try:
    import oulumetrics
    HAS_OULU = True
except ImportError:
    HAS_OULU = False
    print("[WARN] oulumetrics not installed. APCER/BPCER/ACER for OULU will not be computed "
          "(pip install oulumetrics, or a compatible implementation, to enable it).")


def evaluate_all(model, loader, criterion, device, dataset: str = None, kd_student: bool = False):
    """Run `model` over `loader` and compute classification + FacePAD metrics.

    `loader` yields (inputs, labels), (inputs, labels, access_types) for dataset="OULU",
    or (inputs, labels, (video_ids, spoof_types)) for dataset="SIW". `inputs` are
    (B, T, C, H, W); when kd_student=True only the leading 3 (RGB) channels are used.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels, all_probs, all_times = [], [], []
    all_access_types = []
    all_video_ids, all_spoof_types = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            if len(batch) == 2:
                inputs, labels = batch
                if dataset in ("OULU", "SIW"):
                    raise ValueError("OULU / SIW dataset must return access_type or (video_id, spoof_type)")
            elif len(batch) == 3:
                if dataset == "OULU":
                    inputs, labels, access_types = batch
                elif dataset == "SIW":
                    inputs, labels, (video_ids, spoof_types) = batch
                else:
                    raise ValueError("Only OULU/SIW datasets return a 3rd batch element")
            else:
                raise ValueError(f"Unexpected batch size {len(batch)}")

            if kd_student:
                inputs = inputs[:, :, :3, :, :]  # student sees RGB only

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            start_time = time.time()
            outputs = model(inputs)
            all_times.append((time.time() - start_time) / max(1, inputs.size(0)))

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(class 1 = "real")
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if dataset == "OULU":
                all_access_types.extend(access_types.cpu().numpy().tolist())
            if dataset == "SIW":
                all_video_ids.extend(video_ids)
                all_spoof_types.extend(spoof_types)

    all_labels_np = np.array(all_labels, dtype=np.int32)
    all_probs_np = np.array(all_probs, dtype=np.float32)

    fpr, tpr, thresholds = roc_curve(all_labels_np, all_probs_np)
    auc_roc = auc(fpr, tpr)

    fnr = 1.0 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    youdens_index = tpr[optimal_idx] - fpr[optimal_idx]
    far = fpr[optimal_idx]
    frr = fnr[optimal_idx]
    hter = (far + frr) / 2.0

    test_loss = running_loss / max(1, len(loader.dataset))
    test_acc = 100.0 * correct / max(1, total)
    avg_inference_time = float(np.mean(all_times)) if all_times else 0.0

    apcer = bpcer = acer = None
    siw_protocol1 = None

    if dataset == "OULU":
        if not HAS_OULU:
            print("[WARN] oulumetrics not available, skipping APCER/BPCER/ACER.")
        elif len(all_access_types) == 0:
            print("[WARN] No access_type information; cannot compute APCER/BPCER/ACER.")
        else:
            apcer, bpcer, acer = oulumetrics.calculate_metrics(
                np.array(all_access_types, dtype=np.int32).tolist(), all_probs_np.tolist()
            )

    if dataset == "SIW":
        video_scores, video_labels, video_spoof_types = _aggregate_siw_video_scores(
            all_video_ids, all_probs_np, all_labels_np, all_spoof_types
        )
        siw_protocol1 = _evaluate_siw_protocol1(video_scores, video_labels, video_spoof_types)
        # The reference SIW protocol reports per-spoof-type ACER, not a single number;
        # we additionally report the mean across spoof types for a single headline metric.
        apcer = float(np.mean([v["APCER"] for v in siw_protocol1.values()]))
        bpcer = float(np.mean([v["BPCER"] for v in siw_protocol1.values()]))
        acer = float(np.mean([v["ACER"] for v in siw_protocol1.values()]))

    results = {
        'test_loss': float(test_loss), 'test_acc': float(test_acc),
        'auc_roc': float(auc_roc), 'eer': float(eer), 'hter': float(hter),
        'far': float(far), 'frr': float(frr),
        'youdens_index': float(youdens_index), 'optimal_threshold': float(optimal_threshold),
        'avg_inference_time': float(avg_inference_time),
        'fpr': fpr, 'tpr': tpr, 'labels': all_labels_np, 'probs': all_probs_np,
        'apcer': None if apcer is None else float(apcer),
        'bpcer': None if bpcer is None else float(bpcer),
        'acer': None if acer is None else float(acer),
    }
    if siw_protocol1 is not None:
        results["siw_protocol1"] = siw_protocol1
    return results


# ---------------- SiW-Mv2 protocol-1 (per spoof-type ACER) ----------------

def _aggregate_siw_video_scores(video_ids, probs, labels, spoof_types):
    score_dict = defaultdict(list)
    label_dict, spoof_dict = {}, {}
    for vid, prob, lbl, stype in zip(video_ids, probs, labels, spoof_types):
        score_dict[vid].append(1.0 - prob)  # spoof score: higher = more spoof-like
        label_dict[vid] = lbl
        spoof_dict[vid] = stype
    video_scores = {vid: float(np.mean(sorted(scores))) for vid, scores in score_dict.items()}
    return video_scores, label_dict, spoof_dict


def _siw_compute_apcer_bpcer_acer(video_scores, video_labels):
    y_true, y_score = [], []
    for vid in video_scores:
        y_score.append(video_scores[vid])
        y_true.append(1 if video_labels[vid] == 0 else 0)  # 0=Live,1=Spoof

    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=True)
    fnr = 1.0 - tpr

    best_acer, best_apcer, best_bpcer = 1e9, None, None
    for i in range(len(thresholds)):
        apcer, bpcer = fpr[i], fnr[i]
        acer = 0.5 * (apcer + bpcer)
        if acer < best_acer:
            best_acer, best_apcer, best_bpcer = acer, apcer, bpcer
    return best_apcer, best_bpcer, best_acer


def _evaluate_siw_protocol1(video_scores, video_labels, spoof_types):
    results = {}
    live_vids = [v for v in video_scores if spoof_types[v] == "Live"]
    spoof_groups = defaultdict(list)
    for v, stype in spoof_types.items():
        if stype != "Live":
            spoof_groups[stype].append(v)

    for stype, vids in spoof_groups.items():
        eval_vids = live_vids + vids
        scores = {v: video_scores[v] for v in eval_vids}
        labels = {v: video_labels[v] for v in eval_vids}
        apcer, bpcer, acer = _siw_compute_apcer_bpcer_acer(scores, labels)
        results[stype] = {"APCER": apcer, "BPCER": bpcer, "ACER": acer}
    return results
