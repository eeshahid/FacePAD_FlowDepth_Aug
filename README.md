# FacePAD: Flow + Depth Distillation

Face presentation-attack detection (FacePAD) is essential for securing face
recognition systems against spoofing attacks such as printed photographs, replayed
videos, and facial forgeries — but most approaches either generalize poorly across
datasets or need expensive auxiliary sensing at inference time.

This repository implements a multi-modal FacePAD framework that jointly exploits
**appearance, structural, and motion** cues — RGB, monocular depth, and optical flow —
through a multi-branch MobileNetV3-Large **teacher** network. That knowledge is then
**distilled into a lightweight, RGB-only student**, eliminating depth estimation and
optical-flow computation at deployment while retaining almost all of the teacher's
detection performance.

The framework achieves state-of-the-art or highly competitive results across five FacePAD benchmarks, and demonstrates strong cross-dataset generalization under standard OCIM leave-one-dataset-out protocol:

| Dataset | Teacher | Distilled RGB-only student |
|---|---|---|
| Replay-Attack | 0.0% HTER | 0.0% HTER |
| Replay-Mobile | 0.0% HTER | 0.0% HTER |
| ROSE-Youtu | 0.453% HTER | 0.531% HTER |
| OULU-NPU | 0.347% ACER | 0.625% ACER |
| SiW-Mv2 | 4.66% HTER | 4.39% HTER |
| OCIM leave-one-out (avg.) | 14.55% HTER | 14.51% HTER |

Ablation studies (in the accompanying manuscript) further validate the contribution of
each modality, temporal motion modeling, and the feature-fusion strategy.

This repository accompanies the manuscript *"Distilling Structural and Motion Cues for
Lightweight Face Presentation Attack Detection"* (under review, Expert Systems With
Applications). The code here covers the multi-modal teacher and its distilled student
on individual datasets, joint multi-dataset training, and the cross-dataset OCIM
protocol — the core results of the paper.

## Datasets

Replay-Attack, Replay-Mobile, ROSE-Youtu, OULU-NPU, SiW-Mv2, CASIA-MFSD, and MSU-MFSD.

## Repository structure

| file | role |
|---|---|
| `models.py` | the multi-modal teacher and the distilled RGB-only student |
| `datasets.py` | dataset loaders for RA / RM / RY / OULU-NPU / SiW-Mv2 |
| `datasets_ocim.py` | cross-dataset (OCIM) dataset assembly, built on top of `datasets.py` |
| `common.py` | dataset-path registry and transform/dataset/model builders shared by every script |
| `engine.py` | training/validation loops with early stopping and checkpointing |
| `metrics.py` | Accuracy, AUC-ROC, EER, HTER, and (OULU/SiW) APCER/BPCER/ACER |
| `utils.py` | tensor transforms and batch collation |
| `train.py` / `evaluate.py` | teacher training/evaluation on a single dataset |
| `train_kd.py` | student distillation on a single dataset |
| `train_joint.py` / `evaluate_joint.py` | teacher training/evaluation jointly on RA+RM+RY |
| `train_joint_kd.py` | student distillation on the joint teacher |
| `train_ocim.py` / `evaluate_ocim.py` | cross-dataset (OCIM) teacher training/evaluation |
| `train_ocim_kd.py` | student distillation for the OCIM protocol |
| `summarize_ocim.py` | collects all OCIM runs into one results table |

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python train.py --dataset RY                                              # train the teacher
python train_kd.py --teacher_log_dir logs/RY/imgflowdepth/log_001_...     # distill the student
```

Joint (RA+RM+RY) and cross-dataset (OCIM) training follow the same teacher-then-student
pattern via `train_joint.py`/`train_joint_kd.py` and `train_ocim.py`/`train_ocim_kd.py`.
Run any script with `--help` to see the full set of options, and see `common.py` for the expected dataset layout.

## Citation

If you find this code helpful in your work, please cite the accompanying manuscript:

```bibtex
@article{facepad_flow_depth_distillation,
  author  = {Jabbar, Muhammad Shahid and Ibrahim, Muhammad Sohail and Khan, Shujaat},
  title   = {Distilling Structural and Motion Cues for Lightweight Face Presentation Attack Detection},
  note    = {Manuscript submitted for publication},
  year    = {2026}
}
```

Full citation details will be added here once the manuscript is published.
