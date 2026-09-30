# DoseDINO

Official implementation of **DoseDINO: Bridging the Physical Gap in Foundation Models via Anatomical-Geometric Decoupling**.

DoseDINO predicts axial radiotherapy dose slices with three named components from the paper:

- **DINOv3 Encoder**: a ViT-B/16 anatomical stream adapted to five input channels.
- **Parallel Geometric Injection (PGI)**: a CNN stream for a 3D-derived PTV-distance prior, fused by directed attention.
- **Continuous Dose Dynamics Head (CDD-Head)**: a Neural ODE integrated with four fixed RK4 steps and followed by a Softplus dose output.

The model is slice-wise. PTV-distance and beam-plate maps are computed from the complete 3D plan before slicing, and predicted slices are stacked to reconstruct a 3D dose volume.

## Input channels

Every input tensor has shape `(B, 6, 256, 256)` and uses this fixed order:

| Index | Tensor | Stream |
|---:|---|---|
| 0 | normalized CT intensity | DINOv3 |
| 1 | PTV prescription map | DINOv3 |
| 2 | OAR priority map | DINOv3 |
| 3 | binary body mask | DINOv3 |
| 4 | distance-aware beam plate | DINOv3 |
| 5 | PTV distance map | PGI |

When the pretrained DINOv3 projection is expanded from three to five channels, its original three-channel weights and bias are copied. Weights for the body-mask and beam-plate channels are initialized to zero.

## Repository layout

- `dino_unetr/dosedino.py`: paper-aligned `DoseDINO`, `PGIEncoder`, `DirectedAttention`, and `CDDHead` definitions.
- `train_dosedino.py`: training entrypoint with body-masked L1 loss and validation-checkpoint selection.
- `infer_dosedino.py`: slice-wise inference and 3D reconstruction.
- `data_preprocess.py`: construction of plan-conditioned input maps.
- `run_process.py`: deterministic resizing and conversion of volumes into axial NPZ slices.
- `data_loader_lightning_slice.py`: deterministic train/validation/test slice loaders.
- `config_files/`: training and inference configuration.

Former MED-DINO-UNETR module and entrypoint names remain as thin compatibility imports. New work should use the DoseDINO names above.
The legacy names now select the paper implementation; they do not reproduce the historical model's CDD behavior or channel ordering.

## Environment

The reported run used Python with PyTorch 2.7.0 on one NVIDIA RTX 5090. Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Place the trusted MedDINOv3/DINOv3 ViT-B/16 checkpoint at `dino_unetr/model.pth`, or change `model_params.checkpoint_path` in the YAML configuration. Model weights and clinical image data are not redistributed in this repository.

## Data preparation

The included metadata records the patient-level GDP-HMM split used in the paper:

- optimization: 2,730 plans from 1,211 patients;
- validation: 148 plans from 64 patients;
- held-out testing: 356 plans from 147 patients.

Update the local NPZ paths in your metadata CSV, then run:

```bash
python data_preprocess.py
python run_process.py
```

`run_process.py` keeps every plan in its metadata-defined split and writes the six channels in the order listed above. Existing slice files produced with an earlier channel order must be regenerated.

## Training

```bash
python train_dosedino.py config_files/config_train.yaml
```

The paper configuration uses:

- full encoder and PGI fine-tuning;
- batch size 64;
- AdamW with learning rate `1e-4` and weight decay `1e-4`;
- epoch-wise cosine decay without warm-up;
- up to 200 epochs and no automatic early stopping;
- deterministic resizing with no stochastic augmentation;
- body-masked voxel-wise L1 loss;
- the checkpoint with the minimum validation loss.

No global random seed is fixed because the reported result is a single training run. The training entrypoint uses `val_dataloader()` for checkpoint selection; the held-out test loader is not used by `Trainer.fit`.

## Inference

Set `save_model_path` and the local data paths in `config_files/config_infer.yaml`, then run:

```bash
python infer_dosedino.py --cfig_path config_files/config_infer.yaml --phase valid --dev_split test
```

Predicted axial slices are resized to the source in-plane size and stacked into one NumPy volume per plan.

## Reproducibility notes

- The CDD latent state starts from `y(0)=0` and is integrated over `[0, 1]` with exactly four classical RK4 steps (`h=0.25`, 16 dynamics evaluations). The decoder features provide `G(x)`; the block returns the projected final state without adding a residual `x`, and the complete dose head ends in Softplus.
- Validation and testing are separate: validation uses `(phase=train, dev_split=valid)`, while testing uses `(phase=valid, dev_split=test)`.
- Statistical comparisons reported in the paper are paired at the reconstructed-plan level, not at the slice or voxel level.
