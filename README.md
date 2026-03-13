# DinoUNETR

DinoUNETR is a PyTorch/Lightning-based pipeline for dose prediction using MedDINOv3 backbones and U-Net style decoders. The repository includes preprocessing, training, inference, and evaluation utilities for volumetric radiotherapy data.

## Requirements

- Python 3.9+
- PyTorch, PyTorch Lightning
- MONAI
- NumPy, pandas, PyYAML, tqdm

Install dependencies from your environment manager (conda/pip) as needed.

## Project Layout

- `dino_unetr/`: model definitions (MED-DINO-UNETR variants)
- `train_lightning.py`: training entrypoint for standard models
- `train_lightning_distance.py`: training entrypoint for distance-aware models
- `data_preprocess.py`: dataset preprocessing to build NPZ files
- `data_loader_lightning.py`: 3D dataset loader
- `data_loader_lightning_slice.py`: 2D slice loader
- `inference_simple.py`: slice-by-slice inference for standard models
- `inference_distance.py`: slice-by-slice inference for distance-aware models
- `toolkit.py`: geometry, DVH, and augmentation utilities
- `config_files/`: YAML configuration files
- `results/`: inference outputs and metrics

## Preprocessing

1. Update paths in `data_preprocess.py` or supply your own config dictionary.
2. Run:

```bash
python data_preprocess.py
```

This generates per-case NPZ files under `dataset_save_root`.

## Training

### Standard model

```bash
python train_lightning.py config_files/config_DinoUnetr.yaml
```

### Distance-aware model

```bash
python train_lightning_distance.py config_files/config_DinoUnetr.yaml
```

## Inference

### Standard model

```bash
python inference_simple.py --cfig_path config_files/config_infer.yaml --phase valid --dev_split test
```

### Distance-aware model

```bash
python inference_distance.py --cfig_path config_files/config_infer.yaml --phase valid --dev_split test
```

## Configuration

Key settings live in `config_files/*.yaml`:

- `loader_params`: input size, output size, batch sizes
- `model_params`: input channels and checkpoint paths
- `save_model_root` / `save_model_path`: checkpoint location
- `save_pred_path`: inference output path

## Notes

- PyTorch 2.6+ defaults to `weights_only=True` in `torch.load`. The inference scripts set `weights_only=False` for trusted checkpoints.
- Data augmentation is configured in `toolkit.py` and used by `data_loader_lightning.py`.
