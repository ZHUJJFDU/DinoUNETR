import os
import glob
from pathlib import Path
from torch.utils.data import Dataset
import pytorch_lightning as pl
from monai.data import CacheDataset, DataLoader, partition_dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, ToTensord, SpatialPadd, ConcatItemsd, Resized
)
import torch
import numpy as np

class SliceDataset(Dataset):
    """
    Wrap a dataset of 3D volumes to return 2D slices.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.slices = []
        print("Initializing SliceDataset (scanning volumes for slices)...")
        for i in range(len(self.dataset)):
            # We assume the dataset returns a dict with 'data' key which is (C, H, W, D)
            # Accessing dataset[i] might trigger loading/transforms if not cached.
            item = self.dataset[i]
            # Assume 'data' key exists and has the shape we want to slice along last dim
            if 'data' in item:
                d = item['data'].shape[-1]
                for z in range(d):
                    self.slices.append((i, z))
            else:
                # Fallback or skip
                pass
        print(f"SliceDataset initialized with {len(self.slices)} slices from {len(self.dataset)} volumes.")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.slices[idx]
        data_dict = self.dataset[vol_idx]
        new_dict = {}
        
        # Slice all 4D tensors (C, H, W, D) -> (C, H, W)
        # Leave others as is
        for k, v in data_dict.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                if isinstance(v, torch.Tensor) and v.ndim == 4:
                     new_dict[k] = v[..., slice_idx]
                elif isinstance(v, np.ndarray) and v.ndim == 4:
                     new_dict[k] = v[..., slice_idx]
                else:
                     new_dict[k] = v
            else:
                new_dict[k] = v
        return new_dict



class DataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=2, num_workers=4, split_ratio=0.8):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def get_dict_list(self, data_root):
        data_dicts = []
        search_pattern = os.path.join(data_root, "ct", "*.nii.gz")
        print(f"Searching for data in: {search_pattern}")
        image_paths = sorted(glob.glob(search_pattern))
        print(f"Found {len(image_paths)} files.")
        
        for image_path in image_paths:
            file_name = os.path.basename(image_path)
            patient_id = file_name.replace("_aligned_ct.nii.gz", "")
            root = Path(data_root)
            data_dict = {
                "ct": str(image_path),
                "ptv": str(root / "ptv" / f"{patient_id}_PTV_mask.nii.gz"),
                "organ": str(root / "organ" / f"{patient_id}_OAR_mask.nii.gz"),
                "body_mask": str(root / "body_masks" / f"{patient_id}_body_mask.nii.gz"),
                "label": str(root / "dose" / f"{patient_id}_registered_dose.nii.gz"),
                "patient_id": patient_id
            }
            check_keys = ["ct", "ptv", "organ", "body_mask", "label"]
            if all(Path(data_dict[k]).exists() for k in check_keys):
                data_dicts.append(data_dict)
            else:
                print(f"警告: 患者 {patient_id} 文件不完整，已跳过。")
        return data_dicts

    def get_transforms(self, stage):
        data_keys = ["ct", "ptv", "organ", "body_mask", "label"]
        transform = [
            LoadImaged(keys=data_keys, image_only=False),
            EnsureChannelFirstd(keys=data_keys),
            Orientationd(keys=data_keys, axcodes='RAS'),
            Spacingd(
                keys=data_keys, 
                pixdim=(1.25, 1.25, 2.5), 
                mode=("bilinear", "nearest", "nearest", "nearest", "bilinear")
            ),
            ScaleIntensityRanged(keys=["ct"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=data_keys, source_key="body_mask"),
            # Resize H and W to 256, keep Depth (-1)
            Resized(
                keys=data_keys, 
                spatial_size=(256, 256, -1), 
                mode=("bilinear", "nearest", "nearest", "nearest", "bilinear")
            ),
            # Concatenate modalities to form 6-channel input
            # data keys: ct, ptv, organ, body_mask, ct, ptv -> 6 channels
            ConcatItemsd(keys=["ct", "ptv", "organ", "body_mask", "ct", "ptv"], name="data", dim=0),
        ]
        if stage == 'train':
            pass
        elif stage == 'val':
            pass

        transform.append(ToTensord(keys=["data", "label"]))
        return Compose(transform)

    def setup(self, stage=None):
        all_files = self.get_dict_list(self.data_root)

        train_files, val_files = partition_dataset(
            data=all_files, 
            ratios=[self.split_ratio, 1 - self.split_ratio], 
            shuffle=True, 
            seed=42
        )

        if stage == "fit" or stage is None:
            train_ds = CacheDataset(
                data=train_files, 
                transform=self.get_transforms('train'),
                cache_rate=1.0, 
                num_workers=self.num_workers
            )
            self.train_dataset = SliceDataset(train_ds)
            
            val_ds = CacheDataset(
                data=val_files, 
                transform=self.get_transforms('val'),
                cache_rate=1.0, 
                num_workers=self.num_workers
            )
            self.val_dataset = SliceDataset(val_ds)
         

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


if __name__ == "__main__":
    data_root = r"C:\Users\960\Desktop\aiendtoend\dataset_normalized" 
    data_loader = DataModule(
        data_root=data_root,
        batch_size=4,
        num_workers=4,
        split_ratio=0.8
    )
    data_loader.setup(stage="fit")
    train_loader = data_loader.train_dataloader()
    val_loader = data_loader.val_dataloader()
    print(f"Train dataset size: {len(data_loader.train_dataset)}")
    print(f"Val dataset size: {len(data_loader.val_dataset)}")
