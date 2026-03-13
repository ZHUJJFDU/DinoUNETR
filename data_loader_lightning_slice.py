from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import yaml
import os
import glob
import json

HaN_OAR_LIST = [ 'Cochlea_L', 'Cochlea_R','Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'Chiasim', 'LacrimalGlands', 'BrachialPlexus', 'Brain', 'BrainStem_03', 'Esophagus', 'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV', 'OCavity-PTV', 'ParotidCon-PTV', 'Parotidlps-PTV', 'Parotids-PTV', 'PharConst-PTV', 'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05']
HaN_OAR_DICT = {HaN_OAR_LIST[i]: (i+1) for i in range(len(HaN_OAR_LIST))}

Lung_OAR_LIST = ["PTV_Ring.3-2", "Total Lung-GTV", "SpinalCord", "Heart", "LAD", "Esophagus", "BrachialPlexus", "GreatVessels", "Trachea", "Body_Ring0-3"]
Lung_OAR_DICT = {Lung_OAR_LIST[i]: (i+10) for i in range(len(Lung_OAR_LIST))}


from monai.transforms import (
    Compose,
    Resized,
    RandFlipd, 
    RandRotated,
    RandSpatialCropd,
    CenterSpatialCropd
)

class ProcessedSliceDataset(Dataset):
    """Load preprocessed 2D slice NPZ files."""
    def __init__(self, data_root, cfig, phase='train'):
        self.data_root = data_root
        self.phase = phase
        self.cfig = cfig
        
        search_path = os.path.join(data_root, "*.npz")
        self.file_list = glob.glob(search_path)
        
        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_root}. Please check the path.")
            
        print(f"[{phase}] Initialized dataset from {data_root}. Total slices: {len(self.file_list)}")

        self.out_size = cfig.get('out_size', [96, 256, 256])
        self.target_h = self.out_size[1]
        self.target_w = self.out_size[2]
        
        self.train_transforms = Compose([
            # 1. Random Spatial Crop (Directly from source or with minimal scaling)
            RandSpatialCropd(
                keys=self.keys, 
                roi_size=[int(self.target_h * 0.9), int(self.target_w * 0.9)], 
                max_roi_size=[int(self.target_h * 1.1), int(self.target_w * 1.1)], 
                random_center=True, 
                random_size=True, 
                allow_missing_keys=True
            ),
            
            # 2. Random Rotate (Reduced prob to 0.5 for better stability)
            RandRotated(
                keys=self.keys, 
                prob=0.5, 
                range_x=0.2, 
                mode=modes, 
                padding_mode='zeros', 
                allow_missing_keys=True
            ),
            
            # 3. Random Flip (Reduced prob)
            RandFlipd(keys=self.keys, prob=0.3, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=self.keys, prob=0.3, spatial_axis=1, allow_missing_keys=True),
            
            # 4. Final Resize to target size (The ONLY scaling step that determines fixed size)
            Resized(
                keys=self.keys, 
                spatial_size=target_size, 
                mode=modes, 
                allow_missing_keys=True
            )
        ])
        
        self.val_transforms = Compose([
             Resized(keys=self.keys, spatial_size=target_size, mode=modes, allow_missing_keys=True)
        ])


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        try:
            npz = np.load(file_path, allow_pickle=True)
            
            data = npz['data']
            label = npz['label']
            body = npz['body']            
            
            if 'ptv' in npz:
                ptv = npz['ptv']
            else:
                ptv = np.zeros_like(label)
                
            if 'oar_serial' in npz:
                oar_serial = npz['oar_serial']
            else:
                oar_serial = np.zeros_like(label)
                
            if 'oar_parallel' in npz:
                oar_parallel = npz['oar_parallel']
            else:
                oar_parallel = np.zeros_like(label)

            if 'isocenter' in npz:
                isocenter = torch.from_numpy(npz['isocenter']).float()
            else:
                isocenter = torch.zeros(3).float()

            if 'spacing' in npz:
                spacing = torch.from_numpy(npz['spacing']).float()
            else:
                spacing = torch.ones(3).float()

            if 'angle_list' in npz:
                raw_angles = npz['angle_list']
                if raw_angles.shape == ():
                    angle_list = raw_angles.item()
                else:
                    angle_list = raw_angles.tolist()
            else:
                angle_list = []
            
            angle_list_str = json.dumps(angle_list)

            data = torch.from_numpy(data).float()
            label = torch.from_numpy(label).float()
            body = torch.from_numpy(body).float()
            ptv = torch.from_numpy(ptv).float()
            oar_serial = torch.from_numpy(oar_serial).float()
            oar_parallel = torch.from_numpy(oar_parallel).float()

            data_dict = {
                'data': data,
                'label': label,
                'body': body,
                'ptv': ptv,
                'oar_serial': oar_serial,
                'oar_parallel': oar_parallel
            }
            
            if self.phase == 'train':
                data_dict = self.train_transforms(data_dict)
            else:
                data_dict = self.val_transforms(data_dict)
            
            data_dict['isocenter'] = isocenter
            data_dict['spacing'] = spacing
            data_dict['angle_list'] = angle_list_str
            data_dict['id'] = os.path.basename(file_path).replace('.npz', '') 
            
            return data_dict
            
        except Exception as e:
            raise RuntimeError(f"Failed to load file at index {idx}: {file_path}. Original error: {str(e)}")

class GetLoader(object):
    def __init__(self, cfig):
        super().__init__()
        self.cfig = cfig
        
        self.train_root = 'Dataset_256_layout_changechannel_nah&lung/Train'
        self.valid_root = 'Dataset_256_layout_changechannel_nah&lung/Valid'
        self.test_root = 'Dataset_256_layout_changechannel_nah&lung/Test'
        
    def train_dataloader(self):
        dataset = ProcessedSliceDataset(data_root=self.train_root, cfig=self.cfig, phase='train')
        
        kwargs = {
            'batch_size': self.cfig['train_bs'],
            'shuffle': True,
            'num_workers': self.cfig['num_workers'],
        }
        if torch.cuda.is_available():
            kwargs['pin_memory'] = True
        if self.cfig['num_workers'] > 0:
            kwargs['persistent_workers'] = True
            if 'prefetch_factor' in self.cfig:
                kwargs['prefetch_factor'] = self.cfig['prefetch_factor']
                
        return DataLoader(dataset, **kwargs)

    def val_dataloader(self):
        dataset = ProcessedSliceDataset(data_root=self.valid_root, cfig=self.cfig, phase='valid')
        
        kwargs = {
            'batch_size': self.cfig['val_bs'],
            'shuffle': False,
            'num_workers': self.cfig['num_workers'],
        }
        if torch.cuda.is_available():
            kwargs['pin_memory'] = True
        if self.cfig['num_workers'] > 0:
            kwargs['persistent_workers'] = True
            if 'prefetch_factor' in self.cfig:
                kwargs['prefetch_factor'] = self.cfig['prefetch_factor']
                
        return DataLoader(dataset, **kwargs)

    def train_val_dataloader(self):
        return self.val_dataloader()
    
    def test_dataloader(self):
        dataset = ProcessedSliceDataset(data_root=self.test_root, cfig=self.cfig, phase='test')
        
        kwargs = {
            'batch_size': self.cfig['val_bs'],
            'shuffle': False,
            'num_workers': self.cfig['num_workers'],
        }
        if torch.cuda.is_available():
             kwargs['pin_memory'] = True
        if self.cfig['num_workers'] > 0:
            kwargs['persistent_workers'] = True
            if 'prefetch_factor' in self.cfig:
                kwargs['prefetch_factor'] = self.cfig['prefetch_factor']
                
        return DataLoader(dataset, **kwargs)


if __name__ == '__main__':
    cfig_path = 'config_files/config_DinoUnetr.yaml'
    
    if os.path.exists(cfig_path):
        cfig = yaml.load(open(cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)
        loader_params = cfig['loader_params']
    else:
        print("Config file not found, using default parameters for testing.")
        loader_params = {
            'train_bs': 4,
            'val_bs': 4,
            'num_workers': 2,
            'prefetch_factor': 2
        }

    loaders = GetLoader(cfig=loader_params)
    
    try:
        train_loader = loaders.train_dataloader()
        print("\nStarting DataLoader test...")
        
        for batch_idx, data_dict in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  - Data shape: {data_dict['data'].shape}")
            print(f"  - Label shape: {data_dict['label'].shape}")
            print(f"  - Body shape: {data_dict['body'].shape}")
            print(f"  - IDs: {data_dict['id']}")
            
            if batch_idx >= 2:
                print("Test finished successfully.")
                break
    except Exception as e:
        print(f"\n[Test Failed] Could not load data. Reason: {e}")
        print("Check whether the dataset folder and .npz files exist.")