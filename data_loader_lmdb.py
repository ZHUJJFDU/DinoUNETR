from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import yaml
import os
import lmdb
import pickle
import json

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, cfig, phase='train'):
        self.lmdb_path = lmdb_path
        self.phase = phase
        self.cfig = cfig
        self.env = None
        
        # --- Transforms Setup (Consistent with Slice Loader) ---
        # self.out_size = cfig.get('out_size', [96, 256, 256]) # [D, H, W] -> use last 2
        # self.target_h = self.out_size[1]
        # self.target_w = self.out_size[2]
        
        # self.crop_train = RandSpatialCrop(roi_size=[self.target_h, self.target_w], random_size=False)
        # self.crop_valid = CenterSpatialCrop(roi_size=[self.target_h, self.target_w])

        
        if not os.path.exists(lmdb_path):
             raise ValueError(f"LMDB path {lmdb_path} does not exist.")

        # Open env temporarily to extract length
        # lock=False is important for concurrent reading if multiple workers are used, 
        # but pure reading usually doesn't need write lock. 
        # readonly=True is safer.
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin(write=False) as txn:
            self.length = int(txn.get('__len__'.encode()).decode())
        env.close()
        
        print(f"[{phase}] Initialized LMDB dataset from {lmdb_path}. Total samples: {self.length}")

    def _init_db(self):
        # We handle env opening inside getitem/worker to strictly avoid pickling issues with multiprocessing
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()
            
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(str(idx).encode())
        
        # Deserialize
        sample = pickle.loads(byteflow)
        
        # Extract and convert to Tensor
        data = torch.from_numpy(sample['data']).float() # (C, H, W)
        label = torch.from_numpy(sample['label']).float()
        body = torch.from_numpy(sample['body']).float()
        
        # Fallbacks for optional fields
        ptv = torch.from_numpy(sample.get('ptv', np.zeros_like(sample['label']))).float()
        oar_serial = torch.from_numpy(sample.get('oar_serial', np.zeros_like(sample['label']))).float()
        oar_parallel = torch.from_numpy(sample.get('oar_parallel', np.zeros_like(sample['label']))).float()
        
        isocenter = torch.from_numpy(sample.get('isocenter', np.zeros(3))).float()
        spacing = torch.from_numpy(sample.get('spacing', np.ones(3))).float()
        
        angle_list = sample.get('angle_list', [])
        # Serialize angle_list to JSON string to match previous behavior for collate_fn
        angle_list_str = json.dumps(angle_list)

        # --- Safety Preprocessing (Same as naive loader) ---
        if torch.isnan(label).any() or torch.isinf(label).any():
            label = torch.nan_to_num(label, nan=0.0, posinf=7.5, neginf=0.0)
        label = torch.clamp(label, 0, 7.5)

        # --- Online Cropping Logic ---
        # Check if crop is needed (data H/W > target H/W)
        # _, h, w = data.shape
        # if h > self.target_h or w > self.target_w:
            # Stack all tensors to crop consistentlY
            # data: (C,H,W), label: (1,H,W), body: (1,H,W), ptv: (1,H,W), serial: (1,H,W), parallel: (1,H,W)
            # c_data = data.shape[0]
            # stack = torch.cat([data, label, body, ptv, oar_serial, oar_parallel], dim=0)
            
            # if self.phase == 'train':
            #     stack_cropped = self.crop_train(stack)
            # else:
            #     stack_cropped = self.crop_valid(stack)
            
            # # Unstack
            # current_idx = 0
            # data = stack_cropped[current_idx : current_idx+c_data]
            # current_idx += c_data
            
            # label = stack_cropped[current_idx : current_idx+1]
            # current_idx += 1
            
            # body = stack_cropped[current_idx : current_idx+1]
            # current_idx += 1
            
            # ptv = stack_cropped[current_idx : current_idx+1]
            # current_idx += 1
            
            # oar_serial = stack_cropped[current_idx : current_idx+1]
            # current_idx += 1
            
            # oar_parallel = stack_cropped[current_idx : current_idx+1]
            # current_idx += 1

        data_dict = {
            'data': data,
            'label': label,
            'body': body,
            'ptv': ptv,
            'oar_serial': oar_serial,
            'oar_parallel': oar_parallel,
            'isocenter': isocenter,
            'spacing': spacing,
            'angle_list': angle_list_str,
            'id': sample['id']
        }
        
        return data_dict

class GetLoader(object):
    def __init__(self, cfig):
        self.cfig = cfig
        
        # NOTE: Using hardcoded paths as requested, matching data_loader_lightning_slice.py pattern
        # but pointing to the corresponding LMDB files.
        self.train_root = r'D:/data/Dataset_512/Train.lmdb'
        self.valid_root = r'D:/data/Dataset_512/Valid.lmdb'
        self.test_root = r'D:/data/Dataset_512/Test.lmdb'
        
    def train_dataloader(self):
        dataset = LMDBDataset(lmdb_path=self.train_root, phase='train')
        
        kwargs = {
            'batch_size': self.cfig['train_bs'],
            'shuffle': True,
            'num_workers': self.cfig['num_workers'],
            'pin_memory': True if torch.cuda.is_available() else False,
        }
        if self.cfig['num_workers'] > 0:
            kwargs['persistent_workers'] = True
            if 'prefetch_factor' in self.cfig:
                kwargs['prefetch_factor'] = self.cfig['prefetch_factor']
                
        return DataLoader(dataset, **kwargs)

    def val_dataloader(self):
        dataset = LMDBDataset(lmdb_path=self.valid_root, phase='valid')
        
        kwargs = {
            'batch_size': self.cfig['val_bs'],
            'shuffle': False,
            'num_workers': self.cfig['num_workers'],
            'pin_memory': True if torch.cuda.is_available() else False,
        }
        if self.cfig['num_workers'] > 0:
            kwargs['persistent_workers'] = True
            
        return DataLoader(dataset, **kwargs)
    
    def test_dataloader(self):
        dataset = LMDBDataset(lmdb_path=self.test_root, phase='test')
        
        kwargs = {
            'batch_size': self.cfig['val_bs'], 
            'shuffle': False,
            'num_workers': self.cfig['num_workers'],
            'pin_memory': True if torch.cuda.is_available() else False,
        }
        return DataLoader(dataset, **kwargs)

if __name__ == '__main__':
    # Test stub
    pass
