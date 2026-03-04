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
    """
    专门用于读取预处理后的 2D 切片数据 (.npz)
    """
    def __init__(self, data_root, cfig, phase='train'):
        self.data_root = data_root
        self.phase = phase
        self.cfig = cfig
        
        search_path = os.path.join(data_root, "*.npz")
        self.file_list = glob.glob(search_path)
        
        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_root}. Please check the path.")
            
        print(f"[{phase}] Initialized dataset from {data_root}. Total slices: {len(self.file_list)}")

        # --- Transforms Setup ---
        self.out_size = cfig.get('out_size', [96, 256, 256]) # [D, H, W] but dataset is [H, W] usually
        self.target_h = self.out_size[1]
        self.target_w = self.out_size[2]
        
        self.train_transforms = Compose([
            # 1. 先 Resize 到一个略大的尺寸，或者直接 Resize 到 target
            # 既然要做 Global，先统一尺寸，方便后续处理
            Resized(
                keys=self.keys, 
                spatial_size=target_size, # (256, 256)
                mode=modes
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
            # np.load 读取，显式开启 allow_pickle 以支持 object arrays (如 angle_list)
            npz = np.load(file_path, allow_pickle=True)
            
            data = npz['data'] # (C, H, W)
            label = npz['label'] # (1, H, W) usually
            body = npz['body']            
            
            # 尝试读取 DVH Loss 所需的额外掩膜，如果不存在则使用全 0
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

            # --- Layout Metadata ---
            # 必须与 run_process.py 保存时的键名一致
            if 'isocenter' in npz:
                isocenter = torch.from_numpy(npz['isocenter']).float()
            else:
                isocenter = torch.zeros(3).float() # Fallback

            if 'spacing' in npz:
                spacing = torch.from_numpy(npz['spacing']).float()
            else:
                spacing = torch.ones(3).float() # Fallback

            if 'angle_list' in npz:
                # 之前存的是 object array，取出里面的 list
                # npz['angle_list'] 可能是 array([list([...]), dtype=object])
                raw_angles = npz['angle_list']
                if raw_angles.shape == (): # 0-d array
                    angle_list = raw_angles.item()
                else:
                    angle_list = raw_angles.tolist()
            else:
                angle_list = [] # Fallback
            
            # Serialize angle_list to JSON string to avoid collate errors with variable lengths
            angle_list_str = json.dumps(angle_list)

            # 3. 转为 Tensor
            data = torch.from_numpy(data).float()
            label = torch.from_numpy(label).float()
            body = torch.from_numpy(body).float()
            ptv = torch.from_numpy(ptv).float()
            oar_serial = torch.from_numpy(oar_serial).float()
            oar_parallel = torch.from_numpy(oar_parallel).float()

            # --- Augmentation Logic ---
            # Construct Dictionary for Transforms
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
            
            # Update metadata in dict
            data_dict['isocenter'] = isocenter
            data_dict['spacing'] = spacing
            data_dict['angle_list'] = angle_list_str
            data_dict['id'] = os.path.basename(file_path).replace('.npz', '') 
            
            return data_dict
            
            
            return data_dict
            
        except Exception as e:
            # 遇到错误直接抛出，不再递归重试，以便排查根本原因（通常是数据未生成或路径错误）
            raise RuntimeError(f"Failed to load file at index {idx}: {file_path}. Original error: {str(e)}")

class GetLoader(object):
    def __init__(self, cfig):
        super().__init__()
        self.cfig = cfig
        
        self.train_root = 'Dataset_256_layout_changechannel_nah&lung/Train'
        self.valid_root = 'Dataset_256_layout_changechannel_nah&lung/Valid'
        self.test_root = 'Dataset_256_layout_changechannel_nah&lung/Test'
        
    def train_dataloader(self):
        # 直接实例化新的 Dataset
        dataset = ProcessedSliceDataset(data_root=self.train_root, cfig=self.cfig, phase='train')
        
        kwargs = {
            'batch_size': self.cfig['train_bs'],
            'shuffle': True, # 训练集需要打乱
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
        # 验证集
        dataset = ProcessedSliceDataset(data_root=self.valid_root, cfig=self.cfig, phase='valid')
        
        kwargs = {
            'batch_size': self.cfig['val_bs'],
            'shuffle': False, # 验证集不需要打乱
            'num_workers': self.cfig['num_workers'],
        }
        if torch.cuda.is_available():
            kwargs['pin_memory'] = True
        if self.cfig['num_workers'] > 0:
            kwargs['persistent_workers'] = True
            if 'prefetch_factor' in self.cfig:
                kwargs['prefetch_factor'] = self.cfig['prefetch_factor']
                
        return DataLoader(dataset, **kwargs)

    # 兼容旧代码调用的接口
    def train_val_dataloader(self):
        return self.val_dataloader()
    
    def test_dataloader(self):
        # 测试集
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
    # 测试代码
    cfig_path = 'config_files/config_DinoUnetr.yaml'
    
    # 确保 config 文件能读到，如果路径不对请修改
    if os.path.exists(cfig_path):
        cfig = yaml.load(open(cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)
        loader_params = cfig['loader_params']
    else:
        # 如果找不到 config，给一个默认参数方便测试
        print("Config file not found, using default parameters for testing.")
        loader_params = {
            'train_bs': 4,
            'val_bs': 4,
            'num_workers': 2,
            'prefetch_factor': 2
        }

    # ------------ data loader -----------------#
    loaders = GetLoader(cfig=loader_params)
    
    # 尝试获取训练数据
    try:
        train_loader = loaders.train_dataloader()
        print("\nStarting DataLoader test...")
        
        for batch_idx, data_dict in enumerate(train_loader):
            # Forward pass
            print(f"Batch {batch_idx}:")
            print(f"  - Data shape: {data_dict['data'].shape}")   # 应该是 (B, 6, 224, 224)
            print(f"  - Label shape: {data_dict['label'].shape}") # 应该是 (B, 1, 224, 224)
            print(f"  - Body shape: {data_dict['body'].shape}") # 应该是 (B, 1, 224, 224)
            print(f"  - IDs: {data_dict['id']}")
            
            # 只测试前几个 batch 即可
            if batch_idx >= 2:
                print("Test finished successfully.")
                break
    except Exception as e:
        print(f"\n[Test Failed] Could not load data. Reason: {e}")
        print("请检查当前目录下是否存在 'Dataset/Train' 文件夹以及里面是否有 .npz 文件")