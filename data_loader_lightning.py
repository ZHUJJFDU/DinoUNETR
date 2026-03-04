from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
import json
import pdb
import time
from scipy import ndimage 
from toolkit import *
import yaml
import argparse 
import os
from monai.transforms import SpatialPad


HaN_OAR_LIST = [ 'Cochlea_L', 'Cochlea_R','Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'Chiasim', 'LacrimalGlands', 'BrachialPlexus', 'Brain',  'BrainStem_03',  'Esophagus', 'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV', 'OCavity-PTV', 'ParotidCon-PTV', 'Parotidlps-PTV', 'Parotids-PTV', 'PharConst-PTV', 'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05']

HaN_OAR_DICT = {HaN_OAR_LIST[i]: (i+1) for i in range(len(HaN_OAR_LIST))}

Lung_OAR_LIST = ["PTV_Ring.3-2", "Total Lung-GTV", "SpinalCord",  "Heart",  "LAD", "Esophagus",  "BrachialPlexus",  "GreatVessels", "Trachea", "Body_Ring0-3"]

Lung_OAR_DICT = {Lung_OAR_LIST[i]: (i+10) for i in range(len(Lung_OAR_LIST))}


class MyDataset(Dataset):
    
    def __init__(self, cfig, phase, dev_split):
        
        self.cfig = cfig
        
        df = pd.read_csv(cfig['csv_root'])
        
        df = df.loc[(df['phase'] == phase) & (df['dev_split'] == dev_split)] # 筛选phase和dev_split对应的样本

        self.phase = phase
        self.dev_split = dev_split

        # Pandas Series转化为Python List 后续操作更快
        self.data_list = df['npz_path'].tolist()
        self.site_list = df['site'].tolist()
        self.cohort_list = df['cohort'].tolist()

        self.scale_dose_Dict = json.load(open(cfig['scale_dose_dict'], 'r'))
        self.pat_obj_dict = json.load(open(cfig['pat_obj_dict'], 'r'))


    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        """
        data_dict:
            'label' 目标标签 (z,y,x)
            'ref_5Gy_mask' 大于5Gy掩码 (z,y,x)
            'data' 	带处方剂量的PTVs 带优先级权重的OARs OAR到PTV的距离图 人体二进制掩模 质量密度图 归一化距离感知射束板 6个向量沿通道维度拼接在一起
            'Body' 人体二进制掩模 (z,y,x)
            'PTV_expanded' 扩展后的经用人体掩膜的PTV掩模 (z,y,x)
            'PTV' 经用人体掩膜的PTV掩模 (z,y,x)
            'oar_serial' 经用人体掩膜的串行器官掩模 (z,y,x)
            'oar_parallel' 经用人体掩膜的并行器官掩模 (z,y,x)
            'ori_isocenter' 原始数据旋转中心坐标 (z,y,x)
            'ori_img_size' 原始数据大小 (z,y,x)
            'id' 样本ID 字符串类型
        """ 

        data_path = self.data_list[index]
        ID = self.data_list[index].split('/')[-1].replace('.npz', '')
        # PatientID = ID.split('+')[0]

        # if len(str(PatientID)) < 3:
        #     PatientID = f"{PatientID:0>3}"

        data_npz = np.load(data_path, allow_pickle=True)


        In_dict = dict(data_npz)['arr_0'].item()

        isocenter = In_dict['isocenter']
        spacing = In_dict['spacing']
        angle_list = In_dict['angle_list']
        ori_img_size = In_dict['Body'].shape

        
        KEYS = list(In_dict.keys())
        for key in In_dict.keys(): 
            if isinstance(In_dict[key], np.ndarray) and len(In_dict[key].shape) == 3: # 是否是3维NumPy数组类型
                In_dict[key] = torch.from_numpy(In_dict[key].astype('float'))[None] # 转换为PyTorch Tensor 并添加通道维度
            else:
                KEYS.remove(key)
        # 训练集可选数据增强
        if self.phase == 'train':
            if 'with_aug' in self.cfig.keys() and not self.cfig['with_aug']:
                self.aug = tt_augmentation(KEYS, self.cfig['in_size'],  self.cfig['out_size'], isocenter)
            else:
                self.aug = tr_augmentation(KEYS, self.cfig['in_size'], self.cfig['out_size'], isocenter)
        # 测试集不使用数据增强
        if self.phase in ['val', 'test', 'valid', 'external_test'] or self.dev_split in ['test','valid']:
            self.aug = tt_augmentation(KEYS, self.cfig['in_size'], self.cfig['out_size'], isocenter)

        # 应用数据增强
        In_dict = self.aug(In_dict)
        for k in list(In_dict.keys()):
            v = In_dict[k]
            if isinstance(v, torch.Tensor) and v.dim() == 4:
                v[torch.isnan(v)] = 0
                v[torch.isinf(v)] = 0
                In_dict[k] = v
        # 创建一个空字典
        data_dict = dict()

        if 'label' in In_dict.keys():
            data_dict['label'] = In_dict['label']
            ref_dose = In_dict['label'] * 1
            data_dict['ref_5Gy_mask'] = (ref_dose > 5) & (In_dict['Body'] > 0)

        In_dict['Body'] = (In_dict['Body'] > 0.5).type(torch.FloatTensor)
        In_dict['PTV_expanded'] = (In_dict['PTV_expanded'] > 0.5).type(torch.FloatTensor)

        # data_dict['data'] = torch.cat((
        #     In_dict['comb_optptv'],  
        #     In_dict['comb_oar_priority'],  
        #     In_dict['comb_oar_distance'], 
        #     In_dict['Body'], 
        #     In_dict['mass_density'], 
        #     In_dict['beam_plate_norm']), axis=0)

        data_dict['data'] = torch.cat((
            In_dict['mass_density'], 
            In_dict['comb_optptv'],  
            In_dict['comb_oar_priority'],  
            In_dict['beam_plate_norm'],
            In_dict['comb_oar_distance'], 
            In_dict['Body']), axis=0)

        data_dict['Body'] = In_dict['Body']

        data_dict['PTV_expanded'] = In_dict['PTV_expanded'] * In_dict['Body']
        data_dict['PTV'] = In_dict['PTV'] * In_dict['Body']
        data_dict['oar_serial'] = In_dict['oar_serial'] * In_dict['Body']
        data_dict['oar_parallel'] = In_dict['oar_parallel'] * In_dict['Body']
        
        # layout tokens
        data_dict['ori_isocenter'] = torch.tensor(isocenter)
        data_dict['spacing'] = torch.tensor(spacing)
        # Keep angle_list as raw list to handle variable lengths; network will handle resampling
        data_dict['angle_list'] = angle_list

        data_dict['ori_img_size'] = torch.tensor(ori_img_size)
        data_dict['id'] = ID
        # data_dict['direction'] = torch.tensor(In_dict['direction'])
        
        del In_dict
        # print(data_dict['data'].shape) # C D H W
        # print(data_dict['label'].shape) # C D H W

        return data_dict
    
class GetLoader(object):
    def __init__(self, cfig):
        super().__init__()
        self.cfig = cfig
        
    def train_dataloader(self):
        dataset_3d = MyDataset(self.cfig, phase='train', dev_split='train') 

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
        return DataLoader(dataset_3d, **kwargs)

    def train_val_dataloader(self):
        dataset_3d = MyDataset(self.cfig, phase='train', dev_split = 'valid') 

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
        return DataLoader(dataset_3d, **kwargs)
    
    def val_dataloader(self):
        dataset_3d = MyDataset(self.cfig, phase='valid', dev_split = 'test') 
        
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
        return DataLoader(dataset_3d, **kwargs)
    
    def test_dataloader(self):
        dataset_3d = MyDataset(self.cfig, phase='test') 
        
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
        
        return DataLoader(dataset_3d, **kwargs)


if __name__ == '__main__':

    cfig_path = 'config_files\config_DinoUnetr.yaml'

    cfig = yaml.load(open(cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)

    device = torch.device("cuda:0")
    # ------------ data loader -----------------#
    loaders = GetLoader(cfig = cfig['loader_params'])
    train_loader =loaders.train_dataloader()
    
    print("Starting DataLoader test...")
    for batch_idx, data_dict in enumerate(train_loader):
            # Forward pass
            print(f"Batch {batch_idx}: Data shape: {data_dict['data'].shape}, Label shape: {data_dict['label'].shape}")
            # 只测试前几个 batch 即可，避免跑太久
            if batch_idx >= 2:
                print("Test finished.")
                break