import os
import yaml
import numpy as np
import torch
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader_lightning import MyDataset
from toolkit import tt_augmentation

# Global state for threaded workers
GLOBAL_DATASET = None
SAVE_ROOT = None

class ProcessDataset(MyDataset):
    """Keep original depth while resizing only in-plane dimensions."""
    def __getitem__(self, index):
        data_path = self.data_list[index]
        ID = self.data_list[index].split('/')[-1].replace('.npz', '')
        
        data_npz = np.load(data_path, allow_pickle=True)
        In_dict = dict(data_npz)['arr_0'].item()

        isocenter = In_dict['isocenter']
        spacing = In_dict['spacing']
        angle_list = In_dict['angle_list']
        ori_img_size = In_dict['Body'].shape # (D, H, W)

        target_z = ori_img_size[0]
        target_h = self.cfig['out_size'][1]
        target_w = self.cfig['out_size'][2]
        
        target_size = [target_z, target_h, target_w]

        KEYS = list(In_dict.keys())
        for key in In_dict.keys(): 
            if isinstance(In_dict[key], np.ndarray) and len(In_dict[key].shape) == 3: 
                In_dict[key] = torch.from_numpy(In_dict[key].astype('float'))[None] 
            else:
                KEYS.remove(key)

        self.aug = tt_augmentation(KEYS, self.cfig['in_size'], target_size, isocenter)

        In_dict = self.aug(In_dict)
        for k in list(In_dict.keys()):
            v = In_dict[k]
            if isinstance(v, torch.Tensor) and v.dim() == 4:
                v[torch.isnan(v)] = 0
                v[torch.isinf(v)] = 0
                In_dict[k] = v
        
        data_dict = dict()

        if 'label' in In_dict.keys():
            data_dict['label'] = In_dict['label']

        In_dict['Body'] = (In_dict['Body'] > 0.5).type(torch.FloatTensor)
        if 'PTV_expanded' in In_dict:
            In_dict['PTV_expanded'] = (In_dict['PTV_expanded'] > 0.5).type(torch.FloatTensor)

        # Paper order: CT, PTV prescription, OAR priority, body, beam, distance.
        try:
             data_dict['data'] = torch.cat((
                In_dict['mass_density'], 
                In_dict['comb_optptv'],  
                In_dict['comb_oar_priority'],  
                In_dict['Body'],
                In_dict['beam_plate_norm'],
                In_dict['comb_oar_distance'],
            ), axis=0)
        except KeyError as e:
            print(f"Warning: Missing key {e} for {ID}")
            raise e

        data_dict['Body'] = In_dict['Body']

        if 'PTV' in In_dict: data_dict['PTV'] = In_dict['PTV'] * In_dict['Body']
        if 'oar_serial' in In_dict: data_dict['oar_serial'] = In_dict['oar_serial'] * In_dict['Body']
        if 'oar_parallel' in In_dict: data_dict['oar_parallel'] = In_dict['oar_parallel'] * In_dict['Body']
        
        data_dict['ori_isocenter'] = torch.tensor(isocenter)
        data_dict['spacing'] = torch.tensor(spacing)
        data_dict['angle_list'] = angle_list
        data_dict['ori_img_size'] = torch.tensor(ori_img_size)
        data_dict['id'] = ID
        
        del In_dict
        return data_dict

def process_one_case(index):
    """Process a single case into per-slice NPZ files."""
    try:
        data_dict = GLOBAL_DATASET[index]
        case_id = data_dict['id']
        
        img_3d = data_dict['data']   
        label_3d = data_dict.get('label', None)
        body_3d = data_dict['Body']
        
        ptv_3d = data_dict.get('PTV', None)
        oar_serial_3d = data_dict.get('oar_serial', None)
        oar_parallel_3d = data_dict.get('oar_parallel', None)

        isocenter = data_dict['ori_isocenter']
        spacing = data_dict['spacing']
        angle_list = data_dict['angle_list']
        
        depth = img_3d.shape[1] 
        
        for d in range(depth):
            slice_data = img_3d[:, d, :, :].clone().numpy()
            slice_body = body_3d[:, d, :, :].clone().numpy()
            
            slice_label = label_3d[:, d, :, :].clone().numpy() if label_3d is not None else None
            slice_ptv = ptv_3d[:, d, :, :].clone().numpy() if ptv_3d is not None else None
            slice_oar_serial = oar_serial_3d[:, d, :, :].clone().numpy() if oar_serial_3d is not None else None
            slice_oar_parallel = oar_parallel_3d[:, d, :, :].clone().numpy() if oar_parallel_3d is not None else None
            
            save_name = f"{case_id}_{d:03d}.npz"
            save_path = os.path.join(SAVE_ROOT, save_name)
            
            save_dict = {
                'data': slice_data,
                'body': slice_body,
                'isocenter': isocenter,
                'spacing': spacing,
                'angle_list': np.array(angle_list, dtype=object)
            }
            if slice_label is not None: save_dict['label'] = slice_label
            if slice_ptv is not None: save_dict['ptv'] = slice_ptv
            if slice_oar_serial is not None: save_dict['oar_serial'] = slice_oar_serial
            if slice_oar_parallel is not None: save_dict['oar_parallel'] = slice_oar_parallel
            
            np.savez_compressed(save_path, **save_dict)
            
        return f"Success: {case_id} ({depth} slices)"
    
    except Exception as e:
        return f"Error processing index {index}: {str(e)}"

def main():
    global GLOBAL_DATASET, SAVE_ROOT
    
    cfig_path = 'config_files/config_train.yaml'
    max_workers = 4 

    print("Loading configuration...")
    cfig = yaml.load(open(cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)

    loader_config = cfig['loader_params']
    tasks = [
        (loader_config.get('train_root', 'Dataset_256_DoseDINO/Train'), 'train', 'train'),
        (loader_config.get('valid_root', 'Dataset_256_DoseDINO/Valid'), 'train', 'valid'),
        (loader_config.get('test_root', 'Dataset_256_DoseDINO/Test'), 'valid', 'test'),
    ]

    for task_idx, (save_dir, phase, dev_split) in enumerate(tasks):
        print(f"\n{'='*20} Task {task_idx+1}/{len(tasks)}: Processing {phase}/{dev_split} {'='*20}")
        
        SAVE_ROOT = save_dir
        os.makedirs(SAVE_ROOT, exist_ok=True)
        
        print(f"Initializing ProcessDataset (Dynamic Z, Fixed XY) for phase='{phase}'...")
        GLOBAL_DATASET = ProcessDataset(cfig['loader_params'], phase=phase, dev_split=dev_split)
        
        total_cases = len(GLOBAL_DATASET)
        print(f"Dataset initialized. Total volumes: {total_cases}")
        print(f"Output directory: {SAVE_ROOT}")
        print(f"Starting processing with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_one_case, i): i for i in range(total_cases)}
            
            for future in tqdm(as_completed(future_to_idx), total=total_cases, unit="case"):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if "Error" in result:
                        print(f"\n[Warning] {result}")
                except Exception as exc:
                    print(f"\n[Critical Error] Case {idx}: {exc}")

    print("\nAll processing tasks finished!")

if __name__ == '__main__':
    main()
