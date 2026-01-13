from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
import json
import pdb
import os
import zipfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from toolkit import *

HaN_OAR_LIST = [ 'Cochlea_L', 'Cochlea_R','Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'Chiasim', 'LacrimalGlands', 'BrachialPlexus', 'Brain',  'BrainStem_03',  'Esophagus', 'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV', 'OCavity-PTV', 'ParotidCon-PTV', 'Parotidlps-PTV', 'Parotids-PTV', 'PharConst-PTV', 'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05']

HaN_OAR_DICT = {HaN_OAR_LIST[i]: (i+1) for i in range(len(HaN_OAR_LIST))}

Lung_OAR_LIST = ["PTV_Ring.3-2", "Total Lung-GTV", "SpinalCord",  "Heart",  "LAD", "Esophagus",  "BrachialPlexus",  "GreatVessels", "Trachea", "Body_Ring0-3"]

Lung_OAR_DICT = {Lung_OAR_LIST[i]: (i+10) for i in range(len(Lung_OAR_LIST))}


class CreateDataset():
    
    def __init__(self, cfig, phase):
        '''
        phase: train, validation, or testing 
        
        cfig: the configuration dictionary
        
            train_bs: training batch size
            val_bs: validation batch size
            num_workers: the number of workers when call the DataLoader of PyTorch
            
            csv_root: the meta data file, include patient id, plan id, the .npz data path and some conditions of the plan. 
            scale_dose_dict: path of a dictionary. The dictionary includes the prescribed doses of the PTVs. 
            pat_obj_dict: path of a dictionary. The dictionary includes the ROIs (PTVs and OARs) names used in optimization. 
            
            down_HU: bottom clip of the CT HU value. 
            up_HU: upper clip of the CT HU value. 
            denom_norm_HU: the denominator when normalizing the CT. 
            
            in_size & out_size: the size parameters used in data transformation. 

            norm_oar: True or False. Normalize the OAR channel or not. 
            CatStructures: True or False. Concat the PTVs and OARs in multiple channels, or merge them in one channel, respectively. 

            dose_div_factor: the value used to normalize dose. 
            
        '''
        
        self.cfig = cfig
        df = pd.read_csv(cfig['csv_root'])
        df = df.loc[df['phase'] == phase]

        self.phase = phase
        # Panda Series转化为Python List
        self.data_list = df['npz_path'].tolist()
        self.site_list = df['site'].tolist()
        self.cohort_list = df['cohort'].tolist()

        # 读取每一例的高低剂量区的索引 和 勾画名称索引
        self.scale_dose_Dict = json.load(open(cfig['scale_dose_dict'], 'r'))
        self.pat_obj_dict = json.load(open(cfig['pat_obj_dict'], 'r'))
        # 读取HaN OAR优先级表 和 isDmax标志
        df_han_oar = pd.read_csv(cfig['csv_HaN_OAR_priority_root'])
        self.HaN_OAR_name = df_han_oar['OAR_Name'].tolist()
        self.HaN_OAR_priority = df_han_oar['Priority'].tolist()
        # 构建HaN OAR优先级的关联字典
        self.HaN_OAR_PRIORITY_DICT = dict(zip(self.HaN_OAR_name, self.HaN_OAR_priority))
        self.HaN_isDmax = df_han_oar['isDmax'].tolist()
        # 构建HaN OAR isDmax的关联字典
        self.HaN_isDmax_DICT = dict(zip(self.HaN_OAR_name, self.HaN_isDmax))

        # 读取Lung OAR优先级表 和 isDmax标志
        df_lung_oar = pd.read_csv(cfig['csv_LUNG_OAR_priority_root'])
        self.Lung_OAR_name = df_lung_oar['OAR_Name'].tolist()
        self.Lung_OAR_priority = df_lung_oar['Priority'].tolist()
        # 构建Lung OAR优先级的关联字典
        self.Lung_OAR_PRIORITY_DICT = dict(zip(self.Lung_OAR_name, self.Lung_OAR_priority))
        self.Lung_isDmax = df_lung_oar['isDmax'].tolist()
        # 构建Lung OAR isDmax的关联字典
        self.Lung_isDmax_DICT = dict(zip(self.Lung_OAR_name, self.Lung_isDmax))

    
    def create_dataset(self):
        # ensure a log folder exists
        log_dir = os.path.join(os.path.dirname(self.cfig.get('dataset_save_root', '.')), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        bad_log_path = os.path.join(log_dir, 'bad_npz.txt')
        # ensure dataset root exists
        os.makedirs(self.cfig['dataset_save_root'], exist_ok=True)

        lock = threading.Lock()

        def _process_one(index: int):
            """
            In_dict:
                'spacing' 物理间距 (z,y,x)
                'angle_list' 射野角度列表 List
                'ed' HU转化电子密度矩阵 (z,y,x)
                'md' HU转化材料密度矩阵 (z,y,x)
                'img' 经HU值归一化后的CT图像 (z,y,x)
                'dose' 使用D97进行归一化后的Dose图像 (z,y,x)
                'dose_div_factor' 归一化因子
                'OAR' OAR图像 (z,y,x)
                'PTV' PTV图像 (z,y,x)
            save_data_dict:
                'comb_optptv' 合并的PTV处方剂量掩膜 (z,y,x)
                'comb_oar_priority' 合并的OAR优先级图 (z,y,x)
                'comb_oar_distance' 合并的OAR距离图 (z,y,x)
                'mass_density' 材料密度图 (z,y,x)
                'electron_density' 电子密度图 (z,y,x)
                'beam_plate_norm' 束板归一化模板 (z,y,x)
                'oar_serial' 串行OAR掩膜 (z,y,x)
                'oar_parallel' 并行OAR掩膜 (z,y,x)
                'label' 标签图 (z,y,x)
                'prompt' 病例条件提示向量 (1,12)
            """
            data_path = self.data_list[index]
            ID = self.data_list[index].split('/')[-1].replace('.npz', '')
            PatientID = ID.split('+')[0]

            if len(str(PatientID)) < 3:
                PatientID = f"{PatientID:0>3}"

            # robust npz load with diagnostics
            try:
                data_npz = np.load(data_path, allow_pickle=True)
            except Exception as e:
                msg = f"[skip] Failed to load npz: {data_path} | {type(e).__name__}: {e}"
                print(msg)
                try:
                    with lock:
                        with open(bad_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"{ID}\t{data_path}\t{type(e).__name__}: {e}\n")
                except Exception:
                    pass
                try:
                    if not zipfile.is_zipfile(data_path):
                        print(f"[hint] File is not a valid zip/npz: {data_path}. Re-download or fix extension.")
                except Exception:
                    pass
                return False

            # extract dict from npz (expect arr_0 to hold the dict)
            try:
                arr0 = dict(data_npz).get('arr_0', None)
                if arr0 is None:
                    raise KeyError("arr_0 not found in npz")
                In_dict = arr0.item() if isinstance(arr0, np.ndarray) else arr0
                if not isinstance(In_dict, dict):
                    raise TypeError("arr_0 did not contain a Python dict")
            except Exception as e:
                msg = f"[skip] Bad npz content (arr_0): {data_path} | {type(e).__name__}: {e}"
                print(msg)
                try:
                    with lock:
                        with open(bad_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"{ID}\t{data_path}\tBad arr_0: {type(e).__name__}: {e}\n")
                except Exception:
                    pass
                return False

            spacing = [2.0,2.5,2.5]
            In_dict['spacing'] = spacing # To be validated
            angle_list = In_dict['angle_list']
            In_dict['ed'] = HU2electron_density(In_dict['img']) * In_dict['Body'] 
            In_dict['md'] = HU2mass_density(In_dict['img']) * In_dict['Body'] 
            In_dict['img'] = np.clip(In_dict['img'], self.cfig['down_HU'], self.cfig['up_HU']) / self.cfig['denom_norm_HU'] * In_dict['Body'] 

            if 'dose' in In_dict.keys():
                ptv_highdose =  self.scale_dose_Dict[PatientID]['PTV_High']['PDose'] # 高剂量PTV的值
                In_dict['dose'] = In_dict['dose'] * In_dict['dose_scale'] # 缩放因子后的Dose图像
                PTVHighOPT = self.scale_dose_Dict[PatientID]['PTV_High']['OPTName'] # 高剂量PTV的名称
                norm_scale = ptv_highdose / (np.percentile(In_dict['dose'][In_dict[PTVHighOPT].astype('bool')], 3) + 1e-5) # D97 归一化因子
                In_dict['dose'] = In_dict['dose'] * norm_scale / self.cfig['dose_div_factor'] # 归一化后的Dose图像
                In_dict['dose'] = np.clip(In_dict['dose'], 0, ptv_highdose * 1.2) # 截断到0-ptv_highdose*1.2之间

            isocenter = In_dict['isocenter']

            KEYS = list(In_dict.keys())
            for key in list(In_dict.keys()):
                if isinstance(In_dict[key], np.ndarray) and len(In_dict[key].shape) == 3:
                    In_dict[key] = torch.from_numpy(In_dict[key].astype('float'))[None]
                else:
                    if key in KEYS:
                        KEYS.remove(key)

            if self.site_list[index] < 1.5: # 判断ct区域
                OAR_LIST = self.HaN_OAR_name
                OAR_PRIORITY = self.HaN_OAR_PRIORITY_DICT
                OAR_isDmax = self.HaN_isDmax_DICT
            else:
                OAR_LIST = self.Lung_OAR_name
                OAR_PRIORITY = self.Lung_OAR_PRIORITY_DICT
                OAR_isDmax = self.Lung_isDmax_DICT

            try:
                need_list = self.pat_obj_dict[ID.split('+')[0]]
            except Exception:
                need_list = OAR_LIST
                print(ID.split('+')[0], '-------------not in the pat_obj_dict')

            comb_oar_priority  = combine_oar_priority(In_dict, need_list, OAR_PRIORITY)

            tmp_oar_dict = {}
            try:
                tmp_oar_dict['img'] = np.squeeze(In_dict['Body'].to('cpu').numpy())
            except Exception:
                tmp_oar_dict['img'] = np.zeros(np.squeeze(In_dict['img'].to('cpu').numpy()).shape, dtype=np.float32)
            for k in OAR_PRIORITY.keys():
                if k in In_dict.keys():
                    try:
                        tmp_oar_dict[k] = np.squeeze(In_dict[k].to('cpu').numpy())
                    except Exception:
                        tmp_oar_dict[k] = np.zeros_like(tmp_oar_dict['img'], dtype=np.float32)
                else:
                    tmp_oar_dict[k] = np.zeros_like(tmp_oar_dict['img'], dtype=np.float32)
            oar_serial_np, oar_parallel_np = oar_mask(tmp_oar_dict, need_list, OAR_PRIORITY, OAR_isDmax)

            opt_dose_dict = {}
            for key in self.scale_dose_Dict[PatientID].keys():
                if key in ['PTV_High', 'PTV_Mid', 'PTV_Low']:
                    opt_dose_dict[self.scale_dose_Dict[PatientID][key]['OPTName']] = self.scale_dose_Dict[PatientID][key]['PDose'] / self.cfig['dose_div_factor']

            comb_optptv, prs_opt, cat_optptv = combine_ptv(In_dict, opt_dose_dict)

            if 'dose' in In_dict.keys():
                label = In_dict['dose'] * In_dict['Body'] 

            save_data_dict = dict()
            save_data_dict['prompt'] = [In_dict['isVMAT'], len(prs_opt), self.site_list[index], self.cohort_list[index]]
            save_data_dict['comb_optptv'] = np.squeeze(comb_optptv.to('cpu').numpy())
            save_data_dict['comb_oar_priority'] = np.squeeze(comb_oar_priority.to('cpu').numpy())
            save_data_dict['img'] = np.squeeze(In_dict['img'].to('cpu').numpy())
            distance_map = calculate_min_distance_to_tumor_surface(save_data_dict['comb_optptv'], In_dict['spacing']) ##mm
            save_data_dict['comb_oar_distance'] = calculate_distance_to_tumor(In_dict, distance_map, need_list, OAR_PRIORITY)

            save_data_dict['Body'] = np.squeeze(In_dict['Body'].to('cpu').numpy())
            save_data_dict['electron_density'] = np.squeeze(In_dict['ed'].to('cpu').numpy())
            save_data_dict['mass_density'] = np.squeeze(In_dict['md'].to('cpu').numpy())
            save_data_dict['oar_serial'] = (oar_serial_np * save_data_dict['Body']).astype(np.float32)
            save_data_dict['oar_parallel'] = (oar_parallel_np * save_data_dict['Body']).astype(np.float32)

            if 'dose' in In_dict.keys():
                save_data_dict['label'] = np.squeeze(label.to('cpu').numpy())

            if self.site_list[index] < 1.5:
                save_data_dict['PTVHigh'] = np.squeeze(In_dict['PTVHighOPT'].to('cpu').numpy())
            else:
                save_data_dict['PTVHigh'] = np.squeeze(In_dict['PTV'].to('cpu').numpy())
            save_data_dict['PTV'] = save_data_dict['comb_optptv'] > 0
            save_data_dict['PTV_expanded'] = expand_roi(save_data_dict['PTV'], In_dict['spacing'], 15)
            save_data_dict['isocenter'] = isocenter

            try:
                beam_plate = np.squeeze(In_dict['beam_plate'].to('cpu').numpy())
                denom = float(beam_plate[int(isocenter[0]), int(isocenter[1]), int(isocenter[2])])
                if abs(denom) < 1e-6:
                    denom = max(float(beam_plate.max()), 1.0)
                beam_plate_norm = (beam_plate / denom) * save_data_dict['Body']
            except Exception:
                beam_plate = get_allbeam_plate(save_data_dict['PTV'], isocenter, spacing, angle_list, with_distance=True)
                beam_plate_norm = (beam_plate / max(len(angle_list), 1)) * save_data_dict['Body']
            beam_plate_norm = np.nan_to_num(beam_plate_norm, nan=0.0, posinf=0.0, neginf=0.0)

            save_data_dict['beam_plate'] = beam_plate
            save_data_dict['beam_plate_norm'] = beam_plate_norm

            try:
                np.savez_compressed(f"{self.cfig['dataset_save_root']}/{ID}.npz", arr_0=save_data_dict)
            except Exception as e:
                print(f"[skip] Failed to save npz for {ID}: {type(e).__name__}: {e}")
                return False

            del In_dict
            return True

        max_workers = 4
        total_cases = len(self.data_list)
        success_count = 0
        fail_count = 0
        if max_workers <= 1:
            with tqdm(total=total_cases, desc='Preprocess', unit='case') as pbar:
                for idx in range(total_cases):
                    res = _process_one(idx)
                    if res:
                        success_count += 1
                    else:
                        fail_count += 1
                    pbar.update(1)
        else:
            print(f"[INFO] Multithread preprocessing with {max_workers} workers for {total_cases} cases.")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_process_one, idx): idx for idx in range(total_cases)}
                with tqdm(total=total_cases, desc='Preprocess', unit='case') as pbar:
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res:
                            success_count += 1
                        else:
                            fail_count += 1
                        pbar.update(1)
        print(f"[INFO] Preprocess done: {success_count} success, {fail_count} failed, total {total_cases}.")
        return 
    
class GetLoader(object):
    def __init__(self, cfig):
        super().__init__()
        self.cfig = cfig

    
    def dataset_creation(self):
        dataset = CreateDataset(self.cfig, phase='train')
        dataset.create_dataset()
        return 
    

if __name__ == '__main__':

    cfig = {
            'csv_root': 'meta_files/meta_data.csv',
            'csv_HaN_OAR_priority_root': 'meta_files/HaN_OAR_update.csv',
            'csv_LUNG_OAR_priority_root': 'meta_files/LUNG_OAR_update.csv',
            'scale_dose_dict': 'meta_files/PTV_DICT.json',
            'pat_obj_dict': 'meta_files/Pat_Obj_DICT.json',
            'dataset_save_root': 'data/dataset_update_0214',
            'down_HU': -1000,
            'up_HU': 1000,
            'denom_norm_HU': 500,
            'dose_div_factor': 10 
            }
    
    loaders = GetLoader(cfig)
    tqdm(loaders.dataset_creation())
