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
        """Initialize dataset creation for a given split."""
        
        self.cfig = cfig
        df = pd.read_csv(cfig['csv_root'])
        df = df.loc[df['phase'] == phase]

        self.phase = phase
        self.data_list = df['npz_path'].tolist()
        self.site_list = df['site'].tolist()
        self.cohort_list = df['cohort'].tolist()

        self.scale_dose_Dict = json.load(open(cfig['scale_dose_dict'], 'r'))
        self.pat_obj_dict = json.load(open(cfig['pat_obj_dict'], 'r'))
        df_han_oar = pd.read_csv(cfig['csv_HaN_OAR_priority_root'])
        self.HaN_OAR_name = df_han_oar['OAR_Name'].tolist()
        self.HaN_OAR_priority = df_han_oar['Priority'].tolist()
        self.HaN_OAR_PRIORITY_DICT = dict(zip(self.HaN_OAR_name, self.HaN_OAR_priority))
        self.HaN_isDmax = df_han_oar['isDmax'].tolist()
        self.HaN_isDmax_DICT = dict(zip(self.HaN_OAR_name, self.HaN_isDmax))

        df_lung_oar = pd.read_csv(cfig['csv_LUNG_OAR_priority_root'])
        self.Lung_OAR_name = df_lung_oar['OAR_Name'].tolist()
        self.Lung_OAR_priority = df_lung_oar['Priority'].tolist()
        self.Lung_OAR_PRIORITY_DICT = dict(zip(self.Lung_OAR_name, self.Lung_OAR_priority))
        self.Lung_isDmax = df_lung_oar['isDmax'].tolist()
        self.Lung_isDmax_DICT = dict(zip(self.Lung_OAR_name, self.Lung_isDmax))

    
    def create_dataset(self):
        log_dir = os.path.join(os.path.dirname(self.cfig.get('dataset_save_root', '.')), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        bad_log_path = os.path.join(log_dir, 'bad_npz.txt')
        os.makedirs(self.cfig['dataset_save_root'], exist_ok=True)

        lock = threading.Lock()

        def _process_one(index: int):
            """Process one NPZ into the normalized training dictionary."""
            data_path = self.data_list[index]
            ID = self.data_list[index].split('/')[-1].replace('.npz', '')
            PatientID = ID.split('+')[0]

            if len(str(PatientID)) < 3:
                PatientID = f"{PatientID:0>3}"

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
            In_dict['spacing'] = spacing
            angle_list = In_dict['angle_list']
            In_dict['ed'] = HU2electron_density(In_dict['img']) * In_dict['Body'] 
            In_dict['md'] = HU2mass_density(In_dict['img']) * In_dict['Body'] 
            In_dict['img'] = np.clip(In_dict['img'], self.cfig['down_HU'], self.cfig['up_HU']) / self.cfig['denom_norm_HU'] * In_dict['Body'] 

            if 'dose' in In_dict.keys():
                ptv_highdose =  self.scale_dose_Dict[PatientID]['PTV_High']['PDose']
                In_dict['dose'] = In_dict['dose'] * In_dict['dose_scale']
                PTVHighOPT = self.scale_dose_Dict[PatientID]['PTV_High']['OPTName']
                norm_scale = ptv_highdose / (np.percentile(In_dict['dose'][In_dict[PTVHighOPT].astype('bool')], 3) + 1e-5)
                In_dict['dose'] = In_dict['dose'] * norm_scale / self.cfig['dose_div_factor']
                In_dict['dose'] = np.clip(In_dict['dose'], 0, ptv_highdose * 1.2)

            isocenter = In_dict['isocenter']

            KEYS = list(In_dict.keys())
            for key in list(In_dict.keys()):
                if isinstance(In_dict[key], np.ndarray) and len(In_dict[key].shape) == 3:
                    In_dict[key] = torch.from_numpy(In_dict[key].astype('float'))[None]
                else:
                    if key in KEYS:
                        KEYS.remove(key)

            if self.site_list[index] < 1.5:
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
