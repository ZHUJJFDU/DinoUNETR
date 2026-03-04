
import os
import yaml
import numpy as np
import torch
import pandas as pd
import json
import nibabel as nib
import threading
from tqdm import tqdm
from toolkit import *

# --- Global Configuration ---
NNUNET_RAW = r'nnUNet_raw'
TASK_ID = 502
TASK_NAME = 'DinoUNETR_Raw_2D'
TASK_FOLDER = f"Dataset{TASK_ID}_{TASK_NAME}"

# Directories
OUT_DIR_TR = os.path.join(NNUNET_RAW, TASK_FOLDER, 'imagesTr')
OUT_DIR_LB = os.path.join(NNUNET_RAW, TASK_FOLDER, 'labelsTr')
OUT_DIR_TS = os.path.join(NNUNET_RAW, TASK_FOLDER, 'imagesTs')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- Modified CreateDataset Logic ---
class NnunetProcessor():
    def __init__(self, cfig, phase):
        self.cfig = cfig
        self.phase = phase 
        
        print(f"Loading CSV: {cfig['csv_root']}")
        df = pd.read_csv(cfig['csv_root'])
        
        if phase in ['train', 'validation', 'valid']:
            self.nnunet_split = 'Tr'
            self.df = df[df['phase'].isin(['train', 'validation', 'valid'])]
        elif phase in ['test', 'testing']:
            self.nnunet_split = 'Ts'
            self.df = df[df['phase'].isin(['test', 'testing'])]
        else:
            self.nnunet_split = 'Tr' 
            self.df = df
            
        self.data_list = self.df['npz_path'].tolist()
        self.site_list = self.df['site'].tolist()
        self.cohort_list = self.df['cohort'].tolist()
        
        print(f"Found {len(self.data_list)} cases for phase {phase}")
        
        self.scale_dose_Dict = json.load(open(cfig['scale_dose_dict'], 'r'))
        self.pat_obj_dict = json.load(open(cfig['pat_obj_dict'], 'r'))
        
        df_han = pd.read_csv(cfig['csv_HaN_OAR_priority_root'])
        self.HaN_OAR_name = df_han['OAR_Name'].tolist()
        self.HaN_OAR_PRIORITY = dict(zip(df_han['OAR_Name'], df_han['Priority']))
        self.HaN_isDmax = dict(zip(df_han['OAR_Name'], df_han['isDmax']))
        
        df_lung = pd.read_csv(cfig['csv_LUNG_OAR_priority_root'])
        self.Lung_OAR_name = df_lung['OAR_Name'].tolist()
        self.Lung_OAR_PRIORITY = dict(zip(df_lung['OAR_Name'], df_lung['Priority']))
        self.Lung_isDmax = dict(zip(df_lung['OAR_Name'], df_lung['isDmax']))
        
    def process_one(self, index):
        data_path = self.data_list[index]
        filename = os.path.basename(data_path)
        ID = filename.replace('.npz', '')
        PatientID = ID.split('+')[0]
        if len(str(PatientID)) < 3: PatientID = f"{PatientID:0>3}"
        
        
        # 1. Load Data
        try:
            # print(f"Processing {ID}...")
            data_npz = np.load(data_path, allow_pickle=True)
            arr0 = dict(data_npz).get('arr_0', None)
            In_dict = arr0.item() if isinstance(arr0, np.ndarray) else arr0
        except Exception as e:
            return f"Error loading {ID}: {e}"
            
        if not isinstance(In_dict, dict):
            return f"Error: In_dict is not dict for {ID}"

        # 2. Preprocess logic
        spacing = [2.0, 2.5, 2.5] 
        In_dict['spacing'] = spacing 
        angle_list = In_dict['angle_list']
        isocenter = In_dict['isocenter']

        if 'ed' not in In_dict: 
            In_dict['ed'] = HU2electron_density(In_dict['img']) * In_dict['Body']
        if 'md' not in In_dict:
            In_dict['md'] = HU2mass_density(In_dict['img']) * In_dict['Body']
            
        In_dict['img'] = np.clip(In_dict['img'], self.cfig['down_HU'], self.cfig['up_HU']) / self.cfig['denom_norm_HU'] * In_dict['Body']

        if 'dose' in In_dict:
            try:
                ptv_highdose = self.scale_dose_Dict[PatientID]['PTV_High']['PDose']
                In_dict['dose'] = In_dict['dose'] * In_dict['dose_scale']
                PTVHighOPT = self.scale_dose_Dict[PatientID]['PTV_High']['OPTName']
                if PTVHighOPT in In_dict:
                    norm_scale = ptv_highdose / (np.percentile(In_dict['dose'][In_dict[PTVHighOPT].astype('bool')], 3) + 1e-5)
                    In_dict['dose'] = In_dict['dose'] * norm_scale / self.cfig['dose_div_factor']
                    In_dict['dose'] = np.clip(In_dict['dose'], 0, ptv_highdose * 1.2)
            except Exception as e:
                pass

        KEYS_TO_CONVERT = list(In_dict.keys())
        for k in KEYS_TO_CONVERT:
            if isinstance(In_dict[k], np.ndarray) and len(In_dict[k].shape) == 3:
                In_dict[k] = torch.from_numpy(In_dict[k].astype('float'))[None] 

        if self.site_list[index] < 1.5:
            OAR_LIST = self.HaN_OAR_name
            OAR_PRIORITY = self.HaN_OAR_PRIORITY
            OAR_isDmax = self.HaN_isDmax
        else:
            OAR_LIST = self.Lung_OAR_name
            OAR_PRIORITY = self.Lung_OAR_PRIORITY
            OAR_isDmax = self.Lung_isDmax

        try:
            need_list = self.pat_obj_dict[PatientID]
        except:
            need_list = OAR_LIST

        comb_oar_priority = combine_oar_priority(In_dict, need_list, OAR_PRIORITY) 

        opt_dose_dict = {}
        try:
            for key in self.scale_dose_Dict[PatientID].keys():
                if key in ['PTV_High', 'PTV_Mid', 'PTV_Low']:
                    opt_dose_dict[self.scale_dose_Dict[PatientID][key]['OPTName']] = self.scale_dose_Dict[PatientID][key]['PDose'] / self.cfig['dose_div_factor']
        except:
            pass
        comb_optptv, prs_opt, cat_optptv = combine_ptv(In_dict, opt_dose_dict)

        # Fix: Convert to 3D numpy for distance calculation
        comb_optptv_3d_np = np.squeeze(comb_optptv.cpu().numpy())
        distance_map = calculate_min_distance_to_tumor_surface(comb_optptv_3d_np, spacing) # mm
        
        # calculate_distance_to_tumor expects 3D distance_map and looks up OARs in In_dict
        # In_dict has 4D tensors (1, D, H, W). calculate_distance_to_tumor squeezes them internally.
        comb_oar_distance = calculate_distance_to_tumor(In_dict, distance_map, need_list, OAR_PRIORITY)

        PTV_mask = comb_optptv > 0
        try:
            beam_plate = In_dict['beam_plate'] 
            denom_idx = [int(x) for x in isocenter]
            denom = float(beam_plate[0, denom_idx[0], denom_idx[1], denom_idx[2]])
            if abs(denom) < 1e-6:
                denom = max(float(beam_plate.max()), 1.0)
            beam_plate_norm = (beam_plate / denom) * In_dict['Body']
        except:
            beam_plate = get_allbeam_plate(PTV_mask, isocenter, spacing, angle_list, with_distance=True)
            beam_plate_norm = (beam_plate / max(len(angle_list), 1)) * In_dict['Body']
            
        beam_plate_norm = torch.nan_to_num(beam_plate_norm, nan=0.0, posinf=0.0, neginf=0.0)

        def to_n(t):
            if isinstance(t, torch.Tensor):
                return np.squeeze(t.detach().cpu().numpy())
            return np.squeeze(t)

        try:
            ch_md = to_n(In_dict['md'])
            ch_opt = to_n(comb_optptv)
            ch_prio = to_n(comb_oar_priority)
            ch_beam = to_n(beam_plate_norm)
            ch_dist = to_n(comb_oar_distance)
            ch_body = to_n(In_dict['Body'])
            
            if 'dose' in In_dict:
                ch_dose = to_n(In_dict['dose'] * In_dict['Body'])
            else:
                ch_dose = None

        except Exception as e:
            return f"Error reshaping channels for {ID}: {e}"

        depth = ch_md.shape[0]
        
        if self.nnunet_split == 'Tr':
            img_dir = OUT_DIR_TR
            lbl_dir = OUT_DIR_LB
        else:
            img_dir = OUT_DIR_TS
            lbl_dir = None 

        for d in range(depth):
            safe_id = ID.replace('+', '_') 
            case_identifier = f"{safe_id}_{d:03d}"
            
            affine = np.eye(4)
            def save_nii(arr, out_path):
                nib.save(nib.Nifti1Image(arr[..., None], affine), out_path)

            save_nii(ch_md[d], os.path.join(img_dir, f"{case_identifier}_0000.nii.gz"))
            save_nii(ch_opt[d], os.path.join(img_dir, f"{case_identifier}_0001.nii.gz"))
            save_nii(ch_prio[d], os.path.join(img_dir, f"{case_identifier}_0002.nii.gz"))
            save_nii(ch_beam[d], os.path.join(img_dir, f"{case_identifier}_0003.nii.gz"))
            save_nii(ch_dist[d], os.path.join(img_dir, f"{case_identifier}_0004.nii.gz"))
            save_nii(ch_body[d], os.path.join(img_dir, f"{case_identifier}_0005.nii.gz"))
            
            if lbl_dir and ch_dose is not None:
                save_nii(ch_dose[d], os.path.join(lbl_dir, f"{case_identifier}.nii.gz"))

        return f"Success {ID}"

    def run(self):
        ensure_dir(OUT_DIR_TR)
        ensure_dir(OUT_DIR_LB)
        ensure_dir(OUT_DIR_TS)
        
        total_cases = len(self.data_list)
        print(f"Processing {total_cases} cases for {self.phase} (Split: {self.nnunet_split})")
        
        # DEBUG MODE: Single Threaded
        print("Running in SINGLE THREADED mode for debugging...")
        for i in tqdm(range(total_cases)):
             res = self.process_one(i)
             if "Error" in res:
                 print(res)

def main():
    # Config
    cfig = {
        'csv_root': 'meta_files/meta_data.csv',
        'csv_HaN_OAR_priority_root': 'meta_files/HaN_OAR_update.csv',
        'csv_LUNG_OAR_priority_root': 'meta_files/LUNG_OAR_update.csv',
        'scale_dose_dict': 'meta_files/PTV_DICT.json',
        'pat_obj_dict': 'meta_files/Pat_Obj_DICT.json',
        'down_HU': -1000,
        'up_HU': 1000,
        'denom_norm_HU': 500,
        'dose_div_factor': 10.0
    }
    
    print("Step 1: Processing Train + Validation sets -> imagesTr")
    processor_tr = NnunetProcessor(cfig, phase='train') 
    processor_tr.run()
    
    print("Step 2: Processing Test set -> imagesTs")
    processor_ts = NnunetProcessor(cfig, phase='test') 
    processor_ts.run()
    
    print("Generating dataset.json...")
    # NOTE: Counting nii.gz files might be slow if many thousands.
    # Just generating basic json.
    
    json_dict = {
        "channel_names": {
            "0": "mass_density",
            "1": "comb_optptv",
            "2": "comb_oar_priority",
            "3": "beam_plate_norm",
            "4": "comb_oar_distance",
            "5": "Body",
        },
        "labels": {
            "background": 0,
            "Target": 1 
        },
        "numTraining": 0, # Placeholder, update manually if needed or calculate separately
        "file_ending": ".nii.gz",
        "name": TASK_NAME,
        "reference": "DinoUNETR Raw Conversion",
        "release": "1.0",
        "description": "7 Channels Raw -> nnU-Net. Label is Dose."
    }
    
    with open(os.path.join(NNUNET_RAW, TASK_FOLDER, 'dataset.json'), 'w') as f:
        json.dump(json_dict, f, indent=4)
        
    print("Done!")

if __name__ == '__main__':
    main()
