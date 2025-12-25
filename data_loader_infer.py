

'''
This script is adapted from the data loader of below CVPR paper. 
If you find the functions in this script are helpful to you (for the challenge and beyond), please kindly cite the original paper: 

Riqiang Gao, Bin Lou, Zhoubing Xu, Dorin Comaniciu, and Ali Kamen. 
"Flexible-cm gan: Towards precise 3d dose prediction in radiotherapy." 
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.
'''

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
import json
import pdb
import time
from scipy import ndimage 
from toolkit import *


HaN_OAR_LIST = [ 'Cochlea_L', 'Cochlea_R','Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'Chiasim', 'LacrimalGlands', 'BrachialPlexus', 'Brain',  'BrainStem_03',  'Esophagus', 'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV', 'OCavity-PTV', 'ParotidCon-PTV', 'Parotidlps-PTV', 'Parotids-PTV', 'PharConst-PTV', 'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05']

HaN_OAR_DICT = {HaN_OAR_LIST[i]: (i+1) for i in range(len(HaN_OAR_LIST))}

Lung_OAR_LIST = ["PTV_Ring.3-2", "Total Lung-GTV", "SpinalCord",  "Heart",  "LAD", "Esophagus",  "BrachialPlexus",  "GreatVessels", "Trachea", "Body_Ring0-3"]

Lung_OAR_DICT = {Lung_OAR_LIST[i]: (i+10) for i in range(len(Lung_OAR_LIST))}



class MyDataset(Dataset):
    
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
        
        # import pdb
        # pdb.set_trace()


    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):

        data_path = self.data_list[index]
        ID = self.data_list[index].split('/')[-1].replace('.npz', '')
        PatientID = ID.split('+')[0]

        if len(str(PatientID)) < 3:
            PatientID = f"{PatientID:0>3}"

        #preprocess data
        data_npz = np.load(data_path, allow_pickle=True)

        In_dict = dict(data_npz)['arr_0'].item()

        spacing = [2.0,2.5,2.5]
        In_dict['spacing'] = spacing # To be validated
        angle_list = In_dict['angle_list']
        In_dict['ed'] = HU2electron_density(In_dict['img']) * In_dict['Body'] 
        In_dict['md'] = HU2mass_density(In_dict['img']) * In_dict['Body'] 
        In_dict['img'] = np.clip(In_dict['img'], self.cfig['down_HU'], self.cfig['up_HU']) / self.cfig['denom_norm_HU'] 


        
        if 'dose' in In_dict.keys():
            ptv_highdose =  self.scale_dose_Dict[PatientID]['PTV_High']['PDose']
            In_dict['dose'] = In_dict['dose'] * In_dict['dose_scale'] 
            PTVHighOPT = self.scale_dose_Dict[PatientID]['PTV_High']['OPTName']
            norm_scale = ptv_highdose / (np.percentile(In_dict['dose'][In_dict[PTVHighOPT].astype('bool')], 3) + 1e-5) # D97
            In_dict['dose'] = In_dict['dose'] * norm_scale / self.cfig['dose_div_factor']
            In_dict['dose'] = np.clip(In_dict['dose'], 0, ptv_highdose * 1.2)


        isocenter = In_dict['isocenter']

        

        KEYS = list(In_dict.keys())
        for key in In_dict.keys(): 
            if isinstance(In_dict[key], np.ndarray) and len(In_dict[key].shape) == 3:
                In_dict[key] = torch.from_numpy(In_dict[key].astype('float'))[None]
            else:
                KEYS.remove(key)

        if self.site_list[index] < 1.5: #To be validated
            OAR_LIST = self.HaN_OAR_name
            OAR_PRIORITY = self.HaN_OAR_PRIORITY_DICT
            
        else:
            OAR_LIST = self.Lung_OAR_name
            OAR_PRIORITY = self.Lung_OAR_PRIORITY_DICT


        try:
            need_list = self.pat_obj_dict[ID.split('+')[0]] # list(a.values())[0]
        except:
            need_list = OAR_LIST
            print (ID.split('+')[0],  '-------------not in the pat_obj_dict')
        #comb_oar, cat_oar  = combine_oar(In_dict, need_list, self.cfig['norm_oar'], OAR_DICT)
        comb_oar_priority  = combine_oar_priority(In_dict, need_list, OAR_PRIORITY)

        

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

        #distance_map = calculate_min_distance_to_tumor_surface(save_data_dict['comb_optptv'], spacing) ##mm
        distance_map = calculate_min_distance_to_tumor_surface(save_data_dict['comb_optptv'], In_dict['spacing']) ##mm

        # plt.imshow(distance_map[round(isocenter[0]),:,:])
        # plt.colorbar()
        # plt.show()
        save_data_dict['comb_oar_distance'] = calculate_distance_to_tumor(In_dict,distance_map,need_list, OAR_PRIORITY)

        # plt.imshow(save_data_dict['comb_oar_distance'][round(isocenter[0]),:,:])
        # plt.colorbar()
        # plt.show()

        save_data_dict['Body'] = np.squeeze(In_dict['Body'].to('cpu').numpy())

        
        save_data_dict['electron_density'] = np.squeeze(In_dict['ed'].to('cpu').numpy())
        save_data_dict['mass_density'] = np.squeeze(In_dict['md'].to('cpu').numpy())
        #save_data_dict['beam_plate'] = np.squeeze(beam_plate.to('cpu').numpy())

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
            beam_plate_norm = beam_plate / round(beam_plate[int(isocenter[0]),int(isocenter[1]),int(isocenter[2])]) * save_data_dict['Body']
        except:
            beam_plate = get_allbeam_plate(save_data_dict['PTV'], isocenter, spacing, angle_list, with_distance = True)
            beam_plate_norm = beam_plate/len(angle_list) * save_data_dict['Body']


        save_data_dict['beam_plate'] = beam_plate
        save_data_dict['beam_plate_norm'] = beam_plate_norm

        # load data
        In_dict = save_data_dict

        isocenter = In_dict['isocenter']
        ori_img_size = In_dict['Body'].shape
        
        KEYS = list(In_dict.keys())
        for key in In_dict.keys(): 
            if isinstance(In_dict[key], np.ndarray) and len(In_dict[key].shape) == 3:
                In_dict[key] = torch.from_numpy(In_dict[key].astype('float'))[None]
            else:
                KEYS.remove(key)
        
        if self.phase == 'train':
            if 'with_aug' in self.cfig.keys() and not self.cfig['with_aug']:
                self.aug = tt_augmentation(KEYS, self.cfig['in_size'],  self.cfig['out_size'], isocenter)
            else:
                self.aug = tr_augmentation(KEYS, self.cfig['in_size'], self.cfig['out_size'], isocenter)
            
        if self.phase in ['val', 'test', 'valid', 'external_test'] or self.dev_split in ['test','valid']:
            self.aug = tt_augmentation(KEYS, self.cfig['in_size'], self.cfig['out_size'], isocenter)


        In_dict = self.aug(In_dict)

        data_dict = dict()  


        if 'label' in In_dict.keys():
            data_dict['label'] = In_dict['label']
            ref_dose = In_dict['label'] * 1
            data_dict['ref_5Gy_mask'] = (ref_dose > 5) & (In_dict['Body'] > 0)

        In_dict['Body'] = (In_dict['Body'] > 0.5).type(torch.FloatTensor)
        In_dict['PTV_expanded'] = (In_dict['PTV_expanded'] > 0.5).type(torch.FloatTensor)

        # if self.cfig['CatStructures']:
        #     data_dict['data'] = torch.cat((cat_optptv, cat_ptv, cat_oar, In_dict['Body'], In_dict['img'], data_dict['beam_plate'], data_dict['angle_plate'], prompt_extend), axis=0)
        # else:
        data_dict['data'] = torch.cat((In_dict['comb_optptv'],  In_dict['comb_oar_priority'],  In_dict['comb_oar_distance'], In_dict['Body'], In_dict['mass_density'], In_dict['beam_plate_norm']), axis=0) # , In_dict['obj_2DGy'], In_dict['obj_2DWei']

        data_dict['Body'] = In_dict['Body']

        data_dict['PTV_expanded'] = In_dict['PTV_expanded'] * In_dict['Body']
        
        data_dict['ori_isocenter'] = torch.tensor(isocenter)
        data_dict['ori_img_size'] = torch.tensor(ori_img_size)
        data_dict['id'] = ID
        del In_dict

        return data_dict
    
class GetLoader(object):
    def __init__(self, cfig):
        super().__init__()
        self.cfig = cfig
        
    def train_dataloader(self):
        dataset = MyDataset(self.cfig,  phase='train') 
        return DataLoader(dataset, batch_size=self.cfig['train_bs'],  shuffle=True, num_workers=self.cfig['num_workers'])

    def val_dataloader(self):
        dataset = MyDataset(self.cfig, phase='valid') 
        return DataLoader(dataset, batch_size=self.cfig['val_bs'], shuffle=False, num_workers=self.cfig['num_workers'])
    
    def test_dataloader(self):
        dataset = MyDataset(self.cfig, phase='test') 
        return DataLoader(dataset, batch_size=self.cfig['val_bs'], shuffle=False, num_workers=self.cfig['num_workers'])


