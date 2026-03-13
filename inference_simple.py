import torch
import torch.nn.functional as F
import numpy as np
import yaml
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import monai
# Add safe globals for pickle-based loading
torch.serialization.add_safe_globals([monai.utils.enums.TraceKeys])

# Ensure current dir is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from train_lightning import GDPLightningModel
from toolkit import *

def check_list_str(list_in):
    list_out = []
    for i in list_in:
        if isinstance(i, str):
            list_out.append(i)
    return list_out

class SimpleDataset(Dataset):
    """Load full 3D volumes and iterate slice-by-slice."""
    def __init__(self, cfig, phase='test', dev_split='test'):
        self.cfig = cfig
        self.phase = phase
        
        df = pd.read_csv(cfig['csv_root'])
        
        if 'phase' in df.columns and 'dev_split' in df.columns:
             df = df.loc[(df['phase'] == phase) & (df['dev_split'] == dev_split)]
        
        self.data_list = df['npz_path'].tolist()
        self.site_list = df['site'].tolist()
        
        print(f"SimpleDataset Initialized. Found {len(self.data_list)} cases.")

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path = self.data_list[index]
        ID = os.path.basename(data_path).replace('.npz', '')
        
        data_npz = np.load(data_path, allow_pickle=True)
        In_dict = dict(data_npz)['arr_0'].item()
        
        KEYS_TO_LOAD = ['mass_density', 'comb_optptv', 'comb_oar_priority', 'beam_plate_norm', 'comb_oar_distance', 'Body']
        
        for key in KEYS_TO_LOAD:
            if key in In_dict and isinstance(In_dict[key], np.ndarray):
                tensor_val = torch.from_numpy(In_dict[key].astype('float32')).unsqueeze(0)
                tensor_val[torch.isnan(tensor_val)] = 0
                tensor_val[torch.isinf(tensor_val)] = 0
                In_dict[key] = tensor_val
            else:
                 ref_shape = In_dict['Body'].shape if 'Body' in In_dict else (100, 512, 512)
                 In_dict[key] = torch.zeros((1, *ref_shape))
        
        data_tensor = torch.cat((
            In_dict['mass_density'], 
            In_dict['comb_optptv'],  
            In_dict['comb_oar_priority'],  
            In_dict['beam_plate_norm'],
            In_dict['comb_oar_distance'], 
            In_dict['Body']
        ), dim=0)
        
        body_tensor = In_dict['Body'] 
        
        ori_img_size = torch.tensor(data_tensor.shape[1:]) 
        
        return {
            'data': data_tensor,
            'Body': body_tensor,
            'id': ID,
            'ori_img_size': ori_img_size
        }

def inference_simple():
    parser = argparse.ArgumentParser(description='Simple Slice-by-Slice Inference')
    parser.add_argument('--cfig_path', default='config_files/config_infer.yaml', type=str)
    parser.add_argument('--phase', default='valid', type=str, help='Phase to filter data (train/valid/test)')
    parser.add_argument('--dev_split', default='test', type=str, help='Split to filter data (train/valid/test)')
    args = parser.parse_args()

    print(f"Loading config from {args.cfig_path}...")
    cfig = yaml.load(open(args.cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if 'save_model_root' in cfig:
        checkpoint_path = cfig['save_model_root']
    elif 'save_model_path' in cfig:
        checkpoint_path = cfig['save_model_path']
    else:
        print("Error: Config is missing 'save_model_root' or 'save_model_path'.")
        return

    if os.path.isdir(checkpoint_path):
        potential_ckpts = [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]
        if not potential_ckpts:
             print(f"No .ckpt found in {checkpoint_path}. Please check config.")
             return
        best_ckpt = next((x for x in potential_ckpts if 'min_val_loss' in x), potential_ckpts[0])
        checkpoint_path = os.path.join(checkpoint_path, best_ckpt)
    
    print(f"Loading model from: {checkpoint_path}")
    
    try:
        pl_module = GDPLightningModel.load_from_checkpoint(
            checkpoint_path, 
            cfig=cfig, 
            strategy=cfig.get('strategy', 'default'),
            strict=False,
            weights_only=False 
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = pl_module.model.to(device)
    model.eval()
    
    print(f"Initializing Dataset with phase={args.phase}, dev_split={args.dev_split}")
    dataset = SimpleDataset(cfig['loader_params'], phase=args.phase, dev_split=args.dev_split)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 
    
    save_root = cfig.get('save_pred_path', 'results_simple')
    os.makedirs(save_root, exist_ok=True)
    
    print(f"Starting inference... Saving to {save_root}")
    
    target_h, target_w = cfig['loader_params']['in_size'][1], cfig['loader_params']['in_size'][2]
    
    with torch.no_grad():
        for batch in tqdm(loader):
            vol_data = batch['data'][0]
            case_id = batch['id'][0]
            ori_size = batch['ori_img_size'][0]
            
            orig_d, orig_h, orig_w = ori_size[0].item(), ori_size[1].item(), ori_size[2].item()
            
            pred_slices = []
            
            for d in range(orig_d):
                slice_input = vol_data[:, d, :, :]
                
                batch_slice = slice_input.unsqueeze(0).to(device)
                
                batch_slice_resized = F.interpolate(
                    batch_slice, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False, 
                    antialias=True
                )
                
                output, _, _ = model(batch_slice_resized)
                
                output = output * cfig['scale_out']
                if 'dose_div_factor' in cfig['loader_params']:
                    output = output * cfig['loader_params']['dose_div_factor']
                
                output_orig = F.interpolate(
                    output,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                )
                
                pred_slices.append(output_orig.cpu().numpy()[0, 0])
            
            pred_volume = np.stack(pred_slices, axis=0)
            
            save_name = f"{case_id}_pred.npy"
            save_path = os.path.join(save_root, save_name)
            np.save(save_path, pred_volume)

    print("Inference finished.")

if __name__ == "__main__":
    inference_simple()
