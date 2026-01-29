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
    """
    Simplified Dataset that loads data WITHOUT resizing the depth dimension.
    It returns the original 3D volume (C, D, H, W) so we can iterate slice-by-slice.
    """
    def __init__(self, cfig, phase='test', dev_split='test'):
        self.cfig = cfig
        self.phase = phase
        
        # Load CSV
        df = pd.read_csv(cfig['csv_root'])
        
        # Filtering (match existing logic)
        # Assuming we just want to run inference on the specified set
        if 'phase' in df.columns and 'dev_split' in df.columns:
             df = df.loc[(df['phase'] == phase) & (df['dev_split'] == dev_split)]
        
        self.data_list = df['npz_path'].tolist()
        self.site_list = df['site'].tolist()
        
        print(f"SimpleDataset Initialized. Found {len(self.data_list)} cases.")

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path = self.data_list[index]
        # Extract ID from filename
        ID = os.path.basename(data_path).replace('.npz', '')
        
        # Load NPZ
        data_npz = np.load(data_path, allow_pickle=True)
        In_dict = dict(data_npz)['arr_0'].item()
        
        # Original keys usually include: 
        # 'mass_density', 'comb_optptv', 'comb_oar_priority', 'beam_plate_norm', 'comb_oar_distance', 'Body'
        # Convert necessary arrays to Tensor [1, D, H, W]
        
        KEYS_TO_LOAD = ['mass_density', 'comb_optptv', 'comb_oar_priority', 'beam_plate_norm', 'comb_oar_distance', 'Body']
        
        for key in KEYS_TO_LOAD:
            if key in In_dict and isinstance(In_dict[key], np.ndarray):
                # Ensure float type and add channel dim [1, D, H, W]
                tensor_val = torch.from_numpy(In_dict[key].astype('float32')).unsqueeze(0)
                # Handle NaNs/Infs
                tensor_val[torch.isnan(tensor_val)] = 0
                tensor_val[torch.isinf(tensor_val)] = 0
                In_dict[key] = tensor_val
            else:
                 # Backup if missing (should not happen for valid data)
                 # We need the shape of something else to create zeros
                 ref_shape = In_dict['Body'].shape if 'Body' in In_dict else (100, 512, 512)
                 In_dict[key] = torch.zeros((1, *ref_shape))
        
        # Concatenate Input Data (6 Channels)
        # Order matches data_loader_lightning.py: 
        # MassDensity, PTV, OAR_Priority, Beam, OAR_Dist, Body
        data_tensor = torch.cat((
            In_dict['mass_density'], 
            In_dict['comb_optptv'],  
            In_dict['comb_oar_priority'],  
            In_dict['beam_plate_norm'],
            In_dict['comb_oar_distance'], 
            In_dict['Body']
        ), dim=0) # [6, D, H, W]
        
        # Body Mask (for post-processing if needed, though mostly used for metric calc)
        body_tensor = In_dict['Body'] 
        
        # Original Image Size (D, H, W)
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
    # Removed --save_folder to match original logic (use yaml)
    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading config from {args.cfig_path}...")
    cfig = yaml.load(open(args.cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model
    if 'save_model_root' in cfig:
        checkpoint_path = cfig['save_model_root']
    elif 'save_model_path' in cfig:
        checkpoint_path = cfig['save_model_path']
    else:
        print("Error: Config is missing 'save_model_root' or 'save_model_path'.")
        return

    # If save_model_root is a directory, find the best checkpoint or last checkpoint
    # For simplicity, let's assume user passes a .ckpt path in 'save_model_path' often used in inference configs
    # If not, try to find one in the root.
    if os.path.isdir(checkpoint_path):
        potential_ckpts = [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]
        if not potential_ckpts:
             print(f"No .ckpt found in {checkpoint_path}. Please check config.")
             return
        # Pick the one with 'min_val_loss' or just the first one
        best_ckpt = next((x for x in potential_ckpts if 'min_val_loss' in x), potential_ckpts[0])
        checkpoint_path = os.path.join(checkpoint_path, best_ckpt)
    
    print(f"Loading model from: {checkpoint_path}")
    
    try:
        pl_module = GDPLightningModel.load_from_checkpoint(
            checkpoint_path, 
            cfig=cfig, 
            strategy=cfig.get('strategy', 'default'),
            strict=False 
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = pl_module.model.to(device)
    model.eval()
    
    # 3. Prepare Data
    # Assuming we run on 'valid'/'test' split to match inference.py val_dataloader
    print(f"Initializing Dataset with phase={args.phase}, dev_split={args.dev_split}")
    dataset = SimpleDataset(cfig['loader_params'], phase=args.phase, dev_split=args.dev_split)
    
    # We use batch_size=1 because each 'item' is a full 3D volume of variable depth
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 
    
    
    # 4. Inference Loop
    # Use save_pred_path from yaml, default to 'results_simple' if not found
    save_root = cfig.get('save_pred_path', 'results_simple')
    os.makedirs(save_root, exist_ok=True)
    
    print(f"Starting inference... Saving to {save_root}")
    
    # Target slice size from config (e.g. [256, 256])
    # Config format: in_size: [96, 256, 256] -> we want the last two dims
    target_h, target_w = cfig['loader_params']['in_size'][1], cfig['loader_params']['in_size'][2]
    
    with torch.no_grad():
        for batch in tqdm(loader):
            # Unpack
            vol_data = batch['data'][0] # [6, D, H, W] (Unsqueeze batch dim)
            case_id = batch['id'][0]
            ori_size = batch['ori_img_size'][0] # [D, H, W]
            
            orig_d, orig_h, orig_w = ori_size[0].item(), ori_size[1].item(), ori_size[2].item()
            
            # Prepare container for predictions [D, H, W]
            # We collect slice by slice
            pred_slices = []
            
            # Slice-by-Slice Processing
            # Iterate over Depth
            for d in range(orig_d):
                # 1. Extract Slice: [6, H_orig, W_orig]
                slice_input = vol_data[:, d, :, :]
                
                # 2. Resize to Model Input Size: [1, 6, 256, 256]
                # interpolate expects [Batch, Channel, H, W]
                batch_slice = slice_input.unsqueeze(0).to(device) # [1, 6, H_orig, W_orig]
                
                batch_slice_resized = F.interpolate(
                    batch_slice, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False, 
                    antialias=True
                )
                
                # 3. Inference
                # Model output: [1, 1, 256, 256] (Dose)
                output, _, _ = model(batch_slice_resized)
                
                # Scale output
                output = output * cfig['scale_out']
                if 'dose_div_factor' in cfig['loader_params']:
                    output = output * cfig['loader_params']['dose_div_factor']
                
                # 4. Resize back to Original Slice Size: [1, 1, H_orig, W_orig]
                output_orig = F.interpolate(
                    output,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                )
                
                # Append to list (remove batch/channel dim -> [H_orig, W_orig])
                pred_slices.append(output_orig.cpu().numpy()[0, 0])
            
            # Stack slices: [D, H, W]
            pred_volume = np.stack(pred_slices, axis=0) # [D, H, W]
            
            # 5. Save Results
            save_name = f"{case_id}_pred.npy"
            save_path = os.path.join(save_root, save_name)
            np.save(save_path, pred_volume)
            
            # print(f"Saved {save_name} shape={pred_volume.shape}")

    print("Inference finished.")

if __name__ == "__main__":
    inference_simple()
