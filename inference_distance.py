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

# Add safe globals for potentially unsafe pickle loading (if needed by monai)
torch.serialization.add_safe_globals([monai.utils.enums.TraceKeys])

# Ensure current dir is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the Distance Model
from train_lightning_distance import GDPDistanceLightningModel
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
        if 'csv_root' not in cfig:
            raise ValueError("cfig['csv_root'] is missing. Please check your config file.")
            
        df = pd.read_csv(cfig['csv_root'])
        
        # Filtering (match existing logic)
        if 'phase' in df.columns and 'dev_split' in df.columns:
             df = df.loc[(df['phase'] == phase) & (df['dev_split'] == dev_split)]
        
        if 'npz_path' not in df.columns:
             raise ValueError("CSV missing 'npz_path' column.")
             
        self.data_list = df['npz_path'].tolist()
        
        print(f"SimpleDataset Initialized. Found {len(self.data_list)} cases.")

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path = self.data_list[index]
        # Extract ID from filename
        ID = os.path.basename(data_path).replace('.npz', '')
        
        # Load NPZ
        try:
            data_npz = np.load(data_path, allow_pickle=True)
            # data_npz is likely a NpzFile object, need to access the item inside if structured that way
            # Based on inference_simple.py: In_dict = dict(data_npz)['arr_0'].item()
            # This suggests the npz was saved with np.savez(..., arr_0=dict) or similar
            if 'arr_0' in data_npz:
                In_dict = data_npz['arr_0'].item()
            else:
                # Fallback: maybe keys are directly in npz
                In_dict = {k: data_npz[k] for k in data_npz.files}
                
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            # Return dummy
            return {'data': torch.zeros(6, 10, 256, 256), 'id': ID, 'ori_img_size': torch.tensor([10, 256, 256])}

        
        # Keys expected: 'mass_density', 'comb_optptv', 'comb_oar_priority', 'beam_plate_norm', 'comb_oar_distance', 'Body'
        KEYS_TO_LOAD = ['mass_density', 'comb_optptv', 'comb_oar_priority', 'beam_plate_norm', 'comb_oar_distance', 'Body']
        
        for key in KEYS_TO_LOAD:
            if key in In_dict:
                val = In_dict[key]
                if isinstance(val, np.ndarray):
                    # Ensure float type and add channel dim [1, D, H, W]
                    tensor_val = torch.from_numpy(val.astype('float32')).unsqueeze(0)
                    # Handle NaNs/Infs
                    tensor_val[torch.isnan(tensor_val)] = 0
                    tensor_val[torch.isinf(tensor_val)] = 0
                    In_dict[key] = tensor_val
                else:
                    # Already tensor?
                    if isinstance(val, torch.Tensor):
                        if val.dim() == 3: val = val.unsqueeze(0)
                        In_dict[key] = val
            else:
                 # Backup if missing
                 ref_shape = In_dict['Body'].shape if 'Body' in In_dict else (1, 10, 512, 512)
                 if len(ref_shape) == 4: ref_shape = ref_shape[1:] # if (1, D, H, W) -> (D, H, W)
                 In_dict[key] = torch.zeros((1, *ref_shape))
        
        # Concatenate Input Data (6 Channels)
        # Order: MassDensity, PTV, OAR_Priority, Beam, OAR_Dist, Body
        try:
            data_tensor = torch.cat((
                In_dict['mass_density'], 
                In_dict['comb_optptv'],  
                In_dict['comb_oar_priority'],  
                In_dict['beam_plate_norm'],
                In_dict['comb_oar_distance'], 
                In_dict['Body']
            ), dim=0) # [6, D, H, W]
        except Exception as e:
             print(f"Error concatenating tensors for {ID}: {e}")
             # Debug shapes
             for k in KEYS_TO_LOAD:
                 print(f"{k}: {In_dict[k].shape}")
             raise e
        
        # Original Image Size (D, H, W)
        ori_img_size = torch.tensor(data_tensor.shape[1:]) 
        
        return {
            'data': data_tensor,
            'id': ID,
            'ori_img_size': ori_img_size
        }

def inference_distance_simple():
    parser = argparse.ArgumentParser(description='Simple Slice-by-Slice Inference - Distance Model')
    parser.add_argument('--cfig_path', default='config_files/config_infer.yaml', type=str)
    parser.add_argument('--phase', default='valid', type=str, help='Phase to filter data (train/valid/test)')
    parser.add_argument('--dev_split', default='test', type=str, help='Split to filter data (train/valid/test)')
    args = parser.parse_args()

    # 1. Load Config
    if not os.path.exists(args.cfig_path):
        print(f"Config file not found: {args.cfig_path}")
        return

    print(f"Loading config from {args.cfig_path}...")
    cfig = yaml.load(open(args.cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model
    checkpoint_path = None
    if 'save_model_root' in cfig:
        checkpoint_path = cfig['save_model_root']
    elif 'save_model_path' in cfig:
        checkpoint_path = cfig['save_model_path']
    
    if not checkpoint_path:
        print("Error: Config must specify 'save_model_root' or 'save_model_path'.")
        return

    # If directory, find best ckpt
    if os.path.isdir(checkpoint_path):
        potential_ckpts = [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]
        if not potential_ckpts:
             print(f"No .ckpt found in directory {checkpoint_path}.")
             return
        # Try to pick best by val_loss if possible, else first
        best_ckpt = next((x for x in potential_ckpts if 'min_val_loss' in x), potential_ckpts[0])
        checkpoint_path = os.path.join(checkpoint_path, best_ckpt)
    
    print(f"Loading model checkpoint from: {checkpoint_path}")
    
    # Ensure model params 
    if 'model_params' not in cfig:
        cfig['model_params'] = {'input_channels': 6}

    try:
        # Load Distance Model
        pl_module = GDPDistanceLightningModel.load_from_checkpoint(
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
    print(f"Initializing Dataset with phase={args.phase}, dev_split={args.dev_split}")
    dataset = SimpleDataset(cfig['loader_params'], phase=args.phase, dev_split=args.dev_split)
    
    # Batch size 1 because we process full volumes
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 
    
    # 4. Inference
    save_root = cfig.get('save_pred_path', 'results_distance_simple')
    os.makedirs(save_root, exist_ok=True)
    
    print(f"Starting inference... Saving to {save_root}")
    
    # Target slice size
    if 'in_size' not in cfig['loader_params']:
         print("Warning: 'in_size' not in loader_params, defaulting to [96, 256, 256]")
         target_h, target_w = 256, 256
    else:
         target_h, target_w = cfig['loader_params']['in_size'][1], cfig['loader_params']['in_size'][2]
    
    with torch.no_grad():
        for batch in tqdm(loader):
            # Unpack
            vol_data = batch['data'][0] # [6, D, H, W]
            case_id = batch['id'][0]
            ori_size = batch['ori_img_size'][0] # [D, H, W]
            
            orig_d, orig_h, orig_w = ori_size[0].item(), ori_size[1].item(), ori_size[2].item()
            
            # Slice-by-Slice Processing
            pred_slices = []
            
            for d in range(orig_d):
                # 1. Extract Slice: [6, H_orig, W_orig]
                slice_input = vol_data[:, d, :, :]
                
                # 2. Resize to Model Input: [1, 6, 256, 256]
                batch_slice = slice_input.unsqueeze(0).to(device)
                
                batch_slice_resized = F.interpolate(
                    batch_slice, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False, 
                    antialias=True
                )
                
                # 3. Inference
                output, _, _ = model(batch_slice_resized)
                
                # Scale output
                output = output * cfig['scale_out']
                if 'dose_div_factor' in cfig['loader_params']:
                    output = output * cfig['loader_params']['dose_div_factor']
                
                # 4. Resize back to Original: [1, 1, H_orig, W_orig]
                output_orig = F.interpolate(
                    output,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                )
                
                # Append [H, W]
                pred_slices.append(output_orig.cpu().numpy()[0, 0])
            
            # Stack slices: [D, H, W]
            pred_volume = np.stack(pred_slices, axis=0) 
            
            # 5. Save
            save_name = f"{case_id}_pred.npy"
            save_path = os.path.join(save_root, save_name)
            np.save(save_path, pred_volume)

    print("Inference finished.")

if __name__ == "__main__":
    inference_distance_simple()
