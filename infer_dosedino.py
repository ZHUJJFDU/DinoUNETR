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

from train_dosedino import DoseDINOLightningModule
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

        if 'csv_root' not in cfig:
            raise ValueError("cfig['csv_root'] is missing. Please check your config file.")

        df = pd.read_csv(cfig['csv_root'])

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
        ID = os.path.basename(data_path).replace('.npz', '')

        try:
            data_npz = np.load(data_path, allow_pickle=True)
            if 'arr_0' in data_npz:
                In_dict = data_npz['arr_0'].item()
            else:
                In_dict = {k: data_npz[k] for k in data_npz.files}

        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            return {'data': torch.zeros(6, 10, 256, 256), 'id': ID, 'ori_img_size': torch.tensor([10, 256, 256])}

        KEYS_TO_LOAD = [
            'mass_density',
            'comb_optptv',
            'comb_oar_priority',
            'Body',
            'beam_plate_norm',
            'comb_oar_distance',
        ]

        for key in KEYS_TO_LOAD:
            if key in In_dict:
                val = In_dict[key]
                if isinstance(val, np.ndarray):
                    tensor_val = torch.from_numpy(val.astype('float32')).unsqueeze(0)
                    tensor_val[torch.isnan(tensor_val)] = 0
                    tensor_val[torch.isinf(tensor_val)] = 0
                    In_dict[key] = tensor_val
                else:
                    if isinstance(val, torch.Tensor):
                        if val.dim() == 3: val = val.unsqueeze(0)
                        In_dict[key] = val
            else:
                 ref_shape = In_dict['Body'].shape if 'Body' in In_dict else (1, 10, 512, 512)
                 if len(ref_shape) == 4: ref_shape = ref_shape[1:]
                 In_dict[key] = torch.zeros((1, *ref_shape))

        try:
            data_tensor = torch.cat((
                In_dict['mass_density'],
                In_dict['comb_optptv'],
                In_dict['comb_oar_priority'],
                In_dict['Body'],
                In_dict['beam_plate_norm'],
                In_dict['comb_oar_distance'],
            ), dim=0)
        except Exception as e:
             print(f"Error concatenating tensors for {ID}: {e}")
             for k in KEYS_TO_LOAD:
                 print(f"{k}: {In_dict[k].shape}")
             raise e

        ori_img_size = torch.tensor(data_tensor.shape[1:])

        return {
            'data': data_tensor,
            'id': ID,
            'ori_img_size': ori_img_size
        }

def infer_dosedino():
    parser = argparse.ArgumentParser(description='Reconstruct 3D dose plans with DoseDINO.')
    parser.add_argument('--cfig_path', default='config_files/config_infer.yaml', type=str)
    parser.add_argument('--phase', default='valid', type=str, help='Phase to filter data (train/valid/test)')
    parser.add_argument('--dev_split', default='test', type=str, help='Split to filter data (train/valid/test)')
    args = parser.parse_args()

    if not os.path.exists(args.cfig_path):
        print(f"Config file not found: {args.cfig_path}")
        return

    print(f"Loading config from {args.cfig_path}...")
    cfig = yaml.load(open(args.cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = None
    if 'save_model_root' in cfig:
        checkpoint_path = cfig['save_model_root']
    elif 'save_model_path' in cfig:
        checkpoint_path = cfig['save_model_path']

    if not checkpoint_path:
        print("Error: Config must specify 'save_model_root' or 'save_model_path'.")
        return

    if os.path.isdir(checkpoint_path):
        potential_ckpts = [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]
        if not potential_ckpts:
             print(f"No .ckpt found in directory {checkpoint_path}.")
             return
        best_ckpt = next((x for x in potential_ckpts if 'min_val_loss' in x), potential_ckpts[0])
        checkpoint_path = os.path.join(checkpoint_path, best_ckpt)

    print(f"Loading model checkpoint from: {checkpoint_path}")

    if 'model_params' not in cfig:
        cfig['model_params'] = {'input_channels': 6}

    try:
        pl_module = DoseDINOLightningModule.load_from_checkpoint(
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

    save_root = cfig.get('save_pred_path', 'results_distance_simple')
    os.makedirs(save_root, exist_ok=True)

    print(f"Starting inference... Saving to {save_root}")

    if 'in_size' not in cfig['loader_params']:
         print("Warning: 'in_size' not in loader_params, defaulting to [96, 256, 256]")
         target_h, target_w = 256, 256
    else:
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
    infer_dosedino()
