import torch
import torch.nn.functional as F
from tqdm import tqdm
import data_loader_lightning
import yaml
import argparse 
import os
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
from train_lightning import GDPLightningModel
import monai
try:
    torch.serialization.add_safe_globals([monai.utils.enums.TraceKeys])
except AttributeError:
    pass # Older torch versions don't have this


def sliding_window_inference(model, input_tensor, layout_data=None, window_size=256, stride=128, device='cuda'):
    """
    Sliding window inference for 2D slices.
    input_tensor: (1, C, H, W)
    """
    model.eval()
    B, C, H, W = input_tensor.shape
    
    # Grid generation
    h_steps = list(range(0, H - window_size + 1, stride))
    if H > window_size and h_steps[-1] + window_size < H:
        h_steps.append(H - window_size)
        
    w_steps = list(range(0, W - window_size + 1, stride))
    if W > window_size and w_steps[-1] + window_size < W:
        w_steps.append(W - window_size)
        
    # Initialize lazily
    output_tensor = None
    count_tensor = None
    
    for h in h_steps:
        for w in w_steps:
            patch = input_tensor[:, :, h:h+window_size, w:w+window_size]
            
            with torch.no_grad():
                if layout_data is not None:
                    pred_tuple = model(patch, layout_data)
                else:
                    pred_tuple = model(patch)
                
                # Handle tuple output (output, shallow, deep)
                if isinstance(pred_tuple, tuple):
                    pred_patch = pred_tuple[0]
                else:
                    pred_patch = pred_tuple
            
            if output_tensor is None:
                B_out, C_out, _, _ = pred_patch.shape
                output_tensor = torch.zeros((B_out, C_out, H, W), device=device)
                count_tensor = torch.zeros((B_out, C_out, H, W), device=device)
            
            output_tensor[:, :, h:h+window_size, w:w+window_size] += pred_patch
            count_tensor[:, :, h:h+window_size, w:w+window_size] += 1.0
            
    if count_tensor is None: 
         return torch.zeros((B, 1, H, W), device=device)

    return output_tensor / count_tensor

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Inference for 3D Model (Sliding Window)')
    cfig_path = 'config_files/config_infer.yaml'
    parser.add_argument('--visualize', default='on', type=str, choices=['on','off'])
    parser.add_argument('--strategy', default='default', type=str, choices=['default', 'lora', 'llrd', 'frozen'], help='Model strategy used during training')
    parser.add_argument('--window_size', default=256, type=int, help='Sliding window size')
    parser.add_argument('--stride', default=128, type=int, help='Sliding window stride')
    parser.add_argument('--model_path', default='trained_models/trained_model_default_256_meddino_literation_nah&lung/best_model-epoch=148-train_loss=0.1560.ckpt', type=str, help='Override model path')
    
    args = parser.parse_args()

    cfig = yaml.load(open(cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)
    
    # Overwrite strategy from args
    cfig['strategy'] = args.strategy

    if args.model_path:
        cfig['save_model_path'] = args.model_path
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import SimpleDataset
    try:
        from inference_simple import SimpleDataset
    except ImportError:
        print("Error: Could not import SimpleDataset from inference_simple.py")
        exit(1)

    # Load Model
    print(f"Loading model from: {cfig['save_model_path']}")
    pl_module = GDPLightningModel.load_from_checkpoint(
        cfig['save_model_path'], 
        cfig=cfig, 
        strategy=args.strategy, 
        strict=False 
    )
    model = pl_module.model.to(device)
    model.eval()

    # Determine Model Layout
    use_layout = cfig.get('layout', False)

    # Initialize Dataset (Use SimpleDataset to get original volumes)
    # We allow command line args for phase/split if we want, effectively defaulting to config or hardcoded
    # We'll use the arguments or defaults consistent with inference_simple
    phase = 'valid' # Defaulting to valid/test as per inference_simple modification
    dev_split = 'test'
    
    print(f"Initializing SimpleDataset...")
    dataset = SimpleDataset(cfig['loader_params'], phase=phase, dev_split=dev_split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    save_pred_path = cfig.get('save_pred_path', 'results_window')
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)
    
    print(f"Starting sliding window inference on device: {device}")
    print(f"Window Size: {args.window_size}, Stride: {args.stride}")
    print(f"Saving results to: {save_pred_path}")

    # Intermediate Resize Sizes
    INTER_H, INTER_W = 512, 512

    with torch.no_grad():
        for batch in tqdm(loader):
            vol_data = batch['data'][0] # [6, D, H, W]
            case_id = batch['id'][0]
            ori_size = batch['ori_img_size'][0] # [D, H, W]
            
            orig_d, orig_h, orig_w = ori_size[0].item(), ori_size[1].item(), ori_size[2].item()
            
            pred_slices = []
            
            # Slice-by-slice inference
            for d in range(orig_d):
                slice_input = vol_data[:, d, :, :] # [6, H_orig, W_orig]
                
                # 1. Resize to 512x512
                # F.interpolate needs [B, C, H, W]
                batch_slice = slice_input.unsqueeze(0).to(device) # [1, 6, H_orig, W_orig]
                
                batch_slice_512 = F.interpolate(
                    batch_slice, 
                    size=(INTER_H, INTER_W), 
                    mode='bilinear', 
                    align_corners=False, 
                    antialias=True
                )
                
                # Prepare layout data if needed (assuming layout is also resized or not used currently for simplified inference)
                # If layout is strictly needed, we'd need to resize it too. 
                # For now assuming layout_data=None or constructing it if critical.
                # Since SimpleDataset doesn't load layout dicts by default in the same way, we skip layout logic 
                # unless user demands it. Original inference_simple skipped layout.
                layout_data_slice = None 

                # 2. Sliding Window Inference on 512x512
                output_512 = sliding_window_inference(
                    model, 
                    batch_slice_512, 
                    layout_data=layout_data_slice, 
                    window_size=args.window_size, 
                    stride=args.stride, 
                    device=device
                ) # Returns [1, 1, 512, 512]
                
                # Scale output
                output_512 = output_512 * cfig['scale_out']
                if 'dose_div_factor' in cfig['loader_params']:
                    output_512 = output_512 * cfig['loader_params']['dose_div_factor']
                
                # 3. Resize back to Original Size
                output_orig = F.interpolate(
                    output_512,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                )
                
                pred_slices.append(output_orig.cpu().numpy()[0, 0])
            
            # Stack slices: [D, H, W]
            pred_volume = np.stack(pred_slices, axis=0) 
            
            # Save Results (ONLY .npy)
            save_name = f"{case_id}_pred.npy"
            save_path = os.path.join(save_pred_path, save_name)
            np.save(save_path, pred_volume)
            
    print("Inference finished.")
