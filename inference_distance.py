import torch
import data_loader_lightning
import yaml
import argparse 
import os
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import torch.nn.functional as F

# Ensure current dir is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the new Lightning Module
from train_lightning_distance import GDPDistanceLightningModel

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Inference for 3D Model (Slice-by-Slice) - Distance Model')
    # Default to config_infer.yaml but user can override. 
    # Note: user might need to point to a config that has the correct checkpoint path.
    parser.add_argument('--cfig_path', default='config_files/config_infer.yaml', type=str)
    parser.add_argument('--visualize', default='on', type=str, choices=['on','off'])
    args = parser.parse_args()

    cfig = yaml.load(open(args.cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------ data loader (3D) -----------------#
    loaders = data_loader_lightning.GetLoader(cfig = cfig['loader_params'])
    test_loader = loaders.val_dataloader()

    # Determine Model Layout / Strategy
    # For distance model, we usually use 'default' strategy for inference unless specified otherwise
    strategy = cfig.get('strategy', 'default')
    
    # Path to the checkpoint you want to infer on
    checkpoint_path = cfig['save_model_path']
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load from compatible checkpoint
    # We must ensure cfig has 'model_params' populated or passed correctly
    if 'model_params' not in cfig:
        # Fallback or error? Usually config_infer has limited params.
        # We might need to inject model params if they are missing, specifically input_channels=6
        cfig['model_params'] = {'input_channels': 6}

    try:
        pl_module = GDPDistanceLightningModel.load_from_checkpoint(
            checkpoint_path, 
            cfig=cfig, 
            strategy=strategy,
            strict=True 
        )
    except Exception as e:
        print(f"Failed to load with strict=True: {e}")
        print("Retrying with strict=False...")
        pl_module = GDPDistanceLightningModel.load_from_checkpoint(
            checkpoint_path, 
            cfig=cfig, 
            strategy=strategy,
            strict=False 
        )

    model = pl_module.model.to(device)

    save_pred_path = cfig.get('save_pred_path', 'results_distance')
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)
    
    # Initialize Global Metrics Log (Overwrite old file)
    global_metrics_path = Path(save_pred_path) / "global_metrics.txt"
    with open(global_metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"Inference Log (Distance Model) - Started at {datetime.now()}\n")
        f.write("="*80 + "\n")
    
    print(f"Starting inference on device: {device}")

    # Global metrics accumulators
    total_l1 = []
    total_mse = []
    total_rmse = []
    total_psnr = []

    with torch.no_grad():
        model.eval()
        
        for batch_idx, batch in enumerate(test_loader):
            inputs_3d = batch['data']
            labels_3d = batch['label']
            body_mask_3d = batch['Body']
            case_ids = batch['id']
            
            for b in range(len(case_ids)):
                case_id = case_ids[b]
                
                # Extract volumes
                vol_input = inputs_3d[b]      # (C, D, H, W)
                vol_label = labels_3d[b]      # (C, D, H, W)
                vol_body = body_mask_3d[b]    # (1, D, H, W)

                # Channels:
                # [0:MassDensity, 1:PTV, 2:OAR_Priority, 3:Beam, 4:OAR_Dist, 5:Body]
                vol_ptv_dose = vol_input[1]
                vol_oar_priority = vol_input[2]
                vol_beam_plate_norm = vol_input[3]
                vol_distance = vol_input[4]
                
                C, D, H, W = vol_input.shape
                
                # Batch Inference Implementation
                BATCH_SIZE = 16
                vol_input_permuted = vol_input.permute(1, 0, 2, 3) # (D, C, H, W)
                pred_slices = []
                
                for i in range(0, D, BATCH_SIZE):
                    batch_input = vol_input_permuted[i:i+BATCH_SIZE].to(device).float()
                    
                    # Forward pass
                    # model(batch) -> output, f6, f12 (we only need output)
                    output, _, _ = model(batch_input)
                    
                    output = output * cfig['scale_out']
                    pred_slices.append(output.cpu().numpy())
                
                preds_np = np.concatenate(pred_slices, axis=0)
                preds_np = np.transpose(preds_np, (1, 0, 2, 3)) # (C, D, H, W)
                
                labels_np = vol_label.numpy()
                body_np = vol_body.numpy() > 0.5

                if 'dose_div_factor' in cfig['loader_params']:
                    preds_np = preds_np  * cfig['loader_params']['dose_div_factor']
                    labels_np = labels_np * cfig['loader_params']['dose_div_factor']

                # Apply Body Mask to Prediction
                preds_np = preds_np * body_np.astype(preds_np.dtype)

                # --- RESIZE TO ORIGINAL SIZE FOR EVALUATION ---
                if 'ori_img_size' in batch:
                    # ori_img_size is (Batch, 3) -> (D, H, W)
                    # We are inside the loop over case_ids, so we take the b-th element
                    target_shape = batch['ori_img_size'][b].cpu().numpy().astype(int) # [D, H, W]
                    
                    # Current shape: (C, D, H, W)
                    curr_d, curr_h, curr_w = preds_np.shape[1], preds_np.shape[2], preds_np.shape[3]
                    
                    if (curr_h != target_shape[1]) or (curr_w != target_shape[2]) or (curr_d != target_shape[0]):
                        # print(f"Resizing from {(curr_d, curr_h, curr_w)} to {target_shape}")
                        
                        # Convert to Torch for interpolation
                        # Input to interpolate: (Batch, Channel, D, H, W)
                        preds_t = torch.from_numpy(preds_np).float().unsqueeze(0)
                        labels_t = torch.from_numpy(labels_np).float().unsqueeze(0)
                        body_t = torch.from_numpy(body_np).float().unsqueeze(0)

                        # Resize
                        # Trilinear for continuous data (pred, label)
                        preds_t = torch.nn.functional.interpolate(preds_t, size=tuple(target_shape), mode='trilinear', align_corners=False)
                        labels_t = torch.nn.functional.interpolate(labels_t, size=tuple(target_shape), mode='trilinear', align_corners=False)
                        # Nearest for masks (body)
                        body_t = torch.nn.functional.interpolate(body_t, size=tuple(target_shape), mode='nearest')
                        
                        # Back to Numpy
                        preds_np = preds_t.squeeze(0).numpy()
                        labels_np = labels_t.squeeze(0).numpy()
                        body_np = body_t.squeeze(0).numpy() > 0.5
                        
                        # Re-mask after resize to ensure cleanliness
                        preds_np = preds_np * body_np.astype(preds_np.dtype)
                        labels_np = labels_np * body_np.astype(labels_np.dtype)

                # --- Compute Metrics ---
                if np.any(body_np):
                    diff = (preds_np - labels_np)[body_np]
                else:
                    diff = (preds_np - labels_np).flatten()
                
                l1 = float(np.mean(np.abs(diff)))
                mse = float(np.mean(diff ** 2))
                rmse = math.sqrt(mse)
                
                peak = float(np.max(labels_np)) 
                if peak <= 0: peak = float(np.max(preds_np))
                if peak <= 0: peak = 1.0
                
                psnr = float('inf') if rmse == 0 else 20.0 * math.log10(peak / (rmse + 1e-12))
                  
                print(f"[Case: {case_id}] Depth: {D} -> {preds_np.shape[1]} | MAE: {l1:.6f} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | PSNR: {psnr:.3f} dB")
                
                # Save visualization if enabled
                if args.visualize == 'on':
                    # Define and create visualization directory
                    vis_dir = Path(save_pred_path) / 'visualization' / str(case_id)
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Update global metrics file
                    with open(global_metrics_path, 'a', encoding='utf-8') as f:
                        f.write(f"[Case: {case_id}] Depth: {D} | MAE: {l1:.6f} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | PSNR: {psnr:.3f} dB\n")

                    total_l1.append(l1)
                    total_mse.append(mse)
                    total_rmse.append(rmse)
                    if np.isfinite(psnr):
                        total_psnr.append(psnr)
                
                    # Visualize slices
                    for d in range(D):
                        s_pred = preds_np[0, d, :, :]
                        s_true = labels_np[0, d, :, :]

                        slice_ptv = vol_ptv_dose[d, :, :].cpu().numpy()
                        slice_oar = vol_oar_priority[d, :, :].cpu().numpy()
                        slice_beam = vol_beam_plate_norm[d, :, :].cpu().numpy()
                        slice_dist = vol_distance[d, :, :].cpu().numpy()

                        mask_ptv = (slice_ptv > 0).astype(float) 
                        mask_oar = (slice_oar > 0).astype(float)
                        
                        # Layout: 2 rows, 3 columns
                        # Row 1: GT Dose | Pred Dose | Difference
                        # Row 2: Distance Map | OAR | PTV (or Beam)
                        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=100)
                        axes = axes.flatten()
                        
                        vmax = max(np.max(s_true), np.max(s_pred))
                        if vmax <= 0: vmax = 1.0
                        
                        # --- 1. GT Dose ---
                        ax = axes[0]
                        ax.set_title(f"GT Dose (Max: {np.max(s_true):.2f})", fontsize=10)
                        im = ax.imshow(s_true, cmap='jet', vmin=0, vmax=vmax, origin='lower')
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        ax.axis('off')

                        # --- 2. Pred Dose ---
                        ax = axes[1]
                        ax.set_title(f"Pred Dose (Max: {np.max(s_pred):.2f})", fontsize=10)
                        im = ax.imshow(s_pred, cmap='jet', vmin=0, vmax=vmax, origin='lower')
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        ax.axis('off')

                        # --- 3. Difference ---
                        ax = axes[2]
                        diff_slice = s_pred - s_true
                        dmax = max(abs(diff_slice.min()), abs(diff_slice.max()))
                        if dmax == 0: dmax = 1e-6
                        ax.set_title(f"Diff [{diff_slice.min():.2f}, {diff_slice.max():.2f}]", fontsize=10)
                        im = ax.imshow(diff_slice, cmap='RdBu_r', vmin=-dmax, vmax=dmax, origin='lower')
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        ax.axis('off')

                        # --- 4. Distance Map (New!) ---
                        ax = axes[3]
                        ax.set_title("Distance Map Input", fontsize=10)
                        # Distance map usually has value -1 to 1 or 0 to 1 depending on norm
                        im = ax.imshow(slice_dist, cmap='viridis', origin='lower')
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        ax.axis('off')

                        # --- 5. OAR ---
                        ax = axes[4]
                        ax.set_title("OAR", fontsize=10)
                        ax.imshow(mask_oar, cmap='gray', origin='lower')
                        ax.axis('off')

                        # --- 6. PTV ---
                        ax = axes[5]
                        ax.set_title("PTV", fontsize=10)
                        ax.imshow(mask_ptv, cmap='gray', origin='lower')
                        ax.axis('off')

                        fig.suptitle(f"Patient: {case_id} - Slice {d:03d}", fontsize=14)
                        plt.savefig(vis_dir / f"{case_id}_slice_{d:03d}.png", bbox_inches='tight')
                        plt.close(fig)

    # Final Summary
    if len(total_l1) > 0:
        mean_l1 = np.mean(total_l1)
        mean_mse = np.mean(total_mse)
        mean_rmse = np.mean(total_rmse)
        mean_psnr = np.mean(total_psnr)
        
        print("\n" + "="*50)
        print(f"[Final Summary] Total Cases: {len(total_l1)}")
        print(f"Mean MAE:  {mean_l1:.6f}")
        print(f"Mean MSE:  {mean_mse:.6f}")
        print(f"Mean RMSE: {mean_rmse:.6f}")
        print(f"Mean PSNR: {mean_psnr:.3f} dB")
        print("="*50)
        
        with open(global_metrics_path, 'a', encoding='utf-8') as f:
            f.write(f"\nFinal Summary:\nMean MAE: {mean_l1:.6f}\nMean RMSE: {mean_rmse:.6f}\nMean PSNR: {mean_psnr:.3f}\n")
    else:
        print("No cases processed.")
