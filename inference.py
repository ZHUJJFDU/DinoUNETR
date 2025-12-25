import torch
import data_loader_lightning
import yaml
import argparse 
import os
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
from train_lightning import GDPLightningModel

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Inference for 3D Model (Slice-by-Slice)')
    cfig_path = 'config_files/config_infer.yaml'
    parser.add_argument('--visualize', default='on', type=str, choices=['on','off'])
    parser.add_argument('--strategy', default='lora', type=str, choices=['default', 'lora', 'llrd', 'frozen'], help='Model strategy used during training')
    args = parser.parse_args()

    cfig = yaml.load(open(cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------ data loader (3D) -----------------#
    loaders = data_loader_lightning.GetLoader(cfig = cfig['loader_params'])
    test_loader = loaders.val_dataloader()

    pl_module = GDPLightningModel.load_from_checkpoint(
        cfig['save_model_path'], 
        cfig=cfig, 
        strategy=args.strategy, # 传入 strategy
        strict=True # 保持 strict=True，确保所有权重都正确加载
    )
    model = pl_module.model.to(device)

    save_pred_path = cfig.get('save_pred_path', 'results')
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)
    
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

                vol_ptv_dose = vol_input[0]
                vol_oar_priority = vol_input[1]
                vol_beam_plate_norm = vol_input[5]
                
                C, D, H, W = vol_input.shape
                
                pred_slices = []
                
                # Slice-by-slice inference
                for d in range(D):
                    slice_input = vol_input[:, d, :, :].unsqueeze(0).to(device)

                    output,_,_ = model(slice_input)
                    
                    output = output * cfig['scale_out']
                    
                    pred_slices.append(output.cpu().numpy())
                
                preds_np = np.concatenate(pred_slices, axis=0)
                preds_np = np.transpose(preds_np, (1, 0, 2, 3)) # (C, D, H, W)
                
                labels_np = vol_label.numpy()
                body_np = vol_body.numpy() > 0.5

                preds_np = preds_np  * cfig['loader_params']['dose_div_factor']
                labels_np = labels_np * cfig['loader_params']['dose_div_factor']

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
                
                print(f"[Case: {case_id}] Depth: {D} | MAE: {l1:.6f} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | PSNR: {psnr:.3f} dB")
                
                total_l1.append(l1)
                total_mse.append(mse)
                total_rmse.append(rmse)
                if np.isfinite(psnr):
                    total_psnr.append(psnr)
                
                # --- Visualization & Saving ---
                vis_dir = Path(save_pred_path) / 'visualization' / str(case_id)
                vis_dir.mkdir(parents=True, exist_ok=True)
                
                # Save Individual Metrics
                metrics_path = vis_dir / f"{case_id}_metrics.txt"
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    f.write(f"患者ID: {case_id}\n")
                    f.write(f"注意：指标计算基于body mask掩膜后的剂量分布(Gy)\n\n")
                    f.write(f"MSE: {mse:.6f} Gy²\n")
                    f.write(f"RMSE: {rmse:.6f} Gy\n")
                    f.write(f"MAE: {l1:.6f} Gy\n")
                    f.write(f"PSNR: {psnr:.2f} dB\n\n")
                
                if args.visualize == 'on':
                    # Pre-calculate global vmin/vmax for this volume or use slice-wise dynamic range?
                    # The reference code uses `vmin` and `vmax` variables which seem to be defined outside the loop or per slice.
                    # To be safe and informative, we'll use per-slice dynamic range for vmin/vmax, 
                    # but ensuring 0 is always the floor for dose.
                    
                    for d in range(D):
                        s_pred = preds_np[0, d, :, :]
                        s_true = labels_np[0, d, :, :]

                        slice_ptv = vol_ptv_dose[d, :, :].cpu().numpy()
                        slice_oar = vol_oar_priority[d, :, :].cpu().numpy()
                        slice_beam = vol_beam_plate_norm[d, :, :].cpu().numpy()

                        mask_ptv = (slice_ptv > 0).astype(float) 
                        mask_oar = (slice_oar > 0).astype(float)
                        
                        # 3-column layout
                        # Increase height to accommodate 2 rows
                        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=100)
                        axes = axes.flatten() # Flatten 2D array to 1D for easy indexing
                        
                        # Determine dynamic range for dose maps
                        true_slice_max = np.max(s_true)
                        pred_slice_max = np.max(s_pred)
                        
                        # Use the max of both for common scaling
                        vmax = max(true_slice_max, pred_slice_max)
                        if vmax <= 0: vmax = 1.0
                        vmin = 0.0
                        
                        # --- Plot ground truth dose --- 
                        ax1 = axes[0]
                        ax1.set_title(f"Ground Truth Dose\nMax: {true_slice_max:.2f} Gy", fontsize=12)
                        im1 = ax1.imshow(s_true, cmap='jet', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
                        ax1.axis('off')
                        
                        # --- Plot predicted dose --- 
                        ax2 = axes[1]
                        ax2.set_title(f"Predicted Dose\nMax: {pred_slice_max:.2f} Gy", fontsize=12)
                        im2 = ax2.imshow(s_pred, cmap='jet', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
                        ax2.axis('off')
                        
                        # --- Plot difference map --- 
                        ax3 = axes[2]
                        diff_slice = s_pred - s_true
                        diff_max = max(abs(np.min(diff_slice)), abs(np.max(diff_slice)))
                        if diff_max > 0:
                            diff_vmin, diff_vmax = -diff_max, diff_max
                        else:
                            diff_vmin, diff_vmax = -1, 1
                            
                        ax3.set_title(f"Difference (Pred - GT)\nRange: [{np.min(diff_slice):.2f}, {np.max(diff_slice):.2f}] Gy", fontsize=12)
                        im3 = ax3.imshow(diff_slice, cmap='RdBu_r', vmin=diff_vmin, vmax=diff_vmax, aspect='equal', origin='lower')
                        ax3.axis('off')

                        # --- Plot PTV ---
                        ax4 = axes[3]
                        ax4.set_title("PTV Mask", fontsize=12)
                        # 使用灰色或红色显示 mask
                        im4 = ax4.imshow(mask_ptv, cmap='gray', vmin=0, vmax=1, aspect='equal', origin='lower')
                        ax4.axis('off')

                        # --- Plot OAR ---
                        ax5 = axes[4]
                        ax5.set_title("OAR Structure", fontsize=12)
                        im5 = ax5.imshow(mask_oar, cmap='gray', vmin=0, vmax=1, aspect='equal', origin='lower')
                        ax5.axis('off')

                        # --- Plot Beam Plate ---
                        ax6 = axes[5]
                        ax6.set_title("Beam Plate", fontsize=12)
                        im6 = ax6.imshow(slice_beam, cmap='viridis', aspect='equal', origin='lower') # 射束板适合用 viridis
                        ax6.axis('off')
                        
                        # --- Add colorbars (3-column layout style) ---
                        # Adjust spacing: hspace for vertical gap between rows
                        fig.subplots_adjust(right=0.85, wspace=0.2, hspace=0.3)
                        
                        # Dose colorbar (shared for GT and Pred)
                        cbar_ax1 = fig.add_axes([0.87, 0.55, 0.02, 0.35])
                        cbar1 = fig.colorbar(im1, cax=cbar_ax1)
                        cbar1.set_label('Dose (Gy)', fontsize=10)
                        
                        # Difference colorbar
                        cbar_ax2 = fig.add_axes([0.87, 0.1, 0.02, 0.35])
                        cbar2 = fig.colorbar(im3, cax=cbar_ax2)
                        cbar2.set_label('Difference (Gy)', fontsize=10)
                        
                        # --- Add main title ---
                        fig.suptitle(f"Patient: {case_id} - Slice {d:03d}/{D-1}", fontsize=14, y=0.95)
                        
                        plt.savefig(vis_dir / f"{case_id}_slice_{d:03d}.png", bbox_inches='tight', dpi=100)
                        plt.close(fig)

    # Final Global Summary
    if len(total_l1) > 0:
        mean_l1 = np.mean(total_l1)
        mean_mse = np.mean(total_mse)
        mean_rmse = np.mean(total_rmse)
        mean_psnr = np.mean(total_psnr)
        
        print("\n" + "="*50)
        print(f"[Final Global Summary] Total Cases Processed: {len(total_l1)}")
        print(f"Mean MAE (L1): {mean_l1:.6f}")
        print(f"Mean MSE:      {mean_mse:.6f}")
        print(f"Mean RMSE:     {mean_rmse:.6f}")
        print(f"Mean PSNR:     {mean_psnr:.3f} dB")
        print("="*50)
        
        # Save Global Metrics
        global_metrics_path = Path(save_pred_path) / "global_metrics.txt"
        with open(global_metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"全局统计 (Total Cases: {len(total_l1)})\n")
            f.write(f"注意：指标为所有病例的平均值\n\n")
            f.write(f"Mean MSE: {mean_mse:.6f} Gy²\n")
            f.write(f"Mean RMSE: {mean_rmse:.6f} Gy\n")
            f.write(f"Mean MAE: {mean_l1:.6f} Gy\n")
            f.write(f"Mean PSNR: {mean_psnr:.3f} dB\n")
    else:
        print("[Summary] No cases processed.")