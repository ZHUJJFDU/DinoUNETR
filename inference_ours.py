import torch
import data_loader_ours
import yaml
import argparse 
import os
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
from train_lightning import GDPLightningModel

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Inference for 3D Model (Slice-by-Slice) on New Dataset')
    parser.add_argument('cfig_path',  type = str, default = 'config_files/config_infer.yaml')
    parser.add_argument('--visualize', default='on', type=str, choices=['on','off'])
    args = parser.parse_args()

    cfig = yaml.load(open(args.cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------ Data Loader (New Dataset) -----------------#
    # Use path from config or fallback to the one in data_loader_ours.py
    root_dir = cfig.get('loader_params', {}).get('data_root', r"C:\Users\960\Desktop\aiendtoend\dataset_normalized")
    
    print(f"Loading data from: {root_dir}")
    dm = data_loader_ours.DataModule(
        data_root=root_dir,
        batch_size=1,
        num_workers=0 
    )
    dm.setup(stage="fit")
    
    # Access the underlying 3D CacheDataset to iterate over volumes
    # dm.val_dataset is SliceDataset, dm.val_dataset.dataset is the 3D CacheDataset
    val_ds_3d = dm.val_dataset.dataset
    
    # ------------ Model -----------------#
    pl_module = GDPLightningModel.load_from_checkpoint(cfig['save_model_path'], cfig = cfig, strict = True)
    model = pl_module.model.to(device)
    model.eval()

    save_pred_path = cfig.get('save_pred_path', 'results_ours')
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)
    
    print(f"Starting inference on device: {device}")
    print(f"Total volumes to process: {len(val_ds_3d)}")

    # Global metrics accumulators
    total_l1 = []
    total_mse = []
    total_rmse = []
    total_psnr = []

    with torch.no_grad():
        for i in range(len(val_ds_3d)):
            item = val_ds_3d[i]
            
            # Extract data
            # Keys in data_loader_ours: 'ct', 'ptv', 'organ', 'body_mask', 'label', 'data', 'patient_id'
            # Shapes (from transforms): (C, H, W, D) due to RAS orientation and Resized(..., -1)
            
            vol_input = item['data']         # (C, H, W, D) - Input to model (6 channels)
            vol_label = item['label']        # (C, H, W, D) - GT Dose
            vol_body = item['body_mask']     # (1, H, W, D) - Body mask
            case_id = item['patient_id']
            
            # Ensure input is on device
            if isinstance(vol_input, torch.Tensor):
                vol_input = vol_input.to(device)
            
            C, H, W, D = vol_input.shape
            
            pred_slices = []
            
            # Slice-by-slice inference
            for z in range(D):
                # Slice along the last dimension (Depth)
                # Input to model must be (B, C, H, W)
                slice_input = vol_input[..., z].unsqueeze(0) 
                
                output = model(slice_input)
                
                if 'scale_out' in cfig:
                    output = output * cfig['scale_out']
                
                pred_slices.append(output.cpu().numpy())
            
            # Reconstruct 3D volume
            # Concatenate along batch dim (0) -> (D, 1, H, W)
            preds_stacked = np.concatenate(pred_slices, axis=0)
            # Transpose to match label shape (1, H, W, D) -> (C, H, W, D)
            preds_3d = np.transpose(preds_stacked, (1, 2, 3, 0))
            
            # Convert label and body to numpy for metrics
            if isinstance(vol_label, torch.Tensor):
                labels_np = vol_label.cpu().numpy()
            else:
                labels_np = np.array(vol_label)
                
            if isinstance(vol_body, torch.Tensor):
                body_np = vol_body.cpu().numpy()
            else:
                body_np = np.array(vol_body)
            
            # Ensure binary mask
            body_mask_np = body_np > 0.5
            
            # --- Compute Metrics ---
            if np.any(body_mask_np):
                diff = (preds_3d - labels_np)[body_mask_np]
            else:
                diff = (preds_3d - labels_np).flatten()
            
            l1 = float(np.mean(np.abs(diff)))
            mse = float(np.mean(diff ** 2))
            rmse = math.sqrt(mse)
            
            peak = float(np.max(labels_np)) 
            if peak <= 0: peak = float(np.max(preds_3d))
            if peak <= 0: peak = 1.0
            
            psnr = float('inf') if rmse == 0 else 20.0 * math.log10(peak / (rmse + 1e-12))
            
            print(f"[Case: {case_id}] Slices: {D} | MAE: {l1:.6f} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | PSNR: {psnr:.3f} dB")
            
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
                for z in range(D):
                    # Access z-th slice
                    s_pred = preds_3d[0, :, :, z]
                    s_true = labels_np[0, :, :, z]
                    
                    # 3-column layout (GT, Pred, Diff)
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
                    
                    # Dynamic range
                    true_slice_max = np.max(s_true)
                    pred_slice_max = np.max(s_pred)
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
                    
                    # --- Add colorbars ---
                    fig.subplots_adjust(right=0.85, wspace=0.05)
                    
                    # Dose colorbar
                    cbar_ax1 = fig.add_axes([0.87, 0.55, 0.02, 0.35])
                    cbar1 = fig.colorbar(im1, cax=cbar_ax1)
                    cbar1.set_label('Dose (Gy)', fontsize=10)
                    
                    # Difference colorbar
                    cbar_ax2 = fig.add_axes([0.87, 0.1, 0.02, 0.35])
                    cbar2 = fig.colorbar(im3, cax=cbar_ax2)
                    cbar2.set_label('Difference (Gy)', fontsize=10)
                    
                    # --- Add main title ---
                    fig.suptitle(f"Patient: {case_id} - Slice {z:03d}/{D-1}", fontsize=14, y=0.95)
                    
                    plt.savefig(vis_dir / f"{case_id}_slice_{z:03d}.png", bbox_inches='tight', dpi=100)
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