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
    parser.add_argument('--strategy', default='lora', type=str, choices=['default', 'lora', 'llrd', 'frozen'], help='Model strategy used during training')
    parser.add_argument('--window_size', default=256, type=int, help='Sliding window size')
    parser.add_argument('--stride', default=128, type=int, help='Sliding window stride')
    parser.add_argument('--model_path', default='trained_models/trained_model_lora_full_256_crop_meddino/best_model-epoch=69-train_loss=0.0738.ckpt', type=str, help='Override model path')
    
    args = parser.parse_args()

    cfig = yaml.load(open(cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)
    
    # Overwrite strategy from args to ensure model is initialized correctly
    cfig['strategy'] = args.strategy

    if args.model_path:
        cfig['save_model_path'] = args.model_path
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Force Resize to 512x512 for input to sliding window
    cfig['loader_params']['in_size'] = [96, 512, 512]
    cfig['loader_params']['out_size'] = [96, 512, 512]
    
    # ------------ data loader (3D) -----------------#
    loaders = data_loader_lightning.GetLoader(cfig = cfig['loader_params'])
    test_loader = loaders.val_dataloader()

    # Determine Model Layout
    use_layout = cfig.get('layout', False)
    
    pl_module = GDPLightningModel.load_from_checkpoint(
        cfig['save_model_path'], 
        cfig=cfig, 
        strategy=args.strategy, 
        strict=True 
    )
    model = pl_module.model.to(device)

    save_pred_path = cfig.get('save_pred_path', 'results_window')
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)
    
    print(f"Starting sliding window inference on device: {device}")
    print(f"Window Size: {args.window_size}, Stride: {args.stride}")

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

                # Prepare layout data for this patient if needed
                layout_data_slice = None
                if use_layout:
                    spacing_b = batch['spacing'][b].unsqueeze(0).to(device)
                    isocenter_b = batch['ori_isocenter'][b].unsqueeze(0).to(device)
                    raw_angle_list = batch['angle_list']
                    patient_angles = [raw_angle_list[i][b].item() for i in range(len(raw_angle_list))]
                    angle_list_b = [patient_angles]
                    
                    layout_data_slice = {
                        'spacing': spacing_b,
                        'isocenter': isocenter_b,
                        'angle_list': angle_list_b
                    }

                vol_ptv_dose = vol_input[0]
                vol_oar_priority = vol_input[1]
                vol_beam_plate_norm = vol_input[5]
                
                C, D, H, W = vol_input.shape
                
                pred_slices = []
                
                # Slice-by-slice inference
                for d in range(D):
                    slice_input = vol_input[:, d, :, :].unsqueeze(0).to(device)

                    output = sliding_window_inference(
                        model, 
                        slice_input, 
                        layout_data=layout_data_slice, 
                        window_size=args.window_size, 
                        stride=args.stride, 
                        device=device
                    )
                    
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
                    for d in range(D):
                        s_pred = preds_np[0, d, :, :]
                        s_true = labels_np[0, d, :, :]

                        slice_ptv = vol_ptv_dose[d, :, :].cpu().numpy()
                        slice_oar = vol_oar_priority[d, :, :].cpu().numpy()
                        slice_beam = vol_beam_plate_norm[d, :, :].cpu().numpy()

                        mask_ptv = (slice_ptv > 0).astype(float) 
                        mask_oar = (slice_oar > 0).astype(float)
                        
                        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=100)
                        axes = axes.flatten()
                        
                        true_slice_max = np.max(s_true)
                        pred_slice_max = np.max(s_pred)
                        
                        vmax = max(true_slice_max, pred_slice_max)
                        if vmax <= 0: vmax = 1.0
                        vmin = 0.0
                        
                        ax1 = axes[0]
                        ax1.set_title(f"Ground Truth Dose\nMax: {true_slice_max:.2f} Gy", fontsize=12)
                        im1 = ax1.imshow(s_true, cmap='jet', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
                        ax1.axis('off')
                        
                        ax2 = axes[1]
                        ax2.set_title(f"Predicted Dose\nMax: {pred_slice_max:.2f} Gy", fontsize=12)
                        im2 = ax2.imshow(s_pred, cmap='jet', vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
                        ax2.axis('off')

                        # Plot Difference
                        ax3 = axes[2]
                        diff_map = s_pred - s_true
                        max_diff = np.max(np.abs(diff_map))
                        ax3.set_title(f"Difference (Pred - GT)\nMax Diff: {max_diff:.2f} Gy", fontsize=12)
                        im3 = ax3.imshow(diff_map, cmap='bwr', vmin=-max_diff, vmax=max_diff, aspect='equal', origin='lower')
                        ax3.axis('off')

                        # Plot PTV
                        ax4 = axes[3]
                        ax4.set_title("PTV Mask", fontsize=12)
                        ax4.imshow(slice_ptv, cmap='gray', aspect='equal', origin='lower')
                        ax4.axis('off')

                        # Plot OAR
                        ax5 = axes[4]
                        ax5.set_title("OAR Priority", fontsize=12)
                        ax5.imshow(slice_oar, cmap='jet', aspect='equal', origin='lower')
                        ax5.axis('off')

                        # Plot Beam
                        ax6 = axes[5]
                        ax6.set_title("Beam Plate", fontsize=12)
                        ax6.imshow(slice_beam, cmap='gray', aspect='equal', origin='lower')
                        ax6.axis('off')

                        plt.tight_layout()
                        plt.savefig(vis_dir / f"slice_{d:03d}.png")
                        plt.close()

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