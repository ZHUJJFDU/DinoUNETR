import pandas as pd
import numpy as np
import os
import glob
import math
from skimage.metrics import structural_similarity as ssim
# --- Config ---
pred_dir = r'C:\Users\LT\Desktop\DinoUNETR\results\result_default_256_meddino_distance_nah&lung'
GT_DIR = r'D:\GDP-HMM_Challenge\valid'

def compute_metrics(gt, pred, mask=None):
    if mask is not None:
        gt_masked = gt[mask]
        pred_masked = pred[mask]
    else:
        gt_masked = gt.flatten()
        pred_masked = pred.flatten()
    
    if len(gt_masked) == 0:
        return None
    
    # MAE
    mae = np.mean(np.abs(gt_masked - pred_masked))
    
    # MSE
    mse = np.mean((gt_masked - pred_masked) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # PSNR
    max_val = np.max(gt_masked) 
    if max_val == 0:
        psnr = 0
    else:
        psnr = 20 * math.log10(max_val / rmse) if rmse > 0 else 100 

    # SSIM
    try:
        data_range = gt.max() - gt.min()
        if data_range == 0:
             data_range = 1.0
        ssim_val = ssim(gt, pred, data_range=data_range)
    except Exception as e:
        print(f"SSIM computation failed: {e}")
        ssim_val = np.nan
        
    stats = {
        'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'PSNR': psnr, 'SSIM': ssim_val,
        'pred_min': np.min(pred_masked), 'pred_max': np.max(pred_masked), 'pred_mean': np.mean(pred_masked),
        'gt_min': np.min(gt_masked), 'gt_max': np.max(gt_masked), 'gt_mean': np.mean(gt_masked)
    }
    return stats

def find_file_in_data(filename):
    # Generic fallback search in 'data' directory
    for root, dirs, files in os.walk('data'):
        if filename in files:
            return os.path.join(root, filename)
    return None

def process_case(gt_filename):
    local_npz_path = os.path.join(GT_DIR, gt_filename)
    filename = gt_filename
    patient_id = os.path.splitext(filename)[0]
    
    # 1. Find Prediction
    pred_filename = f"{patient_id}_pred.npy"
    pred_path = os.path.join(pred_dir, pred_filename)
    
    if not os.path.exists(pred_path):
        if patient_id.startswith('0'):
            alt_path = os.path.join(pred_dir, f"{patient_id[1:]}_pred.npy")
            if os.path.exists(alt_path):
                pred_path = alt_path
            else:
                return None
        else:
             return None

    # 2. Find GT NPZ in GT_DIR (Already found above)

    try:
        data = np.load(local_npz_path, allow_pickle=True)
        # Handle 'arr_0' dict wrapping
        if 'arr_0' in data:
            data_dict = data['arr_0'].item()
        else:
            data_dict = dict(data)

            
        # Get GT Dose
        if 'dose' in data_dict and 'dose_scale' in data_dict:
            raw_dose = data_dict['dose'].astype(np.float64)
            gt_dose = raw_dose * data_dict.get('dose_scale', 1.0)
            key_used = 'dose * dose_scale'
                
        elif 'label' in data_dict:
            # Fallback for training files which might already be processed?
            # But valid/test are usually raw.
            gt_dose = data_dict['label']
            key_used = 'label'
            # Check if this label is already scaled (e.g. range 0-7 vs 0-80). 
            # If so, might need *10. But user says use dose*dose_scale.
        else:
            print(f"No valid dose/label in {local_npz_path}")
            return None
        
        # No arbitrary *10 here, trusting the scaling logic above.
        
        # Get Body Mask if available
        if 'Body' in data_dict:
            body_mask = data_dict['Body'] > 0
        else:
            body_mask = None
        

    except Exception as e:
        print(f"Error loading {patient_id}: {e}")
        return None

    # Load Prediction
    try:
        pred_dose = np.load(pred_path)
    except:
        return None
        
    if pred_dose.shape != gt_dose.shape:
        print(f"Shape mismatch {patient_id}: {pred_dose.shape} vs {gt_dose.shape}")
        return None

    # Calculate Metrics
    # Only use Body mask if available, no crop mask
    metrics = compute_metrics(gt_dose, pred_dose, body_mask)
    if metrics:
        metrics['ID'] = patient_id
        return metrics
    return None

def main():
    print(f"Scanning GT_DIR: {GT_DIR}...")
    gt_files = glob.glob(os.path.join(GT_DIR, "*.npz"))
    print(f"Found {len(gt_files)} cases.")
    
    all_metrics = []
    lung_metrics = []
    nah_metrics = []
    
    print("Evaluating...")
    count = 0
    for gt_file in gt_files:
        gt_filename = os.path.basename(gt_file)
        res = process_case(gt_filename)
        if res:
            # Classification Logic
            pid = res['ID']
            if pid.startswith('0617') or pid.upper().startswith('LUNG'):
                res['Site'] = 'Lung'
                lung_metrics.append(res)
            else:
                res['Site'] = 'NaH'
                nah_metrics.append(res)
                
            all_metrics.append(res)
            count += 1
            print(f"[{res['ID']}] MAE: {res['MAE']:.4f} | MSE: {res['MSE']:.4f} | RMSE: {res['RMSE']:.4f} | PSNR: {res['PSNR']:.4f} | SSIM: {res['SSIM']:.4f}")

    if not all_metrics:
        print("No cases evaluated successfully.")
        return
        
    # Aggregate
    df_res = pd.DataFrame(all_metrics)
    
    print("\n" + "="*30)
    print("RESULTS Summary")
    print("="*30)
    
    # 1. Lung Summary
    if lung_metrics:
        df_lung = pd.DataFrame(lung_metrics)
        print("\n--- LUNG Summary ---")
        print(df_lung[['MAE', 'MSE', 'RMSE', 'PSNR', 'SSIM']].mean())
    else:
        print("\n--- LUNG Summary ---")
        print("No Lung cases found.")

    # 2. NaH Summary
    if nah_metrics:
        df_nah = pd.DataFrame(nah_metrics)
        print("\n--- NaH Summary ---")
        print(df_nah[['MAE', 'MSE', 'RMSE', 'PSNR', 'SSIM']].mean())
    else:
        print("\n--- NaH Summary ---")
        print("No NaH cases found.")

    # 3. Total Summary
    print("\n--- TOTAL Summary ---")
    print(df_res[['MAE', 'MSE', 'RMSE', 'PSNR', 'SSIM']].mean())
    
    save_csv_path = 'evaluation_results_test.csv'
    df_res.to_csv(save_csv_path, index=False)
    print(f"\nDetailed results saved to {save_csv_path}")

if __name__ == "__main__":
    main()
