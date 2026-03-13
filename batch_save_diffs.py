import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# --- Config ---
script_path = r'c:\Users\960\Desktop\Top2\save_diff_slices.py'
gt_dir = r'D:\GDP-HMM_Challenge\valid_dose'

pred_dirs = [
    r'C:\Users\960\Desktop\DinoUNETR\results\result_default_256_meddino_changechannel_nah&lung',
    r'C:\Users\960\Desktop\DinoUNETR\results\result_default_256_meddino_changechannel_nah&lung1',
    r'C:\Users\960\Desktop\baseline\results\baseline_nah&lung',
    r'c:\Users\960\Desktop\Top2\results\mednext_nah&lung2',
    r'c:\Users\960\Desktop\Top2\results\mednext_nah&lung3',
    r'c:\Users\960\Desktop\compare\PRUnet\results\result_256_prunet',
    r'c:\Users\960\Desktop\compare\DeepLab\results\result_256_deeplab'
]

# Number of parallel tasks
MAX_WORKERS = 5 
# --------------

def process_single_dir(pred_dir):
    if not os.path.exists(pred_dir):
        print(f"[Warning] Directory does not exist, skipping: {pred_dir}")
        return
        
    print(f"[Start] Processing: {pred_dir}")
    
    command = [
        'python', script_path,
        '--gt_dir', gt_dir,
        '--pred_dir', pred_dir
    ]
    
    try:
        # Run the command and wait for it to finish
        subprocess.run(command, check=True)
        print(f"[Success] Completed: {pred_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[Error] Failed to process {pred_dir}: {e}")
    except Exception as e:
        print(f"[Unexpected Error] {pred_dir}: {e}")

def run_batch():
    print(f"Starting batch processing with {MAX_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_single_dir, pred_dirs)

if __name__ == "__main__":
    run_batch()
    print("\nAll tasks finished.")
