import os
import subprocess

def main():
    # Configuration
    gt_dir = r'D:\GDP-HMM_Challenge\valid_dose'
    meta_file = r'c:\Users\960\Desktop\baseline\meta_files\meta_data_infer_val.csv'
    
    pred_dirs = [
        r'C:\Users\960\Desktop\DinoUNETR\results\result_default_256_meddino_changechannel_nah&lung',
        r'C:\Users\960\Desktop\DinoUNETR\results\result_default_256_meddino_changechannel_nah&lung1',
        r'C:\Users\960\Desktop\baseline\results\baseline_nah&lung',
        r'c:\Users\960\Desktop\Top2\results\mednext_nah&lung2',
        r'c:\Users\960\Desktop\Top2\results\mednext_nah&lung3',
        r'c:\Users\960\Desktop\compare\PRUnet\results\result_256_prunet',
        r'c:\Users\960\Desktop\compare\DeepLab\results\result_256_deeplab'
    ]

    script_path = r'c:\Users\960\Desktop\baseline\visualize_dvh.py'

    for pred_dir in pred_dirs:
        if not os.path.exists(pred_dir):
            print(f"Directory not found: {pred_dir}")
            continue
            
        print(f"\n" + "="*50)
        print(f"Processing results in: {pred_dir}")
        print("="*50)
        
        # Execute the visualization script for the current directory
        # The visualize_dvh.py script already handles intra-folder batching
        cmd = [
            'python', script_path,
            '--data_dir', gt_dir,
            '--results_dir', pred_dir,
            '--meta_file', meta_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully finished processing: {pred_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {pred_dir}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
