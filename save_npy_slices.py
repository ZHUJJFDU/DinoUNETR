import os
import glob
import numpy as np
import cv2
import argparse

# --- Config ---
# default_input_dir = r'C:\Users\960\Desktop\Top2\results\mednext_nah&lung3' 
# You can change this manually or pass via command line
default_input_dir = r'C:\Users\960\Desktop\baseline\results\baseline_nah&lung'
# --------------

def normalize_to_uint8(data):
    """Normalize data to 0-255 and convert to uint8."""
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    
    # Handle constant value case
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.uint8)
    
    norm_data = (data - min_val) / (max_val - min_val) * 255.0
    return norm_data.astype(np.uint8)

def process_folder(input_dir, output_dir=None):
    if output_dir is None:
        # Create 'png_slices' inside the input directory
        output_dir = os.path.join(input_dir, "png_slices")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    print(f"Found {len(npy_files)} .npy files in {input_dir}")
    
    for npy_file in npy_files:
        filename = os.path.basename(npy_file)
        file_id = os.path.splitext(filename)[0]
        
        # Create subfolder for this file
        case_dir = os.path.join(output_dir, file_id)
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
            
        try:
            data = np.load(npy_file)
            # Squeeze and handle 4D (C, Z, H, W) where C=1
            data = np.squeeze(data)
            if data.ndim == 4 and data.shape[0] == 1:
                data = data[0]
            
            if data.ndim == 3:
                # Normalize entire volume at once to preserve relative intensity if needed
                # Or normalize per slice? Usually per-volume is better for dose.
                data_norm = normalize_to_uint8(data)
                
                # Save slices
                for i in range(data_norm.shape[0]):
                    slice_img = data_norm[i, :, :]
                    slice_img_color = cv2.applyColorMap(slice_img, cv2.COLORMAP_JET)
                    slice_name = f"slice_{i:04d}.png"
                    save_path = os.path.join(case_dir, slice_name)
                    cv2.imwrite(save_path, slice_img_color)
                
                print(f"[Done] {filename} -> {len(data_norm)} slices saved to {case_dir}")
                
            else:
                print(f"[Skip] {filename}: logical shape {data.shape} not supported (expected 3D).")
                
        except Exception as e:
            print(f"[Error] Failed to process {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NPY volumes to PNG slices.")
    parser.add_argument("--input_dir", type=str, default=default_input_dir, help="Directory containing .npy files")
    
    args = parser.parse_args()
    
    print(f"Input Directory: {args.input_dir}")
    if os.path.exists(args.input_dir):
        process_folder(args.input_dir)
    else:
        print("Error: Input directory does not exist.")
