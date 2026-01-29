import lmdb
import os
import glob
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor

def process_file(file_path):
    """
    Worker function to process a single .npz file.
    Returns serialized bytes of the dictionary or None if failed.
    """
    try:
        # Load NPZ
        npz = np.load(file_path, allow_pickle=True)
        
        # Extract Data
        # We save valid numpy arrays (not tensors) to save space and pickle faster
        save_dict = {
            'data': npz['data'],         # (C, H, W)
            'label': npz['label'],       # (1, H, W)
            'body': npz['body'],         # (1, H, W)
            'id': os.path.basename(file_path).replace('.npz', '')
        }

        # Optional keys
        if 'ptv' in npz: save_dict['ptv'] = npz['ptv']
        if 'oar_serial' in npz: save_dict['oar_serial'] = npz['oar_serial']
        if 'oar_parallel' in npz: save_dict['oar_parallel'] = npz['oar_parallel']
        if 'isocenter' in npz: save_dict['isocenter'] = npz['isocenter']
        if 'spacing' in npz: save_dict['spacing'] = npz['spacing']
        
        # Handle angle_list robustly like in the original loader
        if 'angle_list' in npz:
            raw_angles = npz['angle_list']
            if raw_angles.shape == ():
                    save_dict['angle_list'] = raw_angles.item()
            else:
                    save_dict['angle_list'] = raw_angles.tolist()
        
        # Return serialized data
        return pickle.dumps(save_dict)
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def make_lmdb(src_path, dst_path, map_size=1099511627776, num_workers=4): # 1TB map size default
    """
    Reads all .npz files from src_path and writes them into an LMDB environment at dst_path.
    Uses multiprocessing to speed up reading and pickling.
    """
    if not os.path.exists(src_path):
        raise ValueError(f"Source path {src_path} does not exist.")
    
    # 1. Find all .npz files
    file_list = sorted(glob.glob(os.path.join(src_path, "*.npz")))
    if not file_list:
        raise ValueError(f"No .npz files found in {src_path}")
    
    print(f"Found {len(file_list)} files. Creating LMDB at {dst_path}...")
    
    # Ensure destination directory exists
    os.makedirs(dst_path, exist_ok=True)
    
    # 2. Open LMDB Environment
    env = lmdb.open(dst_path, map_size=map_size)
    
    # 3. Write data using Parallel Processing
    print(f"Using {num_workers} workers for processing...")
    
    with env.begin(write=True) as txn:
        # Save dataset length
        txn.put('__len__'.encode(), str(len(file_list)).encode())
        
        # Use ProcessPoolExecutor for parallel processing
        # Using executor.map ensures results are returned in the same order as file_list
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create an iterator over results
            results = executor.map(process_file, file_list)
            
            count = 0
            for idx, serialized_data in enumerate(tqdm(results, total=len(file_list))):
                if serialized_data is not None:
                    # Write to LMDB
                    txn.put(str(idx).encode(), serialized_data)
                    count += 1
                    
                    # Commit every 1000 records
                    if count % 1000 == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                else:
                    print(f"Warning: Failed to load file index {idx}")

    env.close()
    print("LMDB creation complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Path to source root directory containing Train/Valid/Test folders')
    parser.add_argument('--dst', type=str, required=True, help='Path to destination root directory for .lmdb folders')
    parser.add_argument('--map_size', type=float, default=0.5, help='Map size in TB. Reduce this if you have limited disk space on Windows. (Default: 1.0 TB)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for parallel processing')
    args = parser.parse_args()
    
    # limits map_size to disk space on Windows often requires explicit sizing
    map_size_bytes = int(args.map_size * 1024 * 1024 * 1024 * 1024)
    
    splits = ['Train', 'Valid', 'Test']
    
    for split in splits:
        src_split = os.path.join(args.src, split)
        dst_split = os.path.join(args.dst, f"{split}.lmdb")
        
        if os.path.exists(src_split):
            print(f"================ Processing {split} ================")
            print(f"Source: {src_split}")
            print(f"Target: {dst_split}")
            print(f"Map Size: {args.map_size} TB ({map_size_bytes} bytes)")
            try:
                make_lmdb(src_split, dst_split, map_size=map_size_bytes, num_workers=args.num_workers)
            except Exception as e:
                print(f"Failed to process {split}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Skiping {split}: Folder {src_split} not found.")
