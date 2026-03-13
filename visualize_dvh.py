import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import os
from toolkit import NPZ2DVH_2Dose

def main():
    parser = argparse.ArgumentParser(description="Visualize Dose Volume Histogram (DVH) for radiotherapy plans in batch.")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the .npz data files.')
    parser.add_argument('--results_dir', type=str, default='results/lightning', help='Directory containing the .npy prediction files.')
    parser.add_argument('--meta_file', type=str, default='meta_files/meta_data_infer_val.csv', help='Path to meta_data.csv for site and dose information.')
    parser.add_argument('--ref_ptv_name', type=str, default=None, help='Override reference PTV name.')
    parser.add_argument('--ref_dose', type=float, default=None, help='Override prescribed dose.')
    
    args = parser.parse_args()

    # Define site-specific masks
    hnc_masks = [
        "PTVHighOPT", "PTVLowOPT", "PTVMidOPT", "Parotids-PTV",
        "Esophagus", "Larynx-PTV", "Lips", "BrainStem_03",
        "Mandible-PTV", "SpinalCord_05"
    ]

    lung_masks = [
        "PTV", "Total Lung-GTV", "SpinalCord", "Heart",
        "LAD", "Esophagus", "BrachialPlexus", "GreatVessels", "Trachea"
    ]

    # Create output directory
    output_dir = os.path.join(args.results_dir, 'dvh')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Load metadata if available
    meta_dict = {}
    if os.path.exists(args.meta_file):
        try:
            import pandas as pd
            df = pd.read_csv(args.meta_file)
            for _, row in df.iterrows():
                # Extract full ID from npz_path or PID
                # Matching logic: try to find the filename in npz_path
                full_id = os.path.basename(str(row['npz_path'])).replace('.npz', '')
                meta_dict[full_id] = {
                    'site': row['site'],
                    'dose': row['HighDose']
                }
            print(f"Loaded metadata for {len(meta_dict)} cases.")
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}. Falling back to heuristics.")

    # Iterate over all .npz files in data_dir
    files = [f for f in os.listdir(args.data_dir) if f.endswith('.npz')]
    print(f"Found {len(files)} .npz files in {args.data_dir}")

    for filename in files:
        plan_id = filename.replace('.npz', '')
        data_path = os.path.join(args.data_dir, filename)
        pred_path = os.path.join(args.results_dir, f'{plan_id}_pred.npy')

        if not os.path.exists(pred_path):
            continue

        print(f"Processing {plan_id}...")
        
        # Determine site, PTV name and dose
        ref_ptv = args.ref_ptv_name
        ref_dose = args.ref_dose
        site = 1 # Default HNC

        if plan_id in meta_dict:
            site = meta_dict[plan_id]['site']
            if ref_ptv is None:
                ref_ptv = 'PTVHighOPT' if site == 1 else 'PTV'
            if ref_dose is None:
                ref_dose = meta_dict[plan_id]['dose']
        
        # Select mask list based on site
        initial_masks = hnc_masks if site == 1 else lung_masks
        
        # Load data
        try:
            data_npz = np.load(data_path, allow_pickle=True)
            data_dict = dict(data_npz)['arr_0'].item()
            prediction = np.load(pred_path)
        except Exception as e:
            print(f"Error loading data for {plan_id}: {e}")
            continue

        # Filter masks: only keep those that actually exist in data_dict and have non-zero max
        needed_masks = []
        for m in initial_masks:
            if m in data_dict:
                if np.max(data_dict[m]) > 0:
                    needed_masks.append(m)
        
        if not needed_masks:
            print(f"Warning: No valid masks found for {plan_id}. Skipping.")
            continue

        # Fallback heuristics if still None
        if ref_ptv is None:
            ref_ptv = 'PTVHighOPT' if 'PTVHighOPT' in data_dict else 'PTV'
        if ref_dose is None:
            ref_dose = 60.0 # Default fallback
        
        save_path = os.path.join(output_dir, f'{plan_id}.png')

        # Generate DVH
        try:
            NPZ2DVH_2Dose(
                data_dict, 
                needed_mask=needed_masks, 
                additional_dose=prediction, 
                ref_ptv_name=ref_ptv, 
                ref_dose=ref_dose, 
                bin_size=0.5, 
                with_plt=True, 
                save_plt_path=save_path
            )
            print(f"Successfully saved DVH to {save_path}")
        except Exception as e:
            print(f"Error generating DVH for {plan_id}: {e}")



if __name__ == "__main__":
    main()
