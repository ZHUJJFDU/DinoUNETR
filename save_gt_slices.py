import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

# --- Config ---
default_input_dir = r'C:\Users\LT\Desktop\tmp' 
# --------------

def save_independent_colorbar(vmin, vmax, cmap_name, output_path):
    """单独生成并保存一个颜色条标尺"""
    fig, ax = plt.subplots(figsize=(1.5, 8)) # 细长比例
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(cmap_name),
                                   norm=norm, orientation='vertical')
    cb.set_label('Dose (Gy)', fontsize=12)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Independent colorbar saved to: {output_path}")

def process_folder(input_dir, vmin=0.0, vmax=70.0):
    output_dir = os.path.join(input_dir, "pure_slices")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 首先生成独立的 Bar
    save_independent_colorbar(vmin, vmax, 'jet', os.path.join(output_dir, "colorbar_legend.png"))

    files = glob.glob(os.path.join(input_dir, "*.npz")) + glob.glob(os.path.join(input_dir, "*.npy"))
    
    for f in files:
        file_id = os.path.splitext(os.path.basename(f))[0]
        case_dir = os.path.join(output_dir, file_id)
        if not os.path.exists(case_dir): os.makedirs(case_dir)

        # 加载数据逻辑 (兼容 npz 和 npy)
        if f.endswith('.npz'):
            data_load = np.load(f, allow_pickle=True)
            data_dict = data_load['arr_0'].item() if 'arr_0' in data_load else dict(data_load)
            data = data_dict.get('dose', data_dict.get('label'))
            if 'dose_scale' in data_dict: data = data * data_dict['dose_scale']
        else:
            data = np.load(f)

        data = np.squeeze(data)
        if data.ndim == 4: data = data[0]

        print(f"Processing {file_id}...")

        # 固定 0-70Gy 映射到 0-255 uint8
        # 公式: (val - vmin) / (vmax - vmin) * 255
        data_clipped = np.clip(data, vmin, vmax)
        data_norm = ((data_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

        for i in range(data_norm.shape[0]):
            slice_img = data_norm[i, :, :]
            # 使用 cv2 快速应用 JET 颜色
            slice_color = cv2.applyColorMap(slice_img, cv2.COLORMAP_JET)
            
            # 保存纯净图片 (无文字、无边框)
            save_path = os.path.join(case_dir, f"slice_{i:04d}.png")
            cv2.imwrite(save_path, slice_color)

if __name__ == "__main__":
    process_folder(default_input_dir)