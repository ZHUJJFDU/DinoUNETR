import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------- 1. 加载数据 ----------------------
# 加载原始数据
path_orig = r'D:\GDP-HMM_Challenge\train\0522c0001+9Ag+MOS_23629.npz'
data_npz_orig = np.load(path_orig, allow_pickle=True)
dict_orig = dict(data_npz_orig)['arr_0'].item()

# 加载预处理后的数据
path_proc = r'C:\Users\960\Desktop\Top2\data\dataset_update_NaH\0522c0001+9Ag+MOS_23629.npz'
data_npz_proc = np.load(path_proc, allow_pickle=True)
dict_proc = dict(data_npz_proc)['arr_0'].item()

# ---------------------- 2. 定义绘图帮助函数 ----------------------
def plot_ortho(data_dict, key_list, fig_title):
    """
    通用绘图函数：传入数据字典和要画的key列表
    """
    # 获取中心点坐标 (转为整数)
    iso = [int(x) for x in data_dict['isocenter']]
    
    # 多少个变量就画多少行
    n_rows = len(key_list)
    n_cols = 3 # Axial, Sagittal, Coronal
    
    # 创建画布，高度根据行数动态调整
    fig = plt.figure(figsize=(15, 4 * n_rows))
    fig.suptitle(fig_title, fontsize=16, fontweight='bold')
    
    for i, key in enumerate(key_list):
        if key not in data_dict:
            print(f"Warning: Key '{key}' not found in data.")
            continue
            
        vol = data_dict[key]
        
        # 确保数据至少是3维的，防止报错
        if len(vol.shape) < 3:
            print(f"Skipping {key}: shape is {vol.shape}, not 3D.")
            continue

        # 选择配色方案：Dose或Priority通常用彩色(jet)，解剖结构用灰度(gray)
        cmap = 'jet' if 'dose' in key.lower() or 'priority' in key.lower() or 'dist' in key.lower() else 'gray'

        # --- Axial (横断面) ---
        ax1 = fig.add_subplot(n_rows, n_cols, i*3 + 1)
        # 原始代码逻辑: ct[z]
        ax1.imshow(vol[iso[0]], cmap=cmap)
        ax1.set_title(f'{key} - Axial', fontsize=10)
        ax1.axis('off')

        # --- Sagittal (矢状面) ---
        ax2 = fig.add_subplot(n_rows, n_cols, i*3 + 2)
        # 原始代码逻辑: ct[::-1, y, :] (Z轴翻转)
        ax2.imshow(vol[::-1, iso[1], :], cmap=cmap)
        ax2.set_title(f'{key} - Sagittal', fontsize=10)
        ax2.axis('off')

        # --- Coronal (冠状面) ---
        ax3 = fig.add_subplot(n_rows, n_cols, i*3 + 3)
        # 原始代码逻辑: ct[::-1, ::-1, x] (Z轴和Y轴都翻转)
        ax3.imshow(vol[::-1, ::-1, iso[2]], cmap=cmap)
        ax3.set_title(f'{key} - Coronal', fontsize=10)
        ax3.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # 调整布局给title留位置
    plt.show()

# ---------------------- 3. 执行可视化 ----------------------

# === 任务1: 可视化原始数据 ===
keys_to_plot_orig = ['img', 'dose', 'angle_plate', 'beam_plate']
print(f"正在绘图: 原始数据 ({len(keys_to_plot_orig)} items)...")
plot_ortho(dict_orig, keys_to_plot_orig, "Original Data Visualization")

# === 任务2: 可视化预处理数据 ===
# 注意：Body 在你的key列表里首字母大写，其他有些是小写，请确保key名称完全一致
keys_to_plot_proc = [
    'comb_optptv', 
    'comb_oar_priority', 
    'comb_oar_distance', 
    'Body', 
    'img',
    'beam_plate_norm'
]
print(f"正在绘图: 预处理数据 ({len(keys_to_plot_proc)} items)...")
plot_ortho(dict_proc, keys_to_plot_proc, "Preprocessed Data Visualization"),