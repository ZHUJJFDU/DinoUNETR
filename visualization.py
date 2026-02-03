import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

# ---------------------- 1. 加载数据 ----------------------
# 加载原始数据
path_orig = r'D:\GDP-HMM_Challenge\valid_dose\0522c0002+15Ag+MOS_21176.npz'
data_npz_orig = np.load(path_orig, allow_pickle=True)
dict_orig = dict(data_npz_orig)['arr_0'].item()
print(dict_orig.keys())

# 加载预处理后的数据
path_proc = r'C:\Users\960\Desktop\Top2\data\dataset_update_Lung\0617-355467+imrt+MOS_23242.npz'
data_npz_proc = np.load(path_proc, allow_pickle=True)
dict_proc = dict(data_npz_proc)['arr_0'].item()

# ---------------------- 2. 定义绘图帮助函数 ----------------------
def plot_ortho(data_dict, key_list, fig_title):
    """
    通用绘图函数：传入数据字典和要画的key列表
    并保存Axial视角的PNG图片
    """
    # 获取中心点坐标 (转为整数)
    iso = [int(x) for x in data_dict['isocenter']]

    # 定义配色方案
    # 使用 'magma' (黑/紫 -> 红 -> 黄 -> 白) 这种配色既高对比度又美观，非常有“科技感”
    cmap_base = plt.get_cmap('magma')
    # 设定背景色 r=0, g=0, b=127
    bg_color = (0, 0, 127/255)
    cmap_base.set_bad(color=bg_color)

    # 多少个变量就画多少行
    n_rows = len(key_list)
    n_cols = 3 # Axial, Sagittal, Coronal
    
    # 创建画布，高度根据行数动态调整
    fig = plt.figure(figsize=(15, 4 * n_rows), facecolor=bg_color) # 整个画布背景统一
    fig.suptitle(fig_title, fontsize=16, fontweight='bold', color='white') # 标题设为白

    
    for i, key in enumerate(key_list):
        if key not in data_dict:
            print(f"Warning: Key '{key}' not found in data.")
            continue
            
        vol = data_dict[key]
            
        # --- 确定配色和数据处理 ---
        # 默认情况
        is_mass_density = 'mass_density' in key
        is_dose = 'dose' in key.lower() or 'label' in key.lower()
        
        if is_mass_density:
            # Mass density: 保持灰色
            cmap_to_use = 'gray'
            vol_to_plot = vol
        elif is_dose:
            # Dose: Blue -> Red (Jet)
            # 使用 jet 配色，并设置背景为黑
            cmap_dose = plt.get_cmap('jet')
            cmap_dose.set_bad(color=bg_color)
            cmap_to_use = cmap_dose
            # Mask掉0值以便显示背景色
            vol_to_plot = np.ma.masked_where(vol == 0, vol)
        else:
            # 其他: 使用 Magma 配色
            cmap_to_use = cmap_base
            # Mask掉0值以便显示背景色
            vol_to_plot = np.ma.masked_where(vol == 0, vol)

        # --- 提取 Axial 切片用于保存和显示 ---
        if len(vol.shape) == 2:
            slice_axial = vol_to_plot
        elif len(vol.shape) == 3:
            slice_axial = vol_to_plot[iso[0]]
        else:
            print(f"Skipping {key}: shape is {vol.shape}, not 2D or 3D.")
            continue

        # --- 保存 Axial 图片为 PNG ---
        # 注意: imsave 会归一化数据到 0-1 映射颜色
        try:
            save_name = f"{key}_Axial.png"
            # 保存时使用黑色背景
            # 显式指定 vmin/vmax 防止单值array (如Binary Mask) 归一化出错导致全黑
            v_max = vol.max()
            v_min = 0 # 假设医学数据非负，且我们希望0对应背景
            
            plt.imsave(save_name, slice_axial, cmap=cmap_to_use, vmin=v_min, vmax=v_max)
            print(f"Saved {save_name}")
        except Exception as e:
            print(f"Error saving {key}: {e}")

        # --- 绘图 (Matplotlib Figure) ---
        # 设定子图样式的辅助函数
        def style_ax(ax, title):
            ax.set_title(title, fontsize=10, color='white')
            ax.axis('off')
            ax.set_facecolor(bg_color)
        
        # 绘图也统一使用相同的 vmin/vmax
        v_max = vol.max()
        v_min = 0

        if len(vol.shape) == 2:
            # 2D 数据处理 (如 angle_plate)，直接在三个视图显示同一张图
            # Axial
            ax1 = fig.add_subplot(n_rows, n_cols, i*3 + 1)
            ax1.imshow(vol_to_plot, cmap=cmap_to_use, vmin=v_min, vmax=v_max)
            style_ax(ax1, f'{key} - Axial (2D)')

            # Sagittal
            ax2 = fig.add_subplot(n_rows, n_cols, i*3 + 2)
            ax2.imshow(vol_to_plot, cmap=cmap_to_use, vmin=v_min, vmax=v_max)
            style_ax(ax2, f'{key} - Sagittal (2D)')

            # Coronal
            ax3 = fig.add_subplot(n_rows, n_cols, i*3 + 3)
            ax3.imshow(vol_to_plot, cmap=cmap_to_use, vmin=v_min, vmax=v_max)
            style_ax(ax3, f'{key} - Coronal (2D)')

        elif len(vol.shape) == 3:
            # --- Axial (横断面) ---
            ax1 = fig.add_subplot(n_rows, n_cols, i*3 + 1)
            # 这里的 vol_to_plot 已经是 Masked Array 或者 原图
            # 注意: Masked Array 切片后仍然是 Masked Array
            data_ax = vol_to_plot[iso[0]] if hasattr(vol_to_plot, 'mask') else vol_to_plot[iso[0]]
            ax1.imshow(data_ax, cmap=cmap_to_use, vmin=v_min, vmax=v_max)
            style_ax(ax1, f'{key} - Axial')

            # --- Sagittal (矢状面) ---
            ax2 = fig.add_subplot(n_rows, n_cols, i*3 + 2)
            data_sag = vol_to_plot[::-1, iso[1], :]
            ax2.imshow(data_sag, cmap=cmap_to_use, vmin=v_min, vmax=v_max)
            style_ax(ax2, f'{key} - Sagittal')

            # --- Coronal (冠状面) ---
            ax3 = fig.add_subplot(n_rows, n_cols, i*3 + 3)
            data_cor = vol_to_plot[::-1, ::-1, iso[2]]
            ax3.imshow(data_cor, cmap=cmap_to_use, vmin=v_min, vmax=v_max)
            style_ax(ax3, f'{key} - Coronal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # 调整布局给title留位置
    plt.show()

# ---------------------- 3. 执行可视化 ----------------------

# # === 任务1: 可视化原始数据 ===
# keys_to_plot_orig = ['img', 'dose', 'angle_plate', 'beam_plate']
# print(f"正在绘图: 原始数据 ({len(keys_to_plot_orig)} items)...")
# plot_ortho(dict_orig, keys_to_plot_orig, "Original Data Visualization")

# === 任务2: 可视化预处理数据 ===
# 注意：Body 在你的key列表里首字母大写，其他有些是小写，请确保key名称完全一致
keys_to_plot_proc = [
    'comb_optptv', 
    'comb_oar_priority', 
    'comb_oar_distance', 
    'Body', 
    'mass_density',
    'beam_plate_norm',
    'label'
]
print(f"正在绘图: 预处理数据 ({len(keys_to_plot_proc)} items)...")
plot_ortho(dict_proc, keys_to_plot_proc, "Preprocessed Data Visualization"),