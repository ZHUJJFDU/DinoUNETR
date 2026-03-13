import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Config ---
gt_dir = r'C:\Users\LT\Desktop\tmp\GT'
pred_dir = r'C:\Users\LT\Desktop\tmp\deeplab'

vmax_diff = 10.0  # 已修改为 5.0
# --------------

def save_diff_colorbar(vmax, output_path):
    fig, ax = plt.subplots(figsize=(1.5, 8))
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    # bwr 是标准的 Blue-White-Red 映射
    cb = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap('bwr'),
                                   norm=norm, orientation='vertical')
    cb.set_label('Dose Difference (GT - Pred) [Gy]', fontsize=12)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def load_gt_dose_and_mask(npz_path):
    try:
        data = np.load(npz_path, allow_pickle=True)
        data_dict = data['arr_0'].item() if 'arr_0' in data else dict(data)
        if 'dose' in data_dict and 'dose_scale' in data_dict:
            gt_dose = data_dict['dose'].astype(np.float64) * data_dict.get('dose_scale', 1.0)
        elif 'label' in data_dict:
            gt_dose = data_dict['label'].astype(np.float64)
        else: return None, None
        body_mask = data_dict.get('Body', None)
        return gt_dose, body_mask
    except: return None, None

def process_diff():
    output_base = os.path.join(pred_dir, f"pure_diff_slices_vmax{vmax_diff}")
    if not os.path.exists(output_base): os.makedirs(output_base)
    
    # 保存独立的 5Gy 标尺
    save_diff_colorbar(vmax_diff, os.path.join(output_base, "diff_colorbar_legend.png"))

    gt_files = glob.glob(os.path.join(gt_dir, "*.npz"))
    cmap = plt.get_cmap('bwr')

    for gt_file in gt_files:
        pid = os.path.splitext(os.path.basename(gt_file))[0]
        p_path = os.path.join(pred_dir, f"{pid}_pred.npy")
        if not os.path.exists(p_path) and pid.startswith('0'):
            p_path = os.path.join(pred_dir, f"{pid[1:]}_pred.npy")
        
        if not os.path.exists(p_path): continue
        
        gt_dose, body_mask = load_gt_dose_and_mask(gt_file)
        try:
            pred_dose = np.squeeze(np.load(p_path))
            gt_dose = np.squeeze(gt_dose)
        except: continue

        # 计算差异
        diff = gt_dose - pred_dose
        if body_mask is not None:
            body_mask = np.squeeze(body_mask)

        # 归一化到 [0, 1] 以匹配 cmap
        # 这里就是视觉“放大”发生的地方：所有 >= 5 的都会被 clip 到 1.0 (深红)
        diff_norm = np.clip(diff, -vmax_diff, vmax_diff)
        diff_scaled = (diff_norm + vmax_diff) / (2 * vmax_diff)
        
        case_dir = os.path.join(output_base, pid)
        if not os.path.exists(case_dir): os.makedirs(case_dir)

        for i in range(diff_scaled.shape[0]):
            res_rgba = cmap(diff_scaled[i])
            res_bgr = (res_rgba[..., :3][..., ::-1] * 255).astype(np.uint8)
            
            # 只有 Body Mask 内部显示颜色，外部设为纯黑
            if body_mask is not None:
                res_bgr[body_mask[i] == 0] = 0
            
            cv2.imwrite(os.path.join(case_dir, f"diff_{i:04d}.png"), res_bgr)
        print(f"Done: {pid}")

if __name__ == "__main__":
    process_diff()