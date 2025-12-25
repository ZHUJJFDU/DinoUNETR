import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from nnunet_mednext import create_mednext_v1
import data_loader_dvh
from data_loader_dvh import HaN_OAR_DICT, Lung_OAR_DICT


def offset_spatial_crop(roi_center=None, roi_size=None):
    if roi_center is None or roi_size is None:
        raise ValueError("Both `roi_center` and `roi_size` must be specified.")
    # 将roi_center中的每个坐标c四舍五入并转为整数，确保后续切片索引为整数
    roi_center = [int(round(c)) for c in roi_center]
    # 将roi_size中的每个尺寸s四舍五入并转为整数，保证ROI尺寸为整数，避免后续数组切片报错
    roi_size = [int(round(s)) for s in roi_size]
    start = []
    end = []

    for center, size in zip(roi_center, roi_size):
        half_size = size // 2
        start_i = max(center - half_size, 0)
        end_i = max(start_i + size, start_i)
        start.append(start_i)
        end.append(end_i)
    return start, end

def cropped2ori(crop_data, ori_size, isocenter, trans_in_size):

    '''
    crop_data: the cropped data
    ori_size: the original size of the data
    isocenter: the isocenter of the original data
    trans_in_size: the in_size parameter in the transfromation of loader
    '''

    assert (np.array(trans_in_size) == np.array(crop_data.shape)).all()

    start_coords, end_coords = offset_spatial_crop(roi_center = isocenter, roi_size = trans_in_size)

    # remove the padding
    crop_start, crop_end = [], []
    for i in range(len(ori_size)):
        if end_coords[i] > ori_size[i]:
            diff = end_coords[i] - ori_size[i]
            crop_start.append(diff // 2)
            crop_end.append(crop_data.shape[i] - diff + diff // 2)
        else:
            crop_start.append(0)
            crop_end.append(crop_data.shape[i])

    crop_data = crop_data[crop_start[0]: crop_end[0], crop_start[1]: crop_end[1], crop_start[2]: crop_end[2]]
    pad_out = np.zeros(ori_size)
    pad_out[start_coords[0]: end_coords[0], start_coords[1]: end_coords[1], start_coords[2]: end_coords[2]] = crop_data 
    
    return pad_out


# ---------------- DVH metric helpers ---------------- #
def d_mean(dose_vals):
    """Dmean: mean dose over all voxels."""
    if dose_vals.size == 0:
        return np.nan
    return float(np.mean(dose_vals))


def d_x_percent(dose_vals, xx):
    """Dxx%: dose received by xx% of the volume (cumulative)."""
    if dose_vals.size == 0:
        return np.nan
    p = 1.0 - xx / 100.0
    return float(np.quantile(dose_vals, p))


def d_cc(dose_vals, voxel_vol_mm3, cc):
    """Generic near-maximum dose to `cc` cubic centimeters."""
    if dose_vals.size == 0:
        return np.nan
    hot_volume_mm3 = cc * 1000.0 # cc转化为mm3
    n_vox = int(np.ceil(hot_volume_mm3 / max(voxel_vol_mm3, 1e-6))) # mm3转化为像素数并向上取整
    n_vox = max(n_vox, 1) # 至少为1
    sort_desc = np.sort(dose_vals)[::-1] # 倒序排列
    n_vox = min(n_vox, sort_desc.size)
    return float(sort_desc[n_vox - 1]) # 返回第n_vox个最高剂量


def v_x_percent(dose_vals, xx):
    """Vxx%: percent volume receiving at least Dxx%."""
    if dose_vals.size == 0:
        return np.nan
    p = 1.0 - xx / 100.0
    return float(100.0 * np.quantile(dose_vals, p))


def v_x_gy(dose_vals, x_gy):
    """VxGy: percent volume receiving at least x Gy."""
    if dose_vals.size == 0:
        return np.nan
    # 统计剂量≥x Gy 的体素比例，mean(True=1, False=0) 正好等于该比例
    return float(100.0 * np.mean(dose_vals >= x_gy))


def safe_get_mask(roi_dict, name):
    arr = roi_dict.get(name, None)
    if arr is None:
        return None
    # some arrays may be float mask; ensure boolean
    try:
        return (arr > 0)
    except Exception:
        return None


def summarize_errors(err_list):
    """Return q25, q50, q75, mean±sd, abs_mean±abs_sd for a list of errors."""
    arr = np.array([e for e in err_list if np.isfinite(e)], dtype=float)
    if arr.size == 0:
        return None
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=0))
    abs_arr = np.abs(arr)
    abs_mean = float(np.mean(abs_arr))
    abs_sd = float(np.std(abs_arr, ddof=0))
    return {
        'q25': float(q25), 'q50': float(q50), 'q75': float(q75),
        'mean': mean, 'sd': sd,
        'abs_mean': abs_mean, 'abs_sd': abs_sd
    }


def load_oar_defs(cfig):
    han_df = pd.read_csv(cfig['csv_HaN_OAR_priority_root'])
    lung_df = pd.read_csv(cfig['csv_LUNG_OAR_priority_root'])
    han_defs = {r['OAR_Name']: bool(r['isDmax']) for _, r in han_df.iterrows()}
    lung_defs = {r['OAR_Name']: bool(r['isDmax']) for _, r in lung_df.iterrows()}
    return han_defs, lung_defs


def build_id_maps(csv_root):
    df = pd.read_csv(csv_root)
    id_to_npz = {}
    id_to_site = {}
    for _, r in df.iterrows():
        npz_path = str(r['npz_path'])
        # Use exact basename (with MOS serial) to match loader's ID
        id_key = os.path.basename(npz_path).replace('.npz', '')
        id_to_npz[id_key] = npz_path
        try:
            id_to_site[id_key] = int(r['site'])
        except Exception:
            # Fallback: default to Head & Neck if site missing
            id_to_site[id_key] = 1
    return id_to_npz, id_to_site


# Fixed ROI/metric plan replicating paper figure (Head&Neck, Lung)
METRIC_PLAN = {
    'Head and Neck': [
        ('PTVHighOPT', 'D99.5%'),
        ('PTVHighOPT', 'D0.03cc'),
        ('PTVHighOPT', 'V100%'),
        ('PTVMidOPT', 'D99.5%'),
        ('PTVMidOPT', 'V100%'),
        ('PTVLowOPT', 'D99.5%'),
        ('PTVLowOPT', 'V100%'),
        ('ParotidCon-PTV', 'Dmean'),
        ('Parotidlps-PTV', 'Dmean'),
        ('Parotids-PTV', 'Dmean'),
        ('BrainStem_03', 'D0.03cc'),
        ('SpinalCord_05', 'D0.03cc'),
        ('Esophagus', 'Dmean'),
        ('Larynx-PTV', 'Dmean'),
        ('Lips', 'Dmean'),
        ('Mandible-PTV', 'V70Gy'),
        ('OCavity-PTV', 'Dmean'),
        ('Submand-PTV', 'Dmean'),
    ],
    'Lung': [
        ('PTV', 'D0.03cc'),
        ('PTV', 'V90%'),
        ('PTV', 'V95%'),
        ('PTV', 'V100%'),
        ('Total Lung-GTV', 'Dmean'),
        ('Total Lung-GTV', 'V5Gy'),
        ('Total Lung-GTV', 'V20Gy'),
        ('SpinalCord', 'D0.03cc'),
        ('Esophagus', 'D0.03cc'),
        ('Esophagus', 'Dmean'),
        ('Heart', 'D0.03cc'),
        ('Heart', 'Dmean'),
    ],
}


def compute_metric_from_label(metric_label, dose_vals, voxel_vol_mm3, pdose=None):
    """Compute metric value given metric label string.
    Supports: Dxx%, Dmean, D0.03cc, D0.3cc, VxGy, Vxx% (needs pdose).
    """
    if dose_vals.size == 0:
        return np.nan
    m = metric_label.strip().lower()
    if m == 'dmean':
        return d_mean(dose_vals)
    if m.startswith('d') and m.endswith('%'):
        try:
            xx = float(m[1:-1])
            return d_x_percent(dose_vals, xx)
        except Exception:
            return np.nan
    if m.startswith('d') and 'cc' in m:
        try:
            cc = float(m[1:].replace('cc', ''))
            return d_cc(dose_vals, voxel_vol_mm3, cc)
        except Exception:
            return np.nan
    if m.startswith('v') and m.endswith('gy'):
        try:
            x = float(m[1:].replace('gy', ''))
            return v_x_gy(dose_vals, x)
        except Exception:
            return np.nan
    if m.startswith('v') and m.endswith('%'):
        # percent of PDose
        if pdose is None or pdose <= 0:
            return np.nan
        try:
            xx = float(m[1:-1])
            thr = (xx / 100.0) * pdose
            return v_x_gy(dose_vals, thr)
        except Exception:
            return np.nan
    return np.nan

def add_err(site_name, metric_label, err, agg):
        if not np.isfinite(err):
            return
        if metric_label not in agg[site_name]:
            agg[site_name][metric_label] = []
        agg[site_name][metric_label].append(float(err))

# Print summary results
def print_site(site_name, agg):
    print(f"=== {site_name} ===")
    metrics = agg[site_name]
    if len(metrics) == 0:
        print("No metrics collected.")
        return
    # Print strictly following paper order defined in METRIC_PLAN
    plan = METRIC_PLAN.get(site_name, [])
    printed = set()
    for roi_name, metric_label in plan:
        key = f"{roi_name}_{metric_label}"
        errs = metrics.get(key, None)
        if not errs:
            continue
        s = summarize_errors(errs)
        if s is None:
            continue
        printed.add(key)
        print(
            f"{key}: 25th={s['q25']:.3f}, 50th={s['q50']:.3f}, 75th={s['q75']:.3f}, "
            f"Mean±SD={s['mean']:.3f}±{s['sd']:.3f}, AbsMean±AbsSD={s['abs_mean']:.3f}±{s['abs_sd']:.3f}"
        )
    # Print any extra metrics (not in paper list) afterwards for debugging
    extras = [k for k in metrics.keys() if k not in printed]
    for k in sorted(extras):
        s = summarize_errors(metrics[k])
        if s is None:
            continue
        print(
            f"{k}: 25th={s['q25']:.3f}, 50th={s['q50']:.3f}, 75th={s['q75']:.3f}, "
            f"Mean±SD={s['mean']:.3f}±{s['sd']:.3f}, AbsMean±AbsSD={s['abs_mean']:.3f}±{s['abs_sd']:.3f}"
        )

def collect_site_rows(agg, site_name):
    metrics = agg[site_name]
    plan = METRIC_PLAN.get(site_name, [])
    rows = []
    for roi_name, metric_label in plan:
        key = f"{roi_name}_{metric_label}"
        errs = metrics.get(key, None)
        if not errs:
            continue
        s = summarize_errors(errs)
        if s is None:
            continue
        rows.append((roi_name, metric_label, s['q25'], s['q50'], s['q75'], s['mean'], s['sd'], s['abs_mean'], s['abs_sd']))
    return rows

def print_table(site_name, rows, out_fmt='plain', out_dir=None):
    import os
    import pandas as pd
    if out_fmt == 'csv':
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
        df = pd.DataFrame(rows, columns=['ROI','DVH','25th','50th','75th','Mean','SD','AbsMean','AbsSD'])
        fn = 'metrics_headneck.csv' if site_name == 'Head and Neck' else 'metrics_lung.csv'
        path = os.path.join(out_dir or '.', fn)
        try:
            df.to_csv(path, index=False)
            print(f"[saved] {site_name} -> {path}")
        except Exception as e:
            print(f"[warn] save failed: {e}")
        return
    if out_fmt == 'markdown':
        print(f"| {site_name} | | | | | | |")
        print("| ROI | DVH | 25th | 50th | 75th | Mean ± SD | AbsMean ± AbsSD |")
        for r in rows:
            roi, dvh, q25, q50, q75, mean, sd, am, asd = r
            print(f"| {roi} | {dvh} | {q25:.2f} | {q50:.2f} | {q75:.2f} | {mean:.2f} ± {sd:.2f} | {am:.2f} ± {asd:.2f} |")
        return
    print(f"=== {site_name} ===")
    print(f"{'ROI':20s} {'DVH':10s} {'25th':6s} {'50th':6s} {'75th':6s} {'Mean±SD':18s} {'AbsMean±AbsSD':18s}")
    for r in rows:
        roi, dvh, q25, q50, q75, mean, sd, am, asd = r
        print(f"{roi:20s} {dvh:10s} {q25:6.2f} {q50:6.2f} {q75:6.2f} {mean:6.2f} ± {sd:6.2f} {am:6.2f} ± {asd:6.2f}")

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('cfig_path',  type = str)
    parser.add_argument('--phase', default = 'valid', type = str)
    parser.add_argument('--out_fmt', default='plain', type=str)
    parser.add_argument('--out_dir', default=None, type=str)
    args = parser.parse_args()

    cfig = yaml.load(open(args.cfig_path), Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------ data loader -----------------#
    loaders = data_loader_infer.GetLoader(cfig = cfig['loader_params'])
    phase = (args.phase or 'valid').lower()
    if phase in ('valid', 'val'):
        test_loader = loaders.val_dataloader()
    elif phase in ('test', 'infer'):
        test_loader = loaders.test_dataloader()
    elif phase in ('train', 'training'):
        test_loader = loaders.train_dataloader()
    else:
        print(f"[Warn] Unknown phase '{args.phase}', defaulting to 'valid'.")
        test_loader = loaders.val_dataloader()

    if cfig['model_from_lightning']:
        from train_lightning import GDPLightningModel
        pl_module = GDPLightningModel.load_from_checkpoint(cfig['save_model_path'], cfig=cfig, strict=True)
        model = pl_module.model.to(device)
    else:
        try:
            model = create_mednext_v1(
                num_input_channels=cfig['model_params']['num_input_channels'],
                num_classes=cfig['model_params']['out_channels'],
                model_id=cfig['model_params']['model_id'],
                kernel_size=cfig['model_params']['kernel_size'],
                deep_supervision=cfig['model_params']['deep_supervision']
            ).to(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        model.load_state_dict(torch.load(cfig['save_model_path'], map_location=device))
    
    
    # Prepare ROI definitions and patient meta maps
    id_to_npz, id_to_site = build_id_maps(cfig['loader_params']['csv_root'])
    scale_dose_dict = json.load(open(cfig['loader_params']['scale_dose_dict'], 'r'))

    # Aggregators: {site_name: {metric_label: [errors]}}
    agg = {
        'Head and Neck': {},
        'Lung': {}
    }

    with torch.no_grad():
        model.eval()
        for batch_idx, data_dict in enumerate(test_loader):
            print(data_dict.keys())
            input = data_dict['data'].to(device)
            outputs = model(input)
            if cfig['act_sig']:
                outputs = torch.sigmoid(outputs.clone())
            outputs = outputs * cfig['scale_out']

            # iterate items in batch
            # 推理时batchsize确实为1，但保留循环写法，兼容后续可能调整batchsize>1的情况
            for index in range(outputs.shape[0]):
                # data_dict['id'] 是长度 = batch_size 的 list，所以取第 index 个样本
                patient_id = data_dict['id'][index]
                # prefer site from loader; fallback to CSV maps
                site_num = data_dict['site'][index]
                       
                # reconstruct prediction to original size and rescale to Gy
                ori_size = data_dict['ori_img_size'][index].numpy().tolist()

                isocenter = data_dict['ori_isocenter'][index].numpy().tolist()

                # prefer in_size from dataloader; fallback to config

                in_size = cfig['loader_params']['in_size']

                body_mask_crop = (data_dict['Body'][index][0].cpu().numpy() > 0).astype(np.float32)

                crop_pred = outputs[index][0].cpu().numpy()
                crop_pred[body_mask_crop == 0] = 0

                # 输入 剪裁过的预测结果 原始大小 源点坐标 输入大小
                pred_gy = cropped2ori(crop_pred, ori_size, isocenter, in_size) * cfig['loader_params']['dose_div_factor']

                # spacing and body mask from loader
                spacing = data_dict['spacing'][index].cpu().numpy().tolist()
                
                voxel_vol_mm3 = float(spacing[0] * spacing[1] * spacing[2])
                body_mask = cropped2ori(body_mask_crop, ori_size, isocenter, in_size) > 0.5

                # decide site
                site_name = 'Head and Neck' if site_num < 1.5 else 'Lung'

                # determine primary PTVHigh name and PDose
                pid_only = patient_id.split('+')[0]
                ptv_info = scale_dose_dict.get(pid_only, {}).get('PTV_High', None)
                ptv_high_name = ptv_info['OPTName'] if ptv_info and 'OPTName' in ptv_info else 'PTVHighOPT'
                # GT dose from loader label (already normalized), reconstructed to original size
                gt_gy = None
                gt_crop = data_dict['label'][index][0].cpu().numpy()
                
                gt_gy = cropped2ori(gt_crop, ori_size, isocenter, in_size) * cfig['loader_params']['dose_div_factor']

                # gt_gy 和 pred_gy 

                # PTV masks from loader channels
                ptv_high_mask = None
                ptv_mid_mask = None
                ptv_low_mask = None

                cat_ptv = data_dict['cat_ptv'][index].cpu()
                # import pdb
                # pdb.set_trace()
                try:
                    ptv_map_val = data_dict.get('ptv_channel_map', None)
                    if isinstance(ptv_map_val, list) and len(ptv_map_val) > index and isinstance(ptv_map_val[index], dict):
                        ptv_map = ptv_map_val[index]
                    elif isinstance(ptv_map_val, dict):
                        ph = ptv_map_val.get('PTVHigh', 0)
                        pm = ptv_map_val.get('PTVMid', 1)
                        pl = ptv_map_val.get('PTVLow', 2)
                        if isinstance(ph, (list, tuple, torch.Tensor)):
                            ph = int(ph[index])
                        if isinstance(pm, (list, tuple, torch.Tensor)):
                            pm = int(pm[index])
                        if isinstance(pl, (list, tuple, torch.Tensor)):
                            pl = int(pl[index])
                        ptv_map = {'PTVHigh': int(ph), 'PTVMid': int(pm), 'PTVLow': int(pl)}
                    else:
                        ptv_map = {'PTVHigh': 0, 'PTVMid': 1, 'PTVLow': 2}

                    idx_h = int(ptv_map.get('PTVHigh', 0))
                    idx_m = int(ptv_map.get('PTVMid', 1))
                    idx_l = int(ptv_map.get('PTVLow', 2))

                    if 0 <= idx_h < cat_ptv.shape[0]:
                        ptv_high_mask = cropped2ori(cat_ptv[idx_h], ori_size, isocenter, in_size) > 0
                    if 0 <= idx_m < cat_ptv.shape[0]:
                        ptv_mid_mask = cropped2ori(cat_ptv[idx_m], ori_size, isocenter, in_size) > 0
                    if 0 <= idx_l < cat_ptv.shape[0]:
                        ptv_low_mask = cropped2ori(cat_ptv[idx_l], ori_size, isocenter, in_size) > 0
                except Exception as e:
                    print(f"[ERROR] {patient_id}: PTV mask reconstruction failed: {e}.")
                    ptv_high_mask = None

                # Prescribed doses from loader (preferred), fallback to scale_dose_dict
                prs_gy = None
                if 'ptv_prs' in data_dict:
                    prs = data_dict['ptv_prs'][index].cpu().numpy().tolist()
                    prs_gy = [p * cfig['loader_params']['dose_div_factor'] for p in prs]
                else:
                    ptv_high_dose = float(ptv_info['PDose']) if ptv_info and 'PDose' in ptv_info else np.nan
                    prs_gy = [ptv_high_dose]

                # High-dose PTV metrics are computed via METRIC_PLAN; avoid duplicate specialized computation.

                # OAR metrics according to site-specific definitions
                # Exact ROI/metric plan replication
                plan_list = METRIC_PLAN[site_name]
                # Prepare PDose dictionary for PTVs (from loader or fallback)
                ptv_mid_info = scale_dose_dict.get(pid_only, {}).get('PTV_Mid', {})
                ptv_low_info = scale_dose_dict.get(pid_only, {}).get('PTV_Low', {})
                pdose_map = {
                    'PTVHighOPT': (prs_gy[0] if prs_gy and len(prs_gy) > 0 else (float(ptv_info.get('PDose', np.nan)) if ptv_info else np.nan)),
                    'PTV': (prs_gy[0] if prs_gy and len(prs_gy) > 0 else (float(ptv_info.get('PDose', np.nan)) if ptv_info else np.nan)),
                    ptv_high_name: (prs_gy[0] if prs_gy and len(prs_gy) > 0 else (float(ptv_info.get('PDose', np.nan)) if ptv_info else np.nan)),
                    ptv_mid_info.get('OPTName', 'PTVMidOPT'): (prs_gy[1] if prs_gy and len(prs_gy) > 1 else float(ptv_mid_info.get('PDose', np.nan))),
                    ptv_low_info.get('OPTName', 'PTVLowOPT'): (prs_gy[2] if prs_gy and len(prs_gy) > 2 else float(ptv_low_info.get('PDose', np.nan))),
                }

                # OAR masks from loader
                # Prefer OAR channel map from loader; fallback to built-in dicts
                oar_map_val = data_dict.get('oar_channel_map', None)
                if isinstance(oar_map_val, list) and len(oar_map_val) > index and isinstance(oar_map_val[index], dict):
                    oar_dict = oar_map_val[index]
                elif isinstance(oar_map_val, dict):
                    oar_dict = oar_map_val
                else:
                    oar_dict = HaN_OAR_DICT if site_name == 'Head and Neck' else Lung_OAR_DICT
                cat_oar_masks = None
                if 'cat_oar' in data_dict:
                    cat_oar = data_dict['cat_oar'][index].cpu()
                    if list(cat_oar.shape[-3:]) != list(in_size):
                        cat_oar = torch.nn.functional.interpolate(cat_oar[None], size=in_size, mode='nearest')[0]
                    cat_oar_masks = []
                    for ch in range(cat_oar.shape[0]):
                        mask_ori = cropped2ori(cat_oar[ch].cpu().numpy(), ori_size, isocenter, in_size) > 0.5
                        cat_oar_masks.append(mask_ori)

                for roi_name, metric_label in plan_list:
                    # Build mask from loader channels
                    mask = None
                    if roi_name.lower().startswith('ptv'):
                        ptv_idx = None
                        if roi_name.lower().startswith('ptvhigh'):
                            ptv_idx = 0
                            if ptv_high_mask is not None:
                                mask = ptv_high_mask
                        elif roi_name.lower().startswith('ptvmid'):
                            ptv_idx = 1
                            if ptv_mid_mask is not None:
                                mask = ptv_mid_mask
                        elif roi_name.lower().startswith('ptvlow'):
                            ptv_idx = 2
                            if ptv_low_mask is not None:
                                mask = ptv_low_mask
                        elif roi_name.lower() == 'ptv':
                            ptv_idx = 0
                            if ptv_high_mask is not None:
                                mask = ptv_high_mask
                    else:
                        idx = oar_dict.get(roi_name, None)
                        if idx is not None and cat_oar_masks is not None and 0 <= idx-1 < len(cat_oar_masks):
                            mask = cat_oar_masks[idx-1]
                    # warn if required ROI is missing
                    if mask is None:
                        print(f"[WARN] {patient_id}: Missing ROI '{roi_name}' for site '{site_name}'.")
                        continue
                    # if mask exists but is empty within body region, warn and skip
                    if not np.any(mask & body_mask):
                        print(f"[WARN] {patient_id}: ROI '{roi_name}' has empty mask in body region; skipping metric '{metric_label}'.")
                        continue

                    pred_vals = pred_gy[mask & body_mask]
                    gt_vals = gt_gy[mask & body_mask] if gt_gy is not None else np.array([])
                    pdose = pdose_map.get(roi_name, pdose_map.get(ptv_high_name, np.nan))
                    pred_metric = compute_metric_from_label(metric_label, pred_vals, voxel_vol_mm3, pdose=pdose)
                    gt_metric = compute_metric_from_label(metric_label, gt_vals, voxel_vol_mm3, pdose=pdose)
                    if np.isfinite(pred_metric) and np.isfinite(gt_metric):
                        add_err(site_name, f'{roi_name}_{metric_label}', pred_metric - gt_metric, agg)

                # Note: Lung V5Gy/V20Gy already computed via METRIC_PLAN above; avoid duplicate computation.

    rows_han = collect_site_rows(agg, 'Head and Neck')
    rows_lung = collect_site_rows(agg, 'Lung')
    print_table('Head and Neck', rows_han, out_fmt=args.out_fmt, out_dir=args.out_dir)
    print_table('Lung', rows_lung, out_fmt=args.out_fmt, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
