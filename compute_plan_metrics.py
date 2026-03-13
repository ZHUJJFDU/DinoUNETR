import argparse
import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def normalize_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("/", "")
    )


def find_roi_key(data_dict: Dict, roi_name: str, alias_map: Dict[str, List[str]]) -> Optional[str]:
    candidates = [roi_name] + alias_map.get(roi_name, [])
    normalized_map = {normalize_name(k): k for k in data_dict.keys()}
    for candidate in candidates:
        normalized = normalize_name(candidate)
        if normalized in normalized_map:
            return normalized_map[normalized]
    return None


def get_voxel_volume_cc(data_dict: Dict) -> float:
    spacing = data_dict.get("spacing")
    if spacing is None:
        spacing = data_dict.get("pixel_spacing")
    if spacing is None:
        return 1.0
    spacing = np.array(spacing, dtype=float)
    if spacing.size == 3:
        return float(np.prod(spacing)) / 1000.0
    return 1.0


def scale_prediction_to_ref(prediction: np.ndarray, ptv_mask: np.ndarray, ref_dose: Optional[float]) -> np.ndarray:
    if ref_dose is None:
        return prediction
    if np.max(ptv_mask) <= 0:
        return prediction
    ptv_dose = prediction[ptv_mask > 0]
    if ptv_dose.size == 0:
        return prediction
    scale = ref_dose / (np.percentile(ptv_dose, 3) + 1e-8)
    return prediction * scale


def dose_at_volume(dose_values: np.ndarray, volume_percent: float) -> float:
    percentile = 100.0 - volume_percent
    return float(np.percentile(dose_values, percentile))


def volume_at_dose(dose_values: np.ndarray, dose_threshold: float) -> float:
    return float(np.mean(dose_values >= dose_threshold) * 100.0)


def dose_at_cc(dose_values: np.ndarray, voxel_cc: float, cc: float) -> float:
    if voxel_cc <= 0:
        return float(np.max(dose_values))
    voxels = int(np.ceil(cc / voxel_cc))
    if voxels <= 0:
        return float(np.max(dose_values))
    sorted_vals = np.sort(dose_values)[::-1]
    voxels = min(voxels, sorted_vals.size)
    return float(sorted_vals[voxels - 1])


def compute_hi(dose_values: np.ndarray) -> float:
    d2 = np.percentile(dose_values, 98)
    d98 = np.percentile(dose_values, 2)
    d50 = np.percentile(dose_values, 50)
    if d50 == 0:
        return 0.0
    return float((d2 - d98) / d50)


def compute_ci(dose_array: np.ndarray, ptv_mask: np.ndarray, rx_dose: float, body_mask: Optional[np.ndarray]) -> float:
    if rx_dose is None:
        return 0.0
    isodose_mask = dose_array >= rx_dose
    if body_mask is not None:
        isodose_mask = isodose_mask & (body_mask > 0)
    ptv_mask_bool = ptv_mask > 0
    if np.sum(ptv_mask_bool) == 0:
        return 0.0
    v_ptv = float(np.sum(ptv_mask_bool))
    v_ref = float(np.sum(isodose_mask))
    v_intersect = float(np.sum(isodose_mask & ptv_mask_bool))
    if v_ref == 0:
        return 0.0
    return float((v_intersect / v_ptv) * (v_intersect / v_ref))


def compute_gi(dose_array: np.ndarray, rx_dose: float, body_mask: Optional[np.ndarray]) -> float:
    if rx_dose is None:
        return 0.0
    dose_100 = dose_array >= rx_dose
    dose_50 = dose_array >= (0.5 * rx_dose)
    if body_mask is not None:
        dose_100 = dose_100 & (body_mask > 0)
        dose_50 = dose_50 & (body_mask > 0)
    v100 = float(np.sum(dose_100))
    v50 = float(np.sum(dose_50))
    if v100 == 0:
        return 0.0
    return float(v50 / v100)


def load_score_table(site: int, base_dir: str = "meta_files") -> pd.DataFrame:
    if site == 1:
        score_path = os.path.join(base_dir, "HaN_DVH_Score.csv")
    else:
        score_path = os.path.join(base_dir, "LUNG_DVH_Score.csv")
    return pd.read_csv(score_path)


def parse_para_num(para_num: str) -> Tuple[str, Optional[float]]:
    value = para_num.strip()
    if value in {"", "Mean", "Max"}:
        return value, None
    if value.endswith("%"):
        return "percent", float(value.replace("%", ""))
    if value.endswith("Gy"):
        return "dose", float(value.replace("Gy", ""))
    if value.endswith("cc"):
        return "cc", float(value.replace("cc", ""))
    return value, None


def compute_metrics_for_case(
    plan_id: str,
    data_path: str,
    pred_path: str,
    ref_ptv_name: Optional[str],
    ref_dose: Optional[float],
    score_table: pd.DataFrame,
) -> List[Dict[str, object]]:
    data_npz = np.load(data_path, allow_pickle=True)
    data_dict = dict(data_npz)["arr_0"].item()
    prediction = np.load(pred_path)

    roi_aliases = {
        "PTV": ["PTVHighOPT", "PTVMidOPT", "PTVLowOPT"],
        "Total Lung": ["Total Lung-GTV", "TotalLungGTV"],
        "Spinal Cord": ["SpinalCord", "SpinalCord_05"],
        "cord+5": ["SpinalCord_05", "SpinalCord"],
        "Body-PTV": ["Body", "BodyPTV"],
        "Skin": ["Body", "Skin"],
        "Parotids-PTV": ["Parotids", "ParotidsPTV"],
        "Larynx-PTV": ["Larynx", "LarynxPTV"],
        "Mandible-PTV": ["Mandible", "MandiblePTV"],
    }

    if ref_ptv_name is None:
        ref_ptv_name = "PTVHighOPT" if "PTVHighOPT" in data_dict else "PTV"

    ptv_key = find_roi_key(data_dict, ref_ptv_name, roi_aliases) or ref_ptv_name
    if ptv_key in data_dict:
        ptv_mask = data_dict[ptv_key] > 0
    else:
        ptv_mask = None

    if ptv_key in {"PTVHighOPT", "PTVLowOPT", "PTVMidOPT"}:
        ptv_union = np.zeros_like(prediction, dtype=bool)
        for name in ["PTVHighOPT", "PTVLowOPT", "PTVMidOPT"]:
            if name in data_dict:
                ptv_union |= data_dict[name] > 0
        if np.any(ptv_union):
            ptv_mask = ptv_union

    if ptv_mask is not None:
        prediction = scale_prediction_to_ref(prediction, ptv_mask, ref_dose)

    voxel_cc = get_voxel_volume_cc(data_dict)
    body_mask = data_dict.get("Body")

    rows: List[Dict[str, object]] = []

    for _, item in score_table.iterrows():
        para_item = str(item["Para_item"]).strip()
        para_num = str(item["Para_num"]).strip()
        roi = str(item["ROI"]).strip()
        matched_roi = str(item.get("Matched_ROI", "")).strip()

        roi_key = find_roi_key(data_dict, roi, roi_aliases)
        if roi_key is None and matched_roi and matched_roi != "nan":
            roi_key = find_roi_key(data_dict, matched_roi, roi_aliases)

        if roi == "Body-PTV" and roi_key == "Body" and ptv_mask is not None:
            roi_mask = (data_dict[roi_key] > 0) & (~ptv_mask)
        elif roi_key is not None:
            roi_mask = data_dict[roi_key] > 0
        else:
            roi_mask = None

        if roi_mask is not None and np.sum(roi_mask) == 0:
            roi_mask = None

        value = None
        if para_item in ["D", "V", "Mean Dose", "Max Dose", "HI"] and roi_mask is None:
            value = None
        else:
            if para_item == "D":
                para_kind, para_value = parse_para_num(para_num)
                if para_kind == "percent" and para_value is not None:
                    value = dose_at_volume(prediction[roi_mask], para_value)
                elif para_kind == "cc" and para_value is not None:
                    value = dose_at_cc(prediction[roi_mask], voxel_cc, para_value)
                elif para_num == "Mean":
                    value = float(np.mean(prediction[roi_mask]))
                elif para_num == "Max":
                    value = float(np.max(prediction[roi_mask]))
            elif para_item == "V":
                para_kind, para_value = parse_para_num(para_num)
                if para_kind == "dose" and para_value is not None:
                    value = volume_at_dose(prediction[roi_mask], para_value)
            elif para_item == "Mean Dose":
                value = float(np.mean(prediction[roi_mask]))
            elif para_item == "Max Dose":
                value = float(np.max(prediction[roi_mask]))
            elif para_item == "HI":
                value = compute_hi(prediction[roi_mask])
            elif para_item == "CI":
                para_kind, para_value = parse_para_num(para_num)
                if para_kind == "dose" and para_value is not None and ptv_mask is not None:
                    value = compute_ci(prediction, ptv_mask, para_value, body_mask)
            elif para_item == "GI":
                rx_dose = ref_dose
                value = compute_gi(prediction, rx_dose, body_mask)

        rows.append(
            {
                "PlanID": plan_id,
                "Para_item": para_item,
                "Para_num": para_num,
                "ROI": roi,
                "Matched_ROI": roi_key,
                "Value": value,
            }
            
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Compute dose evaluation metrics for predicted plans.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/result_default_256_meddino_nah&lung",
        help="Directory containing prediction .npy files.",
    )
    parser.add_argument("--meta_file", type=str, default="meta_files/meta_data_infer_val.csv", help="Path to meta_data.csv.")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path. Defaults to results_dir/metrics.csv")
    parser.add_argument("--ref_ptv_name", type=str, default=None, help="Override reference PTV name.")
    parser.add_argument("--ref_dose", type=float, default=None, help="Override prescribed dose.")
    parser.add_argument("--score_dir", type=str, default="meta_files", help="Directory containing DVH score tables.")

    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = os.path.join(args.results_dir, "metrics.csv")

    df = pd.read_csv(args.meta_file)
    df["plan_id"] = df["npz_path"].apply(lambda p: os.path.basename(str(p)).replace(".npz", ""))
    meta_map = df.set_index("plan_id")["npz_path"].to_dict()
    dose_map = df.set_index("plan_id")["HighDose"].to_dict()
    site_map = df.set_index("plan_id")["site"].to_dict()

    pred_files = [f for f in os.listdir(args.results_dir) if f.endswith("_pred.npy")]
    all_rows: List[Dict[str, object]] = []

    for pred_file in pred_files:
        plan_id = pred_file.replace("_pred.npy", "")
        if plan_id not in meta_map:
            continue
        data_path = meta_map[plan_id]
        if not os.path.exists(data_path):
            continue

        site = site_map.get(plan_id, 1)
        ref_ptv_name = args.ref_ptv_name
        ref_dose = args.ref_dose
        if ref_ptv_name is None:
            ref_ptv_name = "PTVHighOPT" if site == 1 else "PTV"
        if ref_dose is None:
            ref_dose = float(dose_map.get(plan_id)) if plan_id in dose_map else None

        score_table = load_score_table(site, args.score_dir)
        pred_path = os.path.join(args.results_dir, pred_file)
        rows = compute_metrics_for_case(plan_id, data_path, pred_path, ref_ptv_name, ref_dose, score_table)
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
