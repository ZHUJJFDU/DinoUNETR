"""Compatibility entrypoint for the former distance-model inference script."""

from infer_dosedino import SimpleDataset, check_list_str, infer_dosedino

inference_distance_simple = infer_dosedino

__all__ = ["SimpleDataset", "check_list_str", "infer_dosedino", "inference_distance_simple"]


if __name__ == "__main__":
    infer_dosedino()
