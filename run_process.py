import os
import yaml
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入你的 Dataset 类
# 确保 data_loader_lightning.py 在同一目录下，或者在 PYTHONPATH 中
from data_loader_lightning import MyDataset 

# --- 全局变量设置 ---
# 定义全局变量，方便线程访问，避免反复传递大对象
GLOBAL_DATASET = None
SAVE_ROOT = None
# ENABLE_CROP = False  # 开关：是否开启裁剪功能 (512 -> 4个256)
NUM_CROPS = 5 # 每张切片生成的 Crop 数量 (1个中心 + 4个随机)
CROP_SIZE = 256

def process_one_case(index):
    """
    单个线程执行的函数：处理一个病例的所有切片
    """
    try:
        # 1. 从全局 Dataset 获取数据 (IO读取)
        # 注意：这里会触发 MyDataset.__getitem__ 里的 np.load
        data_dict = GLOBAL_DATASET[index]
        
        case_id = data_dict['id']
        
        # 2. 获取 3D 数据
        # 假设 MyDataset 输出是 (C, D, H, W)
        img_3d = data_dict['data']   
        label_3d = data_dict['label']
        body_3d = data_dict['Body']
        
        # 获取 DVH Loss 所需的掩膜
        ptv_3d = data_dict['PTV']
        oar_serial_3d = data_dict['oar_serial']
        oar_parallel_3d = data_dict['oar_parallel']

        isocenter = data_dict['ori_isocenter']
        spacing = data_dict['spacing']
        angle_list = data_dict['angle_list']
        
        # 3. 确定深度
        depth = img_3d.shape[1] 
        
        # 4. 遍历切片并保存
        for d in range(depth):
            # 切片逻辑: (C, D, H, W) -> 取第 d 层 -> (C, H, W)
            # 使用 .clone() 确保内存连续性（有时候对多线程安全有帮助）
            slice_data = img_3d[:, d, :, :].clone().numpy()
            slice_label = label_3d[:, d, :, :].clone().numpy()
            slice_body = body_3d[:, d, :, :].clone().numpy()
            
            slice_ptv = ptv_3d[:, d, :, :].clone().numpy()
            slice_oar_serial = oar_serial_3d[:, d, :, :].clone().numpy()
            slice_oar_parallel = oar_parallel_3d[:, d, :, :].clone().numpy()
            
            # --- Random Crop Logic (Reproducible) ---
            # 设置随机种子，确保可复现性。使用 (case_id_hash + d) 作为种子
            # case_id 是字符串，hash 一下转为整数
            seed = (abs(hash(case_id)) + d) % (2**32)
            rng = np.random.RandomState(seed)

            _, h, w = slice_data.shape
            
            # 只有当图像尺寸 > CROP_SIZE 时才需要 Crop
            if h > CROP_SIZE and w > CROP_SIZE:
                # 1. Center Crop
                ch_start = (h - CROP_SIZE) // 2
                cw_start = (w - CROP_SIZE) // 2
                
                # 2. Random Crops
                # 生成 (NUM_CROPS - 1) 个随机坐标
                h_starts = [ch_start] # 第一个是中心
                w_starts = [cw_start]
                
                for _ in range(NUM_CROPS - 1):
                    h_starts.append(rng.randint(0, h - CROP_SIZE + 1))
                    w_starts.append(rng.randint(0, w - CROP_SIZE + 1))
                
                for i, (hs, ws) in enumerate(zip(h_starts, w_starts)):
                    he = hs + CROP_SIZE
                    we = ws + CROP_SIZE
                    
                    # Crop
                    sub_data = slice_data[:, hs:he, ws:we]
                    sub_label = slice_label[:, hs:he, ws:we]
                    sub_body = slice_body[:, hs:he, ws:we]
                    sub_ptv = slice_ptv[:, hs:he, ws:we]
                    sub_oar_serial = slice_oar_serial[:, hs:he, ws:we]
                    sub_oar_parallel = slice_oar_parallel[:, hs:he, ws:we]
                    
                    # 保存: caseID_sliceIndex_cropIndex.npz
                    save_name = f"{case_id}_{d:03d}_{i}.npz"
                    save_path = os.path.join(SAVE_ROOT, save_name)
                    
                    np.savez_compressed(save_path, 
                                        data=sub_data, 
                                        label=sub_label, 
                                        body=sub_body,
                                        ptv=sub_ptv,
                                        oar_serial=sub_oar_serial,
                                        oar_parallel=sub_oar_parallel,
                                        isocenter=isocenter,
                                        spacing=spacing,
                                        angle_list=np.array(angle_list, dtype=object)
                                        )
            else:
                # 如果图像比 Crop Size 还小（极少见），直接 Resize 或者 Padding
                # 这里简单处理：直接保存原图（或者报错，取决于你的需求）
                # 假设都大于 256，这里保留原样作为 fallback
                save_name = f"{case_id}_{d:03d}.npz"
                save_path = os.path.join(SAVE_ROOT, save_name)
                np.savez_compressed(save_path, 
                                    data=slice_data, 
                                    label=slice_label, 
                                    body=slice_body,
                                    ptv=slice_ptv,
                                    oar_serial=slice_oar_serial,
                                    oar_parallel=slice_oar_parallel,
                                    isocenter=isocenter,
                                    spacing=spacing,
                                    angle_list=np.array(angle_list, dtype=object)
                                    )
            
        return f"Success: {case_id} ({depth} slices)"
    
    except Exception as e:
        return f"Error processing index {index}: {str(e)}"

def main():
    global GLOBAL_DATASET, SAVE_ROOT
    
    # --- 1. 配置 ---
    cfig_path = 'config_files/config_DinoUnetr.yaml' # 确保路径正确
    max_workers = 8 # 线程数，建议设置为 CPU 核心数 或 稍微大一点（取决于硬盘读写速度）

    print("Loading configuration...")
    cfig = yaml.load(open(cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)

    # 定义处理任务列表： (保存路径, phase, dev_split)
    tasks = [
        ('Dataset_256_crop/Train', 'train', 'train'),
        ('Dataset_256_crop/Valid', 'train', 'valid'),
        ('Dataset_256_crop/Test',  'valid', 'test')
    ]

    for task_idx, (save_dir, phase, dev_split) in enumerate(tasks):
        print(f"\n{'='*20} Task {task_idx+1}/{len(tasks)}: Processing {phase}/{dev_split} {'='*20}")
        
        # --- 2. 初始化当前任务 ---
        SAVE_ROOT = save_dir
        os.makedirs(SAVE_ROOT, exist_ok=True)
        
        print(f"Initializing dataset for phase='{phase}', dev_split='{dev_split}'...")
        # 实例化 Dataset
        GLOBAL_DATASET = MyDataset(cfig['loader_params'], phase=phase, dev_split=dev_split)
        
        total_cases = len(GLOBAL_DATASET)
        print(f"Dataset initialized. Total volumes: {total_cases}")
        print(f"Output directory: {SAVE_ROOT}")
        print(f"Starting multi-threading processing with {max_workers} workers...")

        # --- 3. 多线程执行 ---
        # 使用 ThreadPoolExecutor 管理线程池
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            # future_to_idx 是一个字典，映射 future 对象到 原始索引
            future_to_idx = {executor.submit(process_one_case, i): i for i in range(total_cases)}
            
            # 使用 tqdm 显示进度
            # as_completed 会在某个任务完成时立刻 yield
            for future in tqdm(as_completed(future_to_idx), total=total_cases, unit="case", desc=f"Task {task_idx+1}"):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    # 如果返回结果包含 Error 字样，打印出来 (可选)
                    if "Error" in result:
                        print(f"\n[Warning] {result}")
                except Exception as exc:
                    print(f"\n[Critical Error] Case {idx} generated an exception: {exc}")

    print("\nAll processing tasks finished!")

if __name__ == '__main__':
    main()