
import torch
import pandas as pd
from dino_unetr.DINO_UNETR import DINO_UNETR
from dino_unetr.tuning_utils import inject_lora, get_llrd_params
import pytorch_lightning as pl
import torch.nn as nn

# ==========================================
# 1. 简化的 Mock 模型类 (直接引用你的类)
# ==========================================
# 为了不依赖复杂的 config 文件，这里手动定义一个最小化的 Config
dummy_cfig = {
    'lr': 1e-3,
    'num_epochs': 100,
    'scale_out': 1.0,
    'scale_loss': 1.0,
    'lora_rank': 8,
    'lora_alpha': 8
}

# 复制你最新的 GDPLightningModel 类逻辑 (简化版，只保留初始化逻辑)
class GDPLightningModel(pl.LightningModule):
    def __init__(self, cfig, strategy='default'):
        super(GDPLightningModel, self).__init__()
        self.model = DINO_UNETR()
        self.strategy = strategy
        self.cfig = cfig
        
        if self.strategy == 'lora':
            # 1. 冻结所有
            for param in self.model.parameters():
                param.requires_grad = False
            
            # 2. 注入 LoRA
            inject_lora(self.model, rank=cfig['lora_rank'], alpha=cfig['lora_alpha'])
            
            # 3. 解冻特定层
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                elif "decoder" in name or "head" in name:
                    param.requires_grad = True
                elif "patch_embed" in name:
                    param.requires_grad = True
                    
        elif self.strategy == 'llrd':
            # 全参解冻，但靠优化器分组
            for param in self.model.parameters():
                param.requires_grad = True
        
        elif self.strategy == 'frozen':
            # 冻结骨干，只训头
            for param in self.model.parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                if "decoder" in name or "head" in name:
                    param.requires_grad = True
                elif "patch_embed" in name:
                    param.requires_grad = True
            
        else: # default
            for param in self.model.parameters():
                param.requires_grad = True

# ==========================================
# 2. 统计工具函数
# ==========================================
def get_param_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    ratio = (trainable_params / total_params) * 100 if total_params > 0 else 0
    return total_params, trainable_params, frozen_params, ratio

# ==========================================
# 3. 主对比脚本
# ==========================================
def main():
    strategies = ['default', 'frozen', 'llrd', 'lora']
    results = []

    print(f"{'='*80}")
    print(f"Comparing Strategies for DINO_UNETR")
    print(f"{'='*80}")

    for strat in strategies:
        print(f"\nInitializing Strategy: [{strat.upper()}] ...")
        
        # 实例化模型
        pl_model = GDPLightningModel(dummy_cfig, strategy=strat)
        
        # 获取统计信息
        total, train, frozen, ratio = get_param_stats(pl_model.model)
        
        # 特殊检查
        note = ""
        if strat == 'llrd':
            # 检查优化器分组
            params = get_llrd_params(pl_model.model, base_lr=1e-3, decay_rate=0.75)
            # 获取不同的学习率值
            lrs = sorted(list(set([g['lr'] for g in params])))
            min_lr = min(lrs)
            max_lr = max(lrs)
            note = f"Optimizer Groups: {len(params)} | LR Range: [{min_lr:.1e} -> {max_lr:.1e}]"
            
        elif strat == 'lora':
            # 检查 LoRA 层是否存在
            has_lora = any("lora_" in n for n, _ in pl_model.model.named_parameters())
            note = f"LoRA Layers Injected: {has_lora}"

        results.append({
            "Strategy": strat,
            "Total Params (M)": total / 1e6,
            "Trainable (M)": train / 1e6,
            "Frozen (M)": frozen / 1e6,
            "Ratio (%)": ratio,
            "Note": note
        })

    # ==========================================
    # 4. 打印结果表格
    # ==========================================
    df = pd.DataFrame(results)
    
    # 格式化输出
    print("\n" + "="*100)
    print("FINAL COMPARISON RESULTS")
    print("="*100)
    
    # 手动打印漂亮的表格
    header = f"{'Strategy':<10} | {'Total (M)':<10} | {'Trainable (M)':<14} | {'Frozen (M)':<10} | {'Ratio (%)':<10} | {'Note'}"
    print(header)
    print("-" * 120)
    
    for _, row in df.iterrows():
        print(f"{row['Strategy']:<10} | "
              f"{row['Total Params (M)']:<10.2f} | "
              f"{row['Trainable (M)']:<14.2f} | "
              f"{row['Frozen (M)']:<10.2f} | "
              f"{row['Ratio (%)']:<10.2f} | "
              f"{row['Note']}")
    
    print("-" * 120)
    print("\n[Analysis]:")
    print("1. Default: 所有参数都参与训练，显存占用最大，计算量最大。")
    print("2. Frozen:  只训练 Decoder，参数量约占 10-20%，DINO 仅做特征提取。")
    print("3. LLRD:    虽然显示 100% 可训练，但底层 LR 极小(3e-5)，实现了'软冻结'，不仅保护了权重还允许微调。")
    print("4. LoRA:    可训练参数极少 (通常比 Frozen 还少，或者接近)，但在 Backbone 中插入了可训练通路，比单纯 Frozen 效果通常更好。")

if __name__ == '__main__':
    main()
