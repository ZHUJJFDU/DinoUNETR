import torch
import torch.nn as nn
import math

# ================= LoRA 相关 =================
class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=8):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.original_layer.parameters():
            param.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        return self.original_layer(x) + self.scale * self.lora_B(self.lora_A(x))

def _get_blocks(model):
    """Helper to find blocks in different model structures"""
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'model') and hasattr(model.backbone.model, 'blocks'):
        return model.backbone.model.blocks
    if hasattr(model, 'blocks'):
        return model.blocks
    raise AttributeError(f"Could not find transformer blocks in {type(model)}")

def inject_lora(model, rank=8, alpha=8):
    blocks = _get_blocks(model)
    print(f"Injecting LoRA (rank={rank}) into {len(blocks)} blocks...")
    for block in blocks:
        old_qkv = block.attn.qkv
        block.attn.qkv = LoRALinear(old_qkv, rank, alpha)
    return model

# ================= LLRD 相关 =================
def get_llrd_params(model, base_lr=1e-3, weight_decay=0.05, decay_rate=0.75):
    print(f"Applying LLRD: Base LR={base_lr}, Decay={decay_rate}")
    param_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "backbone" not in name:
            scale = 1.0 # Decoder / Head
        elif "patch_embed" in name:
            scale = 1.0 # Adaptation Layer
        elif "blocks" in name:
            block_idx = int(name.split('.')[3])
            scale = decay_rate ** (12 - block_idx)
        elif "cls_token" in name or "reg_token" in name or "pos_embed" in name:
            scale = decay_rate ** 12
        else:
            scale = decay_rate ** 1

        real_lr = base_lr * scale
        
        # Bias/Norm 不做 Weight Decay
        if "bias" in name or "norm" in name or "gamma" in name:
            this_wd = 0.0
        else:
            this_wd = weight_decay

        param_groups.append({
            "params": [param],
            "lr": real_lr,
            "weight_decay": this_wd
        })
    return param_groups
