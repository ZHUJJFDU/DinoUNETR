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

def inject_lora(model, rank=8, alpha=8):
    blocks = model.backbone.model.blocks 
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

# ================= Conv-Adapter 相关 =================
class ConvAdapter(nn.Module):
    """
    卷积适配器：将 Transformer 序列数据还原为图像空间进行卷积处理
    用于捕捉剂量预测所需的空间/边界信息
    """
    def __init__(self, in_features, bottleneck_dim=64, kernel_size=3):
        """
        Args:
            in_features: 输入维度 (例如 768)
            bottleneck_dim: 瓶颈层维度 (越小参数越少，建议 64 或 128)
            kernel_size: 卷积核大小 (建议 3x3 以捕捉局部边界)
        """
        super().__init__()
        
        # 1. 降维 (1x1 Conv): 压缩特征，减少参数量
        self.down_proj = nn.Conv2d(in_features, bottleneck_dim, kernel_size=1)
        
        # 2. 空间卷积 (3x3 Conv): 提取剂量预测所需的空间/边界信息
        # padding=kernel_size//2 保证尺寸不变
        self.spatial_conv = nn.Conv2d(
            bottleneck_dim, bottleneck_dim, 
            kernel_size=kernel_size, 
            padding=kernel_size//2,
            groups=1  # 如果想更省参数，可以设为 bottleneck_dim (Depthwise Conv)
        )
        
        # 3. 激活函数
        self.act = nn.GELU()
        
        # 4. 升维 (1x1 Conv): 还原特征维度
        self.up_proj = nn.Conv2d(bottleneck_dim, in_features, kernel_size=1)
        
        # 初始化技巧：将升维层初始化为 0，保证训练初始阶段不影响原模型
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        """
        Args:
            x: (Batch, N_tokens, Dim) 
               N_tokens 可能包含: [CLS] + [Register Tokens] + [Image Patches]
        Returns:
            y: (Batch, N_tokens, Dim) 与输入相同形状
        """
        B, N, C = x.shape
        
        # --- 检测特殊 tokens (CLS + Register Tokens) ---
        # MedDINOv3 使用: 1 CLS token + 4 register tokens + H*W image patches
        # 尝试不同的配置来找到正确的 H, W
        
        num_special_tokens = 0
        H = W = 0
        
        # 尝试不同数量的特殊 tokens
        for n_special in [0, 1, 5]:  # 0: 无特殊token, 1: 只有CLS, 5: CLS+4个register
            n_img = N - n_special
            h = int(math.sqrt(n_img))
            if h * h == n_img:  # 找到完全平方数
                num_special_tokens = n_special
                H = W = h
                break
        
        if H == 0:
            # 如果找不到完全平方数，报错并提供调试信息
            raise ValueError(
                f"无法推断图像尺寸！输入 tokens 数量: {N}, "
                f"尝试过的配置均不是完全平方数。"
                f"可能的图像 patches 数量: {[N, N-1, N-5]}"
            )
        
        # --- 分离特殊 tokens 和图像 patches ---
        if num_special_tokens > 0:
            special_tokens = x[:, :num_special_tokens, :]  # CLS + Register tokens
            img_tokens = x[:, num_special_tokens:, :]      # Image patches
        else:
            special_tokens = None
            img_tokens = x
            
        # --- 序列转图像 (Reshape) ---
        # (B, N_img, C) -> (B, C, N_img) -> (B, C, H, W)
        x_img = img_tokens.transpose(1, 2).view(B, C, H, W)
        
        # --- 卷积处理 ---
        y = self.down_proj(x_img)
        y = self.act(y)
        y = self.spatial_conv(y)  # 关键步骤：捕捉空间结构
        y = self.act(y)
        y = self.up_proj(y)
        
        # --- 图像转序列 (Flatten) ---
        # (B, C, H, W) -> (B, C, N_img) -> (B, N_img, C)
        y = y.flatten(2).transpose(1, 2)
        
        # --- 还原特殊 tokens ---
        if special_tokens is not None:
            # 特殊 tokens 不做卷积处理，补零或保持原样
            zero_tokens = torch.zeros_like(special_tokens)
            y = torch.cat([zero_tokens, y], dim=1)
            
        return y


def inject_conv_adapter(model, bottleneck_dim=64, kernel_size=3):
    """
    将 Conv-Adapter 注入到模型的所有 Transformer Block 中
    Adapter 与 MLP 并联，而不是替换 MLP
    
    Args:
        model: MED_DINO_UNETR 模型实例
        bottleneck_dim: Adapter 的瓶颈层维度，默认 64
        kernel_size: 卷积核大小，默认 3x3
    
    Returns:
        model: 注入 Adapter 后的模型
    """
    # 1. 冻结所有参数 (Backbone Freezing)
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. 获取 Blocks 列表
    # 根据模型结构：model.backbone.model.blocks
    blocks = model.backbone.model.blocks
    
    print(f"检测到 {len(blocks)} 个 Transformer Blocks，开始注入 Conv-Adapter...")
    print(f"Adapter 配置: bottleneck_dim={bottleneck_dim}, kernel_size={kernel_size}")
    
    total_adapter_params = 0
    
    for i, block in enumerate(blocks):
        # 获取输入维度 (从 MLP 的第一层获取)
        if hasattr(block.mlp, 'fc1'):
            input_dim = block.mlp.fc1.in_features
        else:
            input_dim = 768  # 默认
        
        # 创建 Adapter 并添加为 block 的属性
        adapter = ConvAdapter(
            in_features=input_dim, 
            bottleneck_dim=bottleneck_dim,
            kernel_size=kernel_size
        )
        
        # 将 adapter 添加为 block 的属性（不替换 mlp）
        block.adapter = adapter
        
        # 计算 Adapter 参数量
        adapter_params = sum(p.numel() for p in adapter.parameters())
        total_adapter_params += adapter_params
        
        # 确保 Adapter 是可训练的
        for param in block.adapter.parameters():
            param.requires_grad = True
        
        # 保存原始的 _forward 和 _forward_list 方法
        original_forward = block._forward
        original_forward_list = block._forward_list
        
        # 定义新的 _forward 方法（单 tensor 版本）
        def new_forward(self, x, rope=None, _original_forward=original_forward):
            """Modified forward with adapter in parallel"""
            b, _, _ = x.shape
            sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
            residual_scale_factor = b / sample_subset_size

            if self.training and self.sample_drop_ratio > 0.0:
                indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]
                x_subset_1 = x[indices_1]
                rope_subset = self._maybe_index_rope(rope, indices_1)
                residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

                x_attn = torch.index_add(
                    x, dim=0, source=self.ls1(residual_1),
                    index=indices_1, alpha=residual_scale_factor
                )

                indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]
                x_subset_2 = x_attn[indices_2]
                norm2_subset = self.norm2(x_subset_2)
                
                # MLP + Adapter 并联
                residual_2 = self.mlp(norm2_subset) + self.adapter(norm2_subset)

                x_ffn = torch.index_add(
                    x_attn, dim=0, source=self.ls2(residual_2),
                    index=indices_2, alpha=residual_scale_factor
                )
            else:
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                norm2_out = self.norm2(x_attn)
                
                # MLP + Adapter 并联
                x_ffn = x_attn + self.ls2(self.mlp(norm2_out) + self.adapter(norm2_out))

            return x_ffn
        
        # 定义新的 _forward_list 方法（list 版本）
        def new_forward_list(self, x_list, rope_list=None, _original_forward_list=original_forward_list):
            """Modified forward_list with adapter in parallel"""
            b_list = [x.shape[0] for x in x_list]
            sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
            residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

            if self.training and self.sample_drop_ratio > 0.0:
                # 注意力部分（保持不变）
                indices_1_list = [
                    (torch.randperm(b, device=x.device))[:sample_subset_size]
                    for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
                ]
                x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

                if rope_list is not None:
                    rope_subset_list = [
                        self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                    ]
                else:
                    rope_subset_list = rope_list

                from dinov3.utils import cat_keep_shapes, uncat_with_shapes
                flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
                norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
                residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

                x_attn_list = [
                    torch.index_add(x, dim=0, source=self.ls1(residual_1),
                                  index=indices_1, alpha=residual_scale_factor)
                    for x, residual_1, indices_1, residual_scale_factor in zip(
                        x_list, residual_1_list, indices_1_list, residual_scale_factors
                    )
                ]

                # MLP + Adapter 部分
                indices_2_list = [
                    (torch.randperm(b, device=x.device))[:sample_subset_size]
                    for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
                ]
                x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
                flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
                norm2_flat = self.norm2(flattened)
                norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

                # 并联：MLP + Adapter
                residual_2_mlp_list = self.mlp.forward_list(norm2_list)
                residual_2_adapter_list = [self.adapter(norm2) for norm2 in norm2_list]
                residual_2_list = [mlp + adapter for mlp, adapter in zip(residual_2_mlp_list, residual_2_adapter_list)]

                x_ffn = [
                    torch.index_add(x_attn, dim=0, source=self.ls2(residual_2),
                                  index=indices_2, alpha=residual_scale_factor)
                    for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                        x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                    )
                ]
            else:
                x_out = []
                for x, rope in zip(x_list, rope_list):
                    x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                    norm2_out = self.norm2(x_attn)
                    
                    # 并联：MLP + Adapter
                    x_ffn = x_attn + self.ls2(self.mlp(norm2_out) + self.adapter(norm2_out))
                    x_out.append(x_ffn)
                x_ffn = x_out

            return x_ffn
        
        # 使用 functools.partial 或者直接绑定方法
        import types
        block._forward = types.MethodType(new_forward, block)
        block._forward_list = types.MethodType(new_forward_list, block)
    
    # 3. 解冻 Decoder、Head 和 Patch Embedding 层
    print("解冻 Decoder、Head 和 Patch Embedding 层...")
    for name, param in model.named_parameters():
        if "decoder" in name or "head" in name or "patch_embed" in name:
            param.requires_grad = True
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"注入完成！")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - Adapter 参数量: {total_adapter_params:,}")
    print(f"  - 可训练参数量: {trainable_params:,}")
    print(f"  - 可训练参数占比: {100.0 * trainable_params / total_params:.2f}%")
    
    return model


class SequenceMDTAdapter(nn.Module):
    """
    Sequence-compatible MDT Adapter for Transformer blocks.
    Converts sequence format (B, N, C) to image format (B, C, H, W) for MDT processing,
    then converts back to sequence format.
    """
    def __init__(self, in_features, reduction=4):
        """
        Args:
            in_features: Input dimension (e.g., 768)
            reduction: Reduction factor for bottleneck (default: 4)
        """
        super().__init__()
        mid_channels = in_features // reduction
        
        # 1. Dimension Reduction
        self.down_proj = nn.Conv2d(in_features, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.GELU()
        
        # 2. Multi-scale Processing Paths
        self.conv1x1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.conv3x3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, bias=False)
        
        # 3. Fusion and Restoration
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.act2 = nn.GELU()
        self.up_proj = nn.Conv2d(mid_channels, in_features, kernel_size=1, bias=False)
        
        # 4. Spatial Attention
        self.sa = SpatialAttention()
        
        # Initialize up_proj to zero for stable training start
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        """
        Args:
            x: (Batch, N_tokens, Dim) 
               N_tokens may include: [CLS] + [Register Tokens] + [Image Patches]
        Returns:
            y: (Batch, N_tokens, Dim) same shape as input
        """
        B, N, C = x.shape
        
        # --- Detect special tokens (CLS + Register Tokens) ---
        num_special_tokens = 0
        H = W = 0
        
        # Try different numbers of special tokens
        for n_special in [0, 1, 5]:  # 0: no special tokens, 1: CLS only, 5: CLS+4 registers
            n_img = N - n_special
            h = int(math.sqrt(n_img))
            if h * h == n_img:  # Found perfect square
                num_special_tokens = n_special
                H = W = h
                break
        
        if H == 0:
            raise ValueError(
                f"Cannot infer image dimensions! Input tokens: {N}, "
                f"tried configurations but none are perfect squares. "
                f"Possible image patch counts: {[N, N-1, N-5]}"
            )
        
        # --- Separate special tokens and image patches ---
        if num_special_tokens > 0:
            special_tokens = x[:, :num_special_tokens, :]  # CLS + Register tokens
            img_tokens = x[:, num_special_tokens:, :]      # Image patches
        else:
            special_tokens = None
            img_tokens = x
            
        # --- Sequence to Image (Reshape) ---
        # (B, N_img, C) -> (B, C, N_img) -> (B, C, H, W)
        x_img = img_tokens.transpose(1, 2).view(B, C, H, W)
        
        # --- MDT Processing ---
        # Down-projection
        out = self.down_proj(x_img)
        out = self.bn1(out)
        out = self.act1(out)
        
        # Multi-scale branches
        b1 = self.conv1x1(out)
        b2 = self.conv3x3(out)
        b3 = self.conv5x5(out)
        
        # Fusion
        out = b1 + b2 + b3
        out = self.bn2(out)
        out = self.act2(out)
        
        # Up-projection
        out = self.up_proj(out)
        
        # Spatial Attention
        out = self.sa(out)
        
        # --- Image to Sequence (Flatten) ---
        # (B, C, H, W) -> (B, C, N_img) -> (B, N_img, C)
        y = out.flatten(2).transpose(1, 2)
        
        # --- Restore special tokens ---
        if special_tokens is not None:
            # Special tokens don't undergo MDT processing, pad with zeros
            zero_tokens = torch.zeros_like(special_tokens)
            y = torch.cat([zero_tokens, y], dim=1)
            
        return y


def inject_mdt_adapter(model, reduction=4):
    """
    将 MDT Adapter 注入到模型的所有 Transformer Block 中
    Adapter 与 MLP 并联，而不是替换 MLP
    
    Args:
        model: MED_DINO_UNETR 模型实例
        reduction: MDT Adapter 的降维比例，默认 4
    
    Returns:
        model: 注入 MDT Adapter 后的模型
    """
    # 1. 冻结所有参数 (Backbone Freezing)
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. 获取 Blocks 列表
    blocks = model.backbone.model.blocks
    
    print(f"检测到 {len(blocks)} 个 Transformer Blocks，开始注入 MDT Adapter...")
    print(f"MDT Adapter 配置: reduction={reduction}")
    
    total_adapter_params = 0
    
    for i, block in enumerate(blocks):
        # 获取输入维度
        if hasattr(block.mlp, 'fc1'):
            input_dim = block.mlp.fc1.in_features
        else:
            input_dim = 768  # 默认
        
        # 创建 MDT Adapter 并添加为 block 的属性
        adapter = SequenceMDTAdapter(
            in_features=input_dim, 
            reduction=reduction
        )
        
        # 将 adapter 添加为 block 的属性（不替换 mlp）
        block.mdt_adapter = adapter
        
        # 计算 Adapter 参数量
        adapter_params = sum(p.numel() for p in adapter.parameters())
        total_adapter_params += adapter_params
        
        # 确保 Adapter 是可训练的
        for param in block.mdt_adapter.parameters():
            param.requires_grad = True
        
        # 保存原始的 _forward 和 _forward_list 方法
        original_forward = block._forward
        original_forward_list = block._forward_list
        
        # 定义新的 _forward 方法（单 tensor 版本）
        def new_forward(self, x, rope=None, _original_forward=original_forward):
            """Modified forward with MDT adapter in parallel"""
            b, _, _ = x.shape
            sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
            residual_scale_factor = b / sample_subset_size

            if self.training and self.sample_drop_ratio > 0.0:
                indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]
                x_subset_1 = x[indices_1]
                rope_subset = self._maybe_index_rope(rope, indices_1)
                residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

                x_attn = torch.index_add(
                    x, dim=0, source=self.ls1(residual_1),
                    index=indices_1, alpha=residual_scale_factor
                )

                indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]
                x_subset_2 = x_attn[indices_2]
                norm2_subset = self.norm2(x_subset_2)
                
                # MLP + MDT Adapter 并联
                residual_2 = self.mlp(norm2_subset) + self.mdt_adapter(norm2_subset)

                x_ffn = torch.index_add(
                    x_attn, dim=0, source=self.ls2(residual_2),
                    index=indices_2, alpha=residual_scale_factor
                )
            else:
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                norm2_out = self.norm2(x_attn)
                
                # MLP + MDT Adapter 并联
                x_ffn = x_attn + self.ls2(self.mlp(norm2_out) + self.mdt_adapter(norm2_out))

            return x_ffn
        
        # 定义新的 _forward_list 方法（list 版本）
        def new_forward_list(self, x_list, rope_list=None, _original_forward_list=original_forward_list):
            """Modified forward_list with MDT adapter in parallel"""
            b_list = [x.shape[0] for x in x_list]
            sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
            residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

            if self.training and self.sample_drop_ratio > 0.0:
                # 注意力部分（保持不变）
                indices_1_list = [
                    (torch.randperm(b, device=x.device))[:sample_subset_size]
                    for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
                ]
                x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

                if rope_list is not None:
                    rope_subset_list = [
                        self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                    ]
                else:
                    rope_subset_list = rope_list

                from dinov3.utils import cat_keep_shapes, uncat_with_shapes
                flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
                norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
                residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

                x_attn_list = [
                    torch.index_add(x, dim=0, source=self.ls1(residual_1),
                                  index=indices_1, alpha=residual_scale_factor)
                    for x, residual_1, indices_1, residual_scale_factor in zip(
                        x_list, residual_1_list, indices_1_list, residual_scale_factors
                    )
                ]

                # MLP + MDT Adapter 部分
                indices_2_list = [
                    (torch.randperm(b, device=x.device))[:sample_subset_size]
                    for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
                ]
                x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
                flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
                norm2_flat = self.norm2(flattened)
                norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

                # 并联：MLP + MDT Adapter
                residual_2_mlp_list = self.mlp.forward_list(norm2_list)
                residual_2_adapter_list = [self.mdt_adapter(norm2) for norm2 in norm2_list]
                residual_2_list = [mlp + adapter for mlp, adapter in zip(residual_2_mlp_list, residual_2_adapter_list)]

                x_ffn = [
                    torch.index_add(x_attn, dim=0, source=self.ls2(residual_2),
                                  index=indices_2, alpha=residual_scale_factor)
                    for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                        x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                    )
                ]
            else:
                x_out = []
                for x, rope in zip(x_list, rope_list):
                    x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                    norm2_out = self.norm2(x_attn)
                    
                    # 并联：MLP + MDT Adapter
                    x_ffn = x_attn + self.ls2(self.mlp(norm2_out) + self.mdt_adapter(norm2_out))
                    x_out.append(x_ffn)
                x_ffn = x_out

            return x_ffn
        
        # 使用 types.MethodType 绑定方法
        import types
        block._forward = types.MethodType(new_forward, block)
        block._forward_list = types.MethodType(new_forward_list, block)
    
    # 3. 解冻 Decoder、Head 和 Patch Embedding 层
    print("解冻 Decoder、Head 和 Patch Embedding 层...")
    for name, param in model.named_parameters():
        if "decoder" in name or "head" in name or "patch_embed" in name:
            param.requires_grad = True
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"注入完成！")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - MDT Adapter 参数量: {total_adapter_params:,}")
    print(f"  - 可训练参数量: {trainable_params:,}")
    print(f"  - 可训练参数占比: {100.0 * trainable_params / total_params:.2f}%")
    
    return model