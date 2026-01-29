import os
import sys
import torch
import torch.nn as nn

# Ensure we can import from the local dinov3 package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dinov3.models.vision_transformer import vit_base

# --- Backbone Adapter (Reused/Modified) ---
class MedDINOv3Backbone(nn.Module):
    def __init__(self, checkpoint_path, input_dim=3):
        super().__init__()
        # 1. Initialize model with MedDINOv3 specific parameters
        self.model = vit_base(
            img_size=256, 
            patch_size=16,
            drop_path_rate=0.2, 
            layerscale_init=1.0e-05, 
            n_storage_tokens=4, 
            qkv_bias=False, 
            mask_k_bias=True
        )
        
        # 2. Load weights
        self._load_weights(checkpoint_path)

        # 3. Adapt channels if necessary
        if input_dim != 3:
            self.adapt_channels(input_dim)

        self.out_indices = [2, 5, 8, 11] # 0-indexed layers 3, 6, 9, 12

    def _load_weights(self, checkpoint_path):
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}...")
            try:
                chkpt = torch.load(checkpoint_path, map_location='cpu')
                if 'teacher' in chkpt:
                    state_dict = chkpt['teacher']
                else:
                    state_dict = chkpt
                
                # Filter out incompatible keys (like ibot head) and remove prefixes
                new_state_dict = {}
                for k, v in state_dict.items():
                    if 'ibot' in k or 'dino_head' in k:
                        continue
                    # Remove 'backbone.' prefix if present
                    new_key = k.replace('backbone.', '')
                    new_state_dict[new_key] = v
                
                # Load weights (strict=False to allow for some mismatch, e.g. pos_embed resizing if needed)
                missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
                print(f"Weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            except Exception as e:
                print(f"Error loading weights: {e}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Using random initialization.")

    def adapt_channels(self, input_dim):
        patch_embed_layer = self.model.patch_embed.proj
        
        original_weights = patch_embed_layer.weight.data
        original_bias = patch_embed_layer.bias
        
        new_in_channels = input_dim
        embed_dim, _, k_h, k_w = original_weights.shape
        
        new_layer = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=embed_dim,
            kernel_size=(k_h, k_w),
            stride=patch_embed_layer.stride,
            padding=patch_embed_layer.padding
        )
        
        with torch.no_grad():
            # Copy first 3 channels
            new_layer.weight[:, :3, :, :] = original_weights[:, :3, :, :]
            # Zero init remaining channels
            if new_in_channels > 3:
                new_layer.weight[:, 3:, :, :] = 0
            # Copy bias
            if original_bias is not None:
                new_layer.bias.data.copy_(original_bias.data)
                
        self.model.patch_embed.proj = new_layer

    def forward(self, x):
        # Local DINOv3 implementation returns (B, C, H, W) when reshape=True
        features = self.model.get_intermediate_layers(
            x, 
            n=self.out_indices, 
            reshape=True
        )
        return features

# --- Geometry Encoder ---
class GeometryEncoder(nn.Module):
    """
    A lightweight CNN encoder for distance/geometric maps.
    Downsamples from (B, 3, H, W) to (B, embed_dim, H/16, W/16).
    """
    def __init__(self, input_dim=3, embed_dim=768):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),  # H/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # H/4 - Skip maxpool to keep more info early on? Or stick to standard
        )
        # Assuming H/2 after stem. We need to reach H/16. So 3 more downsamples.
        
        # Layer 1: H/2 -> H/4
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: H/4 -> H/8
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Layer 3: H/8 -> H/16
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# --- Cross Attention Fusion Block ---
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_main, x_geo):
        """
        x_main: DINO features (B, C, H, W)
        x_geo: Geometry features (B, C, H, W)
        """
        B, C, H, W = x_main.shape
        # Flatten spatial dimensions for attention
        # (B, H*W, C)
        q = self.norm1(x_main.flatten(2).transpose(1, 2))
        k = self.norm2(x_geo.flatten(2).transpose(1, 2))
        v = self.norm2(x_geo.flatten(2).transpose(1, 2))
        
        q = self.to_q(q).reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.to_k(k).reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.to_v(v).reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Residual connection + Reshape back to CHW
        out = x.transpose(1, 2).reshape(B, C, H, W)
        return x_main + out

# --- Decoder Blocks (Same as original) ---
class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, 2, 2, 0, output_padding=0)
    def forward(self, x):
        return self.block(x)

class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size, 1, (kernel_size - 1) // 2)
    def forward(self, x):
        return self.block(x)

class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.block(x)

class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            SingleConv2DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.block(x)
class nmODEFunc(nn.Module):
    def __init__(self):
        super(nmODEFunc, self).__init__()
        self.gamma = None # 用于存储外部驱动 F(x)
        
    def fresh(self, gamma):
        """注入外部驱动力 F(x)"""
        self.gamma = gamma
        
    def forward(self, t, p):
        # 论文公式: dy/dt = -y + sin^2(y + F(x))
        dpdt = -p + torch.sin(p + self.gamma)**2
        return dpdt

class nmODEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.odefunc = nmODEFunc()
        
        self.drive_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        self.out_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # 1. 计算外部驱动 F(x)
        drive = self.drive_conv(x)
        
        # 2. 注入驱动力到 ODE 函数中
        self.odefunc.fresh(drive)
        
        # 3. 定义初始状态 y(0)
        y0 = torch.zeros_like(x)
        
        # 4. 求解 ODE (积分时间 0 -> 1)
        times = torch.tensor([0, 1.0]).type_as(x)
        out = odeint(self.odefunc, y0, times, method='rk4')[1]
        
        # 5. 输出变换
        return self.out_conv(out)

# --- Main Model ---
class MED_DINO_UNETR_Distance(nn.Module):
    def __init__(self, checkpoint_path, embed_dim=768, input_dim=6, output_dim=1):
        super().__init__()
        
        print(f"Initializing MED_DINO_UNETR_Distance...")
        # 1. Main Backbone (DINOv3) - takes 5 channels
        # (Mass Density, PTV, OAR Priority, Beam Plate, Body)
        # Excluding Distance (index 4)
        self.backbone = MedDINOv3Backbone(checkpoint_path, input_dim=5)
        
        # 2. Geometry Encoder - takes 1 channel (Distance only)
        # Using a simple CNN that outputs same dim as DINO
        self.geo_encoder = GeometryEncoder(input_dim=1, embed_dim=embed_dim)
        
        # 3. Fusion Layer (Cross Attention)
        self.fusion_layer = CrossAttentionFusion(dim=embed_dim)

        # 4. U-Net Decoder
        # Initial convolution for the raw input skip connection. 
        # Using the full 6-channel input here for low-level details.
        # We enforce 6 channels here because the forward pass sends 'x' (6 channels) to decoder0.
        decoder_in_channels = 6
        self.decoder0 = nn.Sequential(
            Conv2DBlock(decoder_in_channels, 32, 3), 
            Conv2DBlock(32, 64, 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv2DBlock(embed_dim, 512),
            Deconv2DBlock(512, 256),
            Deconv2DBlock(256, 128),
        )

        self.decoder6 = nn.Sequential(
            Deconv2DBlock(embed_dim, 512),
            Deconv2DBlock(512, 256),
        )

        self.decoder9 = Deconv2DBlock(embed_dim, 512)

        self.decoder12_upsampler = SingleDeconv2DBlock(embed_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv2DBlock(1024, 512),
            Conv2DBlock(512, 512),
            Conv2DBlock(512, 512),
            SingleDeconv2DBlock(512, 256),
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv2DBlock(512, 256), Conv2DBlock(256, 256), SingleDeconv2DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(256, 128), Conv2DBlock(128, 128), SingleDeconv2DBlock(128, 64)
        )

        self.head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1), 
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=False),
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1, inplace=False),
            
            # 在 16 通道上进行非线性动力学修正
            nmODEBlock(channels=16), 
            
            # 最终输出层 (16 -> output_dim)
            nn.Conv2d(16, output_dim, kernel_size=1),
            nn.Softplus() # 保持 Softplus 因为剂量必须 > 0
        )
    
    def forward(self, x):
        # x: (B, 6, H, W)
        # Channels:
        # 0: mass_density
        # 1: comb_optptv
        # 2: comb_oar_priority
        # 3: beam_plate_norm
        # 4: comb_oar_distance
        # 5: Body
        
        # Split input
        # Path 1: DINO Encoder (5 channels) -> Indices 0, 1, 2, 3, 5
        # We concatenate non-distance channels
        x_img = torch.cat([x[:, :4, :, :], x[:, 5:, :, :]], dim=1)
        
        # Path 2: Geometry Encoder (1 channel) -> Index 4 (Distance)
        x_geo = x[:, 4:5, :, :]

        # 1. Forward Pass DINO
        features = self.backbone(x_img)
        # features[0]: layer 3, [1]: layer 6, [2]: layer 9, [3]: layer 12
        f3, f6, f9, f12 = features[0], features[1], features[2], features[3]

        # 2. Forward Pass Geometry Encoder
        geo_feat = self.geo_encoder(x_geo) # Should be same shape as f12: (B, 768, H/16, W/16)

        # 3. Fusion at Layer 12
        # Enhance DINO features with Geometry features
        f12_fused = self.fusion_layer(f12, geo_feat)

        # 4. Decoder
        # Use fused features for f12
        
        # f0: Skip connection from raw input. Using the full 6-channel input here for low-level details
        f0 = self.decoder0(x) 
        
        f12_up = self.decoder12_upsampler(f12_fused)
        
        f9 = self.decoder9(f9)
        f9 = self.decoder9_upsampler(torch.cat([f9, f12_up], dim=1))
        
        f6 = self.decoder6(f6)
        f6 = self.decoder6_upsampler(torch.cat([f6, f9], dim=1))
        
        f3 = self.decoder3(f3)
        f3 = self.decoder3_upsampler(torch.cat([f3, f6], dim=1))
        
        output = self.head(torch.cat([f0, f3], dim=1))
        
        return output, f6, f12_fused

if __name__ == '__main__':
    # Define checkpoint path
    # Using a dummy path or local path if available
    ckpt_path = r"c:\Users\960\Desktop\DinoUNETR\dino_unetr\model.pth"
    
    print(f"Initializing MED_DINO_UNETR_Distance...")
    model = MED_DINO_UNETR_Distance(checkpoint_path=ckpt_path)
    
    # Test Forward Pass
    print("\nTesting forward pass...")
    # B, C, H, W
    input_tensor = torch.randn(1, 6, 256, 256) 
    try:
        output_tensor,_,_ = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_tensor.shape}")
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
