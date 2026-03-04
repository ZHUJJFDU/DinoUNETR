import os
import sys
import torch
import torch.nn as nn

# Ensure we can import from the local dinov3 package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dinov3.models.vision_transformer import vit_base

# --- Backbone Adapter ---
class MedDINOv3Backbone(nn.Module):
    def __init__(self, checkpoint_path, input_dim=6):
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
                if len(missing) > 0:
                    print(f"First few missing: {missing[:5]}")
            except Exception as e:
                print(f"Error loading weights: {e}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Using random initialization.")

    def adapt_channels(self, input_dim):
        # print(f"Adapting first layer from 3 to {input_dim} channels...")
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

# --- Decoder Blocks ---
class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(
            in_planes, 
            out_planes, 
            kernel_size=2, 
            stride=2, 
            padding=0, 
            output_padding=0
        )

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2),
        )

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

# --- EPA Module (UNETR++) ---
class EPA(nn.Module):
    def __init__(self, input_dim, channel_attention_reduce=16):
        super().__init__()
        self.input_dim = input_dim
        self.norm = nn.GroupNorm(8, input_dim)
        # Shared 1x1 convolutions for Query and Key
        self.conv_q = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.conv_k = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        
        # Independent 1x1 convolutions for Value
        self.conv_v_spatial = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.conv_v_channel = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        
        # Channel Attention Output projection
        self.channel_out = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        
        # Spatial Attention Output projection
        self.spatial_out = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        
        # Fusion
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        # Shared Q, K
        q = self.conv_q(x) # B, C, H, W
        k = self.conv_k(x) # B, C, H, W
        
        # --- Channel Attention Branch (Q^T * K) ---
        # Reshape to (B, C, N) where N = H*W
        # Cast to float32 for numerical stability in mixed precision training
        q_c = q.view(b, c, -1).float()     # B, C, N
        k_c = k.view(b, c, -1).float()     # B, C, N
        
        # Attention map: (B, C, C)
        attn_c = torch.bmm(q_c, k_c.transpose(1, 2)) 
        # Stability: Scale and Clamp
        # 通道注意力的缩放应基于通道数 C，而不是空间分辨率 H*W
        attn_c = attn_c / (c ** 0.5 + 1e-6)
        
        attn_c = torch.softmax(attn_c, dim=-1)
        
        v_c = self.conv_v_channel(x).view(b, c, -1).float() # B, C, N
        
        # Apply attention: (B, C, C) @ (B, C, N) -> (B, C, N)
        out_c = torch.bmm(attn_c, v_c) 
        out_c = out_c.view(b, c, h, w).type_as(x)
        out_c = self.channel_out(out_c)
        
        # --- Spatial Attention Branch (Linear Efficient Attention) ---
        # Cast to float32
        q_s = q.view(b, c, -1).permute(0, 2, 1).float() # B, N, C
        k_s = k.view(b, c, -1).permute(0, 2, 1).float() # B, N, C
        v_s = self.conv_v_spatial(x).view(b, c, -1).permute(0, 2, 1).float() # B, N, C
        
        # Stability: Add eps
        q_s = torch.softmax(q_s, dim=-1) 
        k_s = torch.softmax(k_s, dim=-2) # Softmax over N (spatial)
        
        # Context: K^T (C, N) * V (N, C) -> (C, C) (Actually (B, C, C))
        context_s = torch.bmm(k_s.transpose(1, 2), v_s) 
        
        # Out: Q (N, C) * Context (C, C) -> (N, C)
        out_s = torch.bmm(q_s, context_s) 
        
        out_s = out_s.permute(0, 2, 1).contiguous().view(b, c, h, w).type_as(x)
        out_s = self.spatial_out(out_s)
        
        # Stability: Clamp the output of attention before residual addition to avoid unexpected large values
        # out_c = torch.clamp(out_c, min=-100, max=100)
        # out_s = torch.clamp(out_s, min=-100, max=100)
        
        return self.gamma * out_c + self.beta * out_s + x

# --- Main Model ---
class MED_DINO_UNETR(nn.Module):
    def __init__(self, checkpoint_path, embed_dim=768, input_dim=6, output_dim=1):
        super().__init__()
        # 1. Use the new MedDINOv3 Backbone
        self.backbone = MedDINOv3Backbone(checkpoint_path, input_dim)

        # 2. U-Net Decoder (Same as before)
        self.decoder0 = nn.Sequential(
            Conv2DBlock(input_dim, 32, 3), 
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
            EPA(512),
            Conv2DBlock(512, 512),
            Conv2DBlock(512, 512),
            SingleDeconv2DBlock(512, 256),
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv2DBlock(512, 256),
            EPA(256),
            Conv2DBlock(256, 256),
            SingleDeconv2DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(256, 128),
            EPA(128),
            Conv2DBlock(128, 128),
            SingleDeconv2DBlock(128, 64)
        )

        self.head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1), # 128 -> 32
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(16, output_dim, kernel_size=1),
            nn.Softplus()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        f0, f3, f6, f9, f12 = x, features[0], features[1], features[2], features[3] # 1,768,16,16
        
        # print(f0.shape, f3.shape, f6.shape, f9.shape, f12.shape)
        
        f12 = self.decoder12_upsampler(f12)
        f9 = self.decoder9(f9)
        f9 = self.decoder9_upsampler(torch.cat([f9, f12], dim=1))
        f6 = self.decoder6(f6)
        f6 = self.decoder6_upsampler(torch.cat([f6, f9], dim=1))
        f3 = self.decoder3(f3)
        f3 = self.decoder3_upsampler(torch.cat([f3, f6], dim=1))
        f0 = self.decoder0(f0)
        output = self.head(torch.cat([f0, f3], dim=1))
        return output, features[1], features[3]

if __name__ == '__main__':
    # Define checkpoint path
    ckpt_path = r"c:\Users\960\Desktop\DinoUNETR\dino_unetr\model.pth"
    
    print(f"Initializing MED_DINO_UNETR with checkpoint: {ckpt_path}")
    model = MED_DINO_UNETR(checkpoint_path=ckpt_path)
    print(model)
    
    # Test Forward Pass
    print("\nTesting forward pass...")
    input_tensor = torch.randn(1, 6, 512, 512)
    try:
        output_tensor,_,_ = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_tensor.shape}")
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")