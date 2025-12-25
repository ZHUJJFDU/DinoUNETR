import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
# Ensure we can import from the local dinov3 package
current_dir = os.path.dirname(os.path.abspath(__file__))

from dinov3.models.vision_transformer import vit_base


class LayoutAdapter(nn.Module):
    def __init__(self, feature_dim=768, layout_dim=768, num_heads=8):
        super().__init__()
        # 1. Pre-Norm
        self.norm = nn.LayerNorm(feature_dim)
        
        # 2. Cross-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads, 
            kdim=layout_dim, 
            vdim=layout_dim,
            batch_first=True
        )
        
        # 3. Gating Parameter (Zero Init)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feature, layout):
        """
        :param feature: [batch_size, feature_dim, H, W]
        :param layout: [batch_size, seq_len, layout_dim]
        :return: [batch_size, feature_dim, H, W]
        """
        B, C, H, W = feature.shape

        # (B, C, H, W) -> (B, H*W, C)
        x = feature.flatten(2).transpose(1, 2)

        # 1. Norm (Query)
        x_norm = self.norm(x)
        
        # 2. Cross-Attention (Query=Image, Key/Value=Layout)
        attn_out, _ = self.attn(
            query=x_norm,
            key=layout,
            value=layout
        )
        
        # 3. Gated Residual
        x = x + self.gamma * attn_out
        
        # 4. Reshape
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class LayoutEmbedding(nn.Module):
    def __init__(self, embed_dim=768, num_angle_samples=36):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_angle_samples = num_angle_samples

        input_scalar_dim = 6
        self.scalar_mlp = nn.Sequential(
            nn.Linear(input_scalar_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

        input_angle_dim = num_angle_samples * 2
        self.angle_mlp = nn.Sequential(
            nn.Linear(input_angle_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )

    def _process_angles(self, batch_angle_lists):
        """
        Convert angle list to sine/cosine encoding
        :param batch_angle_lists: List[List[float]] or List[JSON string]
        :return: [batch_size, num_angle_samples * 2]
        """
        batch_feats = []
        for angles in batch_angle_lists:
            # Parse JSON if needed
            if isinstance(angles, str):
                angles = json.loads(angles)

            # Convert to numpy
            arr = np.array(angles, dtype=np.float32)

            if len(arr) == 0:
                resampled = np.zeros(self.num_angle_samples)
            else:
                # Interpolate to fixed length
                if len(arr) == 1:
                    resampled = np.full(self.num_angle_samples, arr[0])
                else:
                    old = np.linspace(0, 1, len(arr))
                    new = np.linspace(0, 1, self.num_angle_samples)
                    resampled = np.interp(new, old, arr)
            
            # To radians
            rads = np.deg2rad(resampled)
            sin_angles = np.sin(rads)
            cos_angles = np.cos(rads)

            # Flatten -> (num_angle_samples * 2,)
            feats = np.stack([sin_angles, cos_angles], axis=1).flatten()
            batch_feats.append(feats)

        # Return Tensor
        return torch.tensor(np.array(batch_feats), dtype=torch.float32)
    
    def forward(self, data):
        """
        :param data: Dict containing 'spacing', 'isocenter', 'angle_list'
        :return: [batch_size, 2, embed_dim]
        """
        space = data['spacing'] # (B, 3)
        isocenter = data['isocenter'] # (B, 3)
        angle_list = data['angle_list'] # List[List]
        
        # 1. Scalar Token
        scalar_input = torch.cat([space, isocenter], dim=1) # (B, 6)
        scalar_token = self.scalar_mlp(scalar_input).unsqueeze(1) # (B, 1, 768)

        # 2. Angle Token
        angle_input = self._process_angles(angle_list).to(space.device) # (B, 72)
        angle_token = self.angle_mlp(angle_input).unsqueeze(1) # (B, 1, 768)

        # 3. Combine
        layout_tokens = torch.cat([scalar_token, angle_token], dim=1) # (B, 2, 768)
        return layout_tokens

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
                
                # Filter incompatible keys and remove prefixes
                new_state_dict = {}
                for k, v in state_dict.items():
                    if 'ibot' in k or 'dino_head' in k:
                        continue
                    new_key = k.replace('backbone.', '')
                    new_state_dict[new_key] = v
                
                # Load weights
                missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
                print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            except Exception as e:
                print(f"Error loading weights: {e}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Using random init.")

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

# --- Main Model ---
class MED_DINO_UNETR(nn.Module):
    def __init__(self, checkpoint_path, embed_dim=768, input_dim=6, output_dim=1):
        super().__init__()
        # 1. Use the new MedDINOv3 Backbone
        self.backbone = MedDINOv3Backbone(checkpoint_path, input_dim)

        self.layout_embedder = LayoutEmbedding(embed_dim=embed_dim)

        self.adapter3 = LayoutAdapter(feature_dim=embed_dim)
        self.adapter6 = LayoutAdapter(feature_dim=embed_dim)
        self.adapter9 = LayoutAdapter(feature_dim=embed_dim)
        self.adapter12 = LayoutAdapter(feature_dim=embed_dim)

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
            nn.Conv2d(128, 32, kernel_size=3, padding=1), # 128 -> 32
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(16, output_dim, kernel_size=1),
            nn.Softplus()
        )
    
    def forward(self, x, data):
        features = self.backbone(x)
        f0 = x
        f3 = features[0].detach() 
        f6 = features[1].detach()
        f9 = features[2].detach()
        f12 = features[3].detach()
        
        layout_tokens = self.layout_embedder(data)

        f3_layout = self.adapter3(f3, layout_tokens)
        f6_layout = self.adapter6(f6, layout_tokens)
        f9_layout = self.adapter9(f9, layout_tokens)
        f12_layout = self.adapter12(f12, layout_tokens)

        f12 = self.decoder12_upsampler(f12_layout)
        f9 = self.decoder9(f9_layout)
        f9 = self.decoder9_upsampler(torch.cat([f9, f12], dim=1))
        f6 = self.decoder6(f6_layout)
        f6 = self.decoder6_upsampler(torch.cat([f6, f9], dim=1))
        f3 = self.decoder3(f3_layout)
        f3 = self.decoder3_upsampler(torch.cat([f3, f6], dim=1))
        f0 = self.decoder0(f0)
        output = self.head(torch.cat([f0, f3], dim=1))
        return output, features[1], features[3]

if __name__ == '__main__':
    # ... (前面的初始化保持不变) ...
    
    # Test Forward Pass
    print("\nTesting forward pass...")
    input_tensor = torch.randn(1, 6, 512, 512)
    
    # [关键修改] 构造假的 data 字典
    fake_data = {
        'spacing': torch.tensor([[1.0, 1.0, 3.0]]),       # (B, 3)
        'isocenter': torch.tensor([[0.5, 0.5, 0.5]]),     # (B, 3)
        'angle_list': [[0, 30, 60, 90]]                   # List of lists
    }
    
    try:
        # [关键修改] 传入 fake_data
        output_tensor, _, _ = model(input_tensor, fake_data)
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_tensor.shape}")
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc() # 打印详细报错信息