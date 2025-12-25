import timm
import torch
import torch.nn as nn

class DINOv3(nn.Module):
    def __init__(self, model_name, input_dim=6):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)

        # print(self.model)
        self.adapt_dinov3_channels(input_dim)
        # print(self.model)

        self.out_indices = [2, 5, 8, 11]
        self.model.prune_intermediate_layers(
            indices=self.out_indices, 
            prune_head=True, 
            prune_norm=True
        )
        self.num_registers = 4

    def adapt_dinov3_channels(self, input_dim=6):
        patch_embed_layer = self.model.patch_embed.proj
        
        # 获取原有参数
        original_weights = patch_embed_layer.weight.data
        original_bias = patch_embed_layer.bias

        new_in_channels = input_dim
        embed_dim, _, k_h, k_w = original_weights.shape
        
        # 创建一个新的6通道卷积层
        new_layer = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=embed_dim,
            kernel_size=(k_h, k_w),
            stride=patch_embed_layer.stride,
            padding=patch_embed_layer.padding
        )

        # 使用 torch.no_grad() 来确保权重操作不被追踪梯度
        with torch.no_grad():
            # 复制前3个通道的权重
            new_layer.weight[:, :3, :, :] = original_weights[:, :3, :, :]
            # 将后3个通道的权重初始化为零
            new_layer.weight[:, new_in_channels-3:, :, :] = 0
            # 复制偏置
            if original_bias is not None:
                new_layer.bias.data.copy_(original_bias.data)

        # 直接替换模型中的层
        self.model.patch_embed.proj = new_layer

    def forward(self, x):
        return self.model.forward_intermediates(x, indices=self.out_indices, intermediates_only=True)


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


class DINO_UNETR(nn.Module):
    def __init__(self, embed_dim=768, input_dim=6, output_dim=1):
        super().__init__()
        # DINOv3 Encoder
        self.backbone = DINOv3('vit_base_patch16_dinov3.lvd1689m', input_dim)

        # U-Net Decoder
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
    model = DINO_UNETR()
    # print(model)
    # 测试模型
    input_tensor = torch.randn(1, 6, 512, 512)  
    output_tensor, f3, f6 = model(input_tensor)
    print(f3.shape, f6.shape)
    print(output_tensor.shape)
