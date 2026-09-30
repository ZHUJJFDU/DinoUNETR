"""Paper-aligned implementation of DoseDINO.

The six input channels are ordered as follows:

1. normalized CT intensity
2. PTV prescription
3. OAR priority
4. body mask
5. beam plate
6. PTV distance map

The first five channels form the anatomical stream. The final channel is
encoded separately by Parallel Geometric Injection (PGI).
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Mapping

import torch
import torch.nn as nn

# The vendored DINOv3 package uses absolute ``dinov3.*`` imports internally.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

from dinov3.models.vision_transformer import vit_base


INPUT_CHANNELS = (
    "ct",
    "ptv_prescription",
    "oar_priority",
    "body_mask",
    "beam_plate",
    "ptv_distance",
)
ANATOMICAL_CHANNELS = 5
GEOMETRIC_CHANNEL = 5


class DINOv3Encoder(nn.Module):
    """DINOv3 ViT-B/16 encoder adapted from three to five channels."""

    def __init__(self, checkpoint_path: str | None, input_channels: int = ANATOMICAL_CHANNELS):
        super().__init__()
        self.model = vit_base(
            img_size=256,
            patch_size=16,
            drop_path_rate=0.2,
            layerscale_init=1.0e-5,
            n_storage_tokens=4,
            qkv_bias=False,
            mask_k_bias=True,
        )
        self._load_pretrained_weights(checkpoint_path)
        if input_channels != 3:
            self._expand_patch_projection(input_channels)
        self.out_indices = [2, 5, 8, 11]

    def _load_pretrained_weights(self, checkpoint_path: str | None) -> None:
        if not checkpoint_path:
            warnings.warn("No DINOv3 checkpoint was supplied; the encoder is randomly initialized.")
            return
        if not os.path.exists(checkpoint_path):
            warnings.warn(
                f"DINOv3 checkpoint not found at {checkpoint_path!r}; "
                "the encoder is randomly initialized."
            )
            return

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("teacher", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
        backbone_state = {}
        for key, value in state_dict.items():
            if "ibot" in key or "dino_head" in key:
                continue
            backbone_state[key.replace("backbone.", "")] = value
        missing, unexpected = self.model.load_state_dict(backbone_state, strict=False)
        print(
            "Loaded DINOv3 encoder weights "
            f"({len(missing)} missing keys, {len(unexpected)} unexpected keys)."
        )

    def _expand_patch_projection(self, input_channels: int) -> None:
        """Copy pretrained RGB weights and zero initialize added channels."""
        projection = self.model.patch_embed.proj
        old_weight = projection.weight.detach()
        old_bias = projection.bias.detach() if projection.bias is not None else None
        embed_dim, pretrained_channels, kernel_h, kernel_w = old_weight.shape
        if input_channels < pretrained_channels:
            raise ValueError(
                f"input_channels must be at least {pretrained_channels}, got {input_channels}."
            )

        expanded = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embed_dim,
            kernel_size=(kernel_h, kernel_w),
            stride=projection.stride,
            padding=projection.padding,
            dilation=projection.dilation,
            groups=projection.groups,
            bias=old_bias is not None,
        )
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :pretrained_channels].copy_(old_weight)
            if old_bias is not None:
                expanded.bias.copy_(old_bias)
        self.model.patch_embed.proj = expanded

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.model.get_intermediate_layers(x, n=self.out_indices, reshape=True)


class PGIEncoder(nn.Module):
    """CNN branch that encodes the single-channel PTV-distance prior."""

    def __init__(self, input_channels: int = 1, embed_dim: int = 768):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.layer2(self.layer1(self.stem(x))))


class DirectedAttention(nn.Module):
    """Condition anatomical queries on geometric keys and values."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        # Keep the historical attribute names so existing state dictionaries load.
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, anatomical: torch.Tensor, geometric: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = anatomical.shape
        if geometric.shape != anatomical.shape:
            raise ValueError(
                "DirectedAttention inputs must have equal shapes, got "
                f"{tuple(anatomical.shape)} and {tuple(geometric.shape)}."
            )

        tokens = height * width
        q = self.norm1(anatomical.flatten(2).transpose(1, 2))
        k = self.norm2(geometric.flatten(2).transpose(1, 2))
        v = self.norm2(geometric.flatten(2).transpose(1, 2))
        head_dim = channels // self.num_heads

        q = self.to_q(q).reshape(batch, tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.to_k(k).reshape(batch, tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.to_v(v).reshape(batch, tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attention = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        fused = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        fused = self.proj_drop(self.proj(fused))
        fused = fused.transpose(1, 2).reshape(batch, channels, height, width)
        return anatomical + self.gamma * fused


class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.block = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_channels, out_channels),
            SingleConv2DBlock(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CDDDynamics(nn.Module):
    """dy/dt = -y + sin^2(y + G(x))."""

    def forward(self, y: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        return -y + torch.sin(y + drive).square()


class FixedStepCDD(nn.Module):
    """Continuous dose dynamics integrated by exactly four RK4 steps."""

    def __init__(self, channels: int, num_steps: int = 4):
        super().__init__()
        if num_steps != 4:
            raise ValueError("DoseDINO uses exactly four fixed RK4 steps, as reported in the paper.")
        self.num_steps = num_steps
        self.odefunc = CDDDynamics()
        self.drive_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        drive = self.drive_conv(x)
        y = torch.zeros_like(x)
        step_size = 1.0 / self.num_steps
        for _ in range(self.num_steps):
            k1 = self.odefunc(y, drive)
            k2 = self.odefunc(y + 0.5 * step_size * k1, drive)
            k3 = self.odefunc(y + 0.5 * step_size * k2, drive)
            k4 = self.odefunc(y + step_size * k3, drive)
            y = y + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.out_conv(y)


class CDDHead(nn.Sequential):
    """Continuous Dose Dynamics head with a non-negative Softplus output."""

    def __init__(self, in_channels: int = 128, output_channels: int = 1, rk4_steps: int = 4):
        super().__init__(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=False),
            FixedStepCDD(channels=32, num_steps=rk4_steps),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(16, output_channels, kernel_size=1),
            nn.Softplus(),
        )


class DoseDINO(nn.Module):
    """DoseDINO with a DINOv3 stream, PGI stream, and CDD-Head."""

    def __init__(
        self,
        checkpoint_path: str | None,
        embed_dim: int = 768,
        input_channels: int = 6,
        output_channels: int = 1,
        rk4_steps: int = 4,
        **legacy_kwargs,
    ):
        super().__init__()
        if "input_dim" in legacy_kwargs:
            input_channels = legacy_kwargs.pop("input_dim")
        if "output_dim" in legacy_kwargs:
            output_channels = legacy_kwargs.pop("output_dim")
        if legacy_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(legacy_kwargs)}")
        if input_channels != len(INPUT_CHANNELS):
            raise ValueError(f"DoseDINO expects 6 input channels, got {input_channels}.")

        self.backbone = DINOv3Encoder(checkpoint_path, input_channels=ANATOMICAL_CHANNELS)
        self.geo_encoder = PGIEncoder(input_channels=1, embed_dim=embed_dim)
        self.fusion_layer = DirectedAttention(dim=embed_dim)

        self.decoder0 = nn.Sequential(Conv2DBlock(input_channels, 32), Conv2DBlock(32, 64))
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
            Conv2DBlock(512, 256),
            Conv2DBlock(256, 256),
            SingleDeconv2DBlock(256, 128),
        )
        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(256, 128),
            Conv2DBlock(128, 128),
            SingleDeconv2DBlock(128, 64),
        )
        self.head = CDDHead(128, output_channels, rk4_steps=rk4_steps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != len(INPUT_CHANNELS):
            raise ValueError(
                "DoseDINO expects input shaped (B, 6, H, W) with channel order "
                f"{INPUT_CHANNELS}; got {tuple(x.shape)}."
            )

        anatomical = x[:, :ANATOMICAL_CHANNELS]
        geometric = x[:, GEOMETRIC_CHANNEL : GEOMETRIC_CHANNEL + 1]
        f3, f6, f9, f12 = self.backbone(anatomical)
        f12_fused = self.fusion_layer(f12, self.geo_encoder(geometric))

        f0 = self.decoder0(x)
        f12_up = self.decoder12_upsampler(f12_fused)
        f9 = self.decoder9_upsampler(torch.cat([self.decoder9(f9), f12_up], dim=1))
        f6 = self.decoder6_upsampler(torch.cat([self.decoder6(f6), f9], dim=1))
        f3 = self.decoder3_upsampler(torch.cat([self.decoder3(f3), f6], dim=1))
        dose = self.head(torch.cat([f0, f3], dim=1))
        return dose, f6, f12_fused


# Backward-compatible names for checkpoints and scripts from the development repo.
MedDINOv3Backbone = DINOv3Encoder
GeometryEncoder = PGIEncoder
CrossAttentionFusion = DirectedAttention
nmODEFunc = CDDDynamics
nmODEBlock = FixedStepCDD
MED_DINO_UNETR_Distance_nmODE = DoseDINO


__all__ = [
    "INPUT_CHANNELS",
    "DINOv3Encoder",
    "PGIEncoder",
    "DirectedAttention",
    "CDDDynamics",
    "FixedStepCDD",
    "CDDHead",
    "DoseDINO",
    "MED_DINO_UNETR_Distance_nmODE",
]
