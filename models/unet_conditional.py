# models/cond_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Noise Encoder 
# ---------------------------
class NoiseEncoder(nn.Module):
    """
    Takes a distorted plate (3x256x64) and produces a latent code z ∈ R^Z that
    summarizes which distortion pattern is present. We'll use a small CNN + pooling.
    """

    def __init__(self, in_ch: int = 3, base: int = 32, z_dim: int = 128):
        super().__init__()
        # We progressively downsample from 256x64 -> 128x32 -> 64x16 -> 32x8 -> 16x4
        # Then produce a z_dim vector via global pooling + FC.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  
        )
        # Final projection to a z_dim vector
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Linear(base * 8, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 256, 64)
        returns z: (B, z_dim)
        """
        h = self.conv1(x)  
        h = self.conv2(h)  
        h = self.conv3(h)  
        h = self.conv4(h)  
        h = self.global_pool(h).flatten(1)  
        z = self.fc(h)  
        return z


# ---------------------------
# FiLM conditioning block
# ---------------------------
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation: Given a latent code z ∈ R^Z, produce
    (gamma, beta) so that for an intermediate feature map F (shape (B, C, H, W)),
    we do F' = gamma[...,None,None] * F + beta[...,None,None].
    """

    def __init__(self, z_dim: int, num_channels: int):
        super().__init__()
        self.fc = nn.Linear(z_dim, num_channels * 2)
        # split the output into gamma (scale) and beta (shift)
        # Each of shape (B, num_channels).

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Input: z (B, z_dim)
        Output: a tensor of shape (B, 2*num_channels) = [gamma||beta]
        """
        out = self.fc(z)  # (B, 2 * num_channels)
        return out  # caller must split into gamma, beta.


# ---------------------------
# The Conditional U-Net Restorer R
# ---------------------------
class CondUNet(nn.Module):
    """
    A U-Net that restores a clean plate from a distorted input,
    but at each encoder & decoder stage we apply FiLM with z.
    - in_ch = 3 (RGB distorted)
    - out_ch = 3 (RGB restored)
    - base = 32
    - z_dim = 128
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 32, z_dim: int = 128):
        super().__init__()
        # Encoder blocks: each has DoubleConvBN followed by FiLM conditioning
        self.enc0_conv = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.enc0_film = FiLM(z_dim, base)

        self.enc1_pool = nn.MaxPool2d(2)  
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )
        self.enc1_film = FiLM(z_dim, base * 2)

        self.enc2_pool = nn.MaxPool2d(2)  
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )
        self.enc2_film = FiLM(z_dim, base * 4)

        self.enc3_pool = nn.MaxPool2d(2) 
        self.enc3_conv = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 8, base * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )
        self.enc3_film = FiLM(z_dim, base * 8)

        self.enc4_pool = nn.MaxPool2d(2)  
        self.enc4_conv = nn.Sequential(
            nn.Conv2d(base * 8, base * 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 16, base * 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 16),
            nn.ReLU(inplace=True),
        )
        self.enc4_film = FiLM(z_dim, base * 16)

        # ------------------------------
        # Decoder blocks
        # Note: we also apply FiLM on decoder features
        # ------------------------------
        self.up3 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2) 
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(base * 16, base * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 8, base * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )
        self.dec3_film = FiLM(z_dim, base * 8)

        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)  
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )
        self.dec2_film = FiLM(z_dim, base * 4)

        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2) 
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )
        self.dec1_film = FiLM(z_dim, base * 2)

        self.up0 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2) 
        self.dec0_conv = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.dec0_film = FiLM(z_dim, base)

        # Final output conv
        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1, bias=True)

    def _apply_film(self, feat: torch.Tensor, z: torch.Tensor, film: FiLM) -> torch.Tensor:
        """
        Split the 2·C outputs of film(z) into gamma, beta ∈ R^(BxC),
        then do feat' = gamma[...,None,None] * feat + beta[...,None,None].
        """
        B, C, H, W = feat.shape
        gammas_betas = film(z)  
        gamma, beta = torch.split(gammas_betas, C, dim=1)  
        gamma = gamma.view(B, C, 1, 1)  
        beta = beta.view(B, C, 1, 1)
        return gamma * feat + beta

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: distorted (B, 3, 256, 64)
        z: noise code      (B, z_dim)
        returns: restored (B, 3, 256, 64)
        """
        # ------- Encoder -------
        e0 = self.enc0_conv(x) 
        e0 = self._apply_film(e0, z, self.enc0_film)

        e1 = self.enc1_pool(e0)  
        e1 = self.enc1_conv(e1)  
        e1 = self._apply_film(e1, z, self.enc1_film)

        e2 = self.enc2_pool(e1) 
        e2 = self.enc2_conv(e2) 
        e2 = self._apply_film(e2, z, self.enc2_film)

        e3 = self.enc3_pool(e2)  
        e3 = self.enc3_conv(e3)  
        e3 = self._apply_film(e3, z, self.enc3_film)

        e4 = self.enc4_pool(e3)  
        e4 = self.enc4_conv(e4)  
        e4 = self._apply_film(e4, z, self.enc4_film)

        # ------- Decoder -------
        d3 = self.up3(e4)  
        d3 = torch.cat([d3, e3], dim=1)  
        d3 = self.dec3_conv(d3) 
        d3 = self._apply_film(d3, z, self.dec3_film)

        d2 = self.up2(d3) 
        d2 = torch.cat([d2, e2], dim=1) 
        d2 = self.dec2_conv(d2)  
        d2 = self._apply_film(d2, z, self.dec2_film)

        d1 = self.up1(d2) 
        d1 = torch.cat([d1, e1], dim=1) 
        d1 = self.dec1_conv(d1)  
        d1 = self._apply_film(d1, z, self.dec1_film)

        d0 = self.up0(d1)  
        d0 = torch.cat([d0, e0], dim=1)  
        d0 = self.dec0_conv(d0) 
        d0 = self._apply_film(d0, z, self.dec0_film)

        out = self.out_conv(d0) 
        return out


# ---------------------------
# wrapper that fuses NoiseEncoder + CondUNet
# ---------------------------
class FullModel(nn.Module):
    def __init__(self, noise_enc: nn.Module, restorer: nn.Module):
        super().__init__()
        self.noise_enc = noise_enc
        self.restorer = restorer

    def forward(self, distorted: torch.Tensor) -> torch.Tensor:
        """
        distorted: (B, 3, 256, 64)
        returns: restored -> (B, 3, 256, 64)
        """
        z = self.noise_enc(distorted)  # (B, z_dim)
        restored = self.restorer(distorted, z)  # (B, 3, 256, 64)
        return restored