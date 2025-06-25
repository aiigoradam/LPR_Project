# models/diffusion_sr3.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers import DDIMScheduler


# ========= helpers =====================================================
def sinusoidal_embedding(level: torch.Tensor, dim: int) -> torch.Tensor:
    """
    sine-cosine embedding.
      level : sqrt(alpha_bat_t)  ∈ (0,1]   - shape (B,)
      dim   : embedding dimension (even)
      return: shape (B, dim)
    """
    half = dim // 2
    step = torch.arange(half, device=level.device, dtype=level.dtype) / half
    emb = level[:, None] * torch.exp(-math.log(1e4) * step[None])
    return torch.cat([emb.sin(), emb.cos()], dim=1)


class GNFactory:
    """Creates GroupNorm layers with a preferred group count."""

    def __init__(self, groups: int = 8):
        self.groups = groups

    def __call__(self, channels: int) -> nn.GroupNorm:
        g = self.groups
        # ensure groups divide channels; fall back to the greatest common divisor
        if channels % g != 0 or channels < g:
            g = math.gcd(channels, g) or 1
        return nn.GroupNorm(g, channels, eps=1e-6, affine=True)


class TimeFiLM(nn.Module):
    """Affine FiLM from a time embedding."""

    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels * 2))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp(t_emb).chunk(2, dim=1)
        return x * (1 + gamma[..., None, None]) + beta[..., None, None]


class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, time_dim: int, gn):
        super().__init__()
        self.norm1 = gn(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.norm2 = gn(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.film = TimeFiLM(time_dim, out_c)
        self.skip = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        h = self.film(h, t_emb)
        return F.silu(h + self.skip(x))


class Attention(nn.Module):
    """Single-head spatial attention."""

    def __init__(self, channels: int, gn):
        super().__init__()
        self.norm = gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels**-0.5

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        n = h * w
        q = q.reshape(b, c, n).permute(0, 2, 1)
        k = k.reshape(b, c, n)
        v = v.reshape(b, c, n).permute(0, 2, 1)
        attn = torch.bmm(q * self.scale, k)
        attn = attn.softmax(-1)
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        return self.proj(out) + x


class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNetDiffusion(nn.Module):
    """
    SR3-style ε-prediction U-Net for 256x64 RGB licence plates.
      in  : (x_t || distorted) → 6 channels
      out : eps_pred_t               → 3 channels
    """

    def __init__(self, base: int = 32, time_dim: int = 128, gn_groups: int = 8):
        super().__init__()
        gn = GNFactory(gn_groups)

        ch = [base, base * 2, base * 4, base * 8, base * 8]

        # --- time embedding ---
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim))

        # --- stem ---
        self.init_conv = nn.Conv2d(6, ch[0], 3, padding=1, bias=False)

        # --- encoder ---
        self.enc0 = nn.ModuleList([ResBlock(ch[0], ch[0], time_dim, gn) for _ in range(2)])
        self.down1 = Down(ch[0])
        self.enc1 = nn.ModuleList([ResBlock(ch[0], ch[1], time_dim, gn), ResBlock(ch[1], ch[1], time_dim, gn)])
        self.down2 = Down(ch[1])
        self.enc2 = nn.ModuleList([ResBlock(ch[1], ch[2], time_dim, gn), ResBlock(ch[2], ch[2], time_dim, gn)])
        self.down3 = Down(ch[2])
        self.enc3 = nn.ModuleList([ResBlock(ch[2], ch[3], time_dim, gn), ResBlock(ch[3], ch[3], time_dim, gn)])
        self.attn3 = Attention(ch[3], gn)
        self.down4 = Down(ch[3])

        # --- bottleneck ---
        self.enc4 = nn.ModuleList([ResBlock(ch[3], ch[4], time_dim, gn), ResBlock(ch[4], ch[4], time_dim, gn)])
        self.attn4 = Attention(ch[4], gn)

        # --- decoder ---
        self.up4 = Up(ch[4])
        self.dec3 = nn.ModuleList([ResBlock(ch[4] + ch[3], ch[3], time_dim, gn), ResBlock(ch[3], ch[3], time_dim, gn)])
        self.up3 = Up(ch[3])
        self.dec2 = nn.ModuleList([ResBlock(ch[3] + ch[2], ch[2], time_dim, gn), ResBlock(ch[2], ch[2], time_dim, gn)])
        self.up2 = Up(ch[2])
        self.dec1 = nn.ModuleList([ResBlock(ch[2] + ch[1], ch[1], time_dim, gn), ResBlock(ch[1], ch[1], time_dim, gn)])
        self.up1 = Up(ch[1])
        self.dec0 = nn.ModuleList([ResBlock(ch[1] + ch[0], ch[0], time_dim, gn), ResBlock(ch[0], ch[0], time_dim, gn)])

        # --- head ---
        self.outc = nn.Conv2d(ch[0], 3, 1)

    def _run(self, blocks, x, t_emb):
        for blk in blocks:
            x = blk(x, t_emb)
        return x

    def forward(self, x_t, distorted, noise_level):
        """
        Args
          x_t        : noisy plate (B,3,256,64)
          distorted  : distorted plate (B,3,256,64)
          noise_level: sqrt(alpha_bar_t) (B,)
        Returns
          eps_pred_t : predicted noise (B,3,256,64)
        """
        t_emb = self.time_mlp(sinusoidal_embedding(noise_level, self.time_mlp[0].in_features))

        h = self.init_conv(torch.cat([x_t, distorted], 1))

        e0 = self._run(self.enc0, h, t_emb)
        e1 = self._run(self.enc1, self.down1(e0), t_emb)
        e2 = self._run(self.enc2, self.down2(e1), t_emb)
        e3 = self._run(self.enc3, self.down3(e2), t_emb)
        e3 = self.attn3(e3)
        e4 = self._run(self.enc4, self.down4(e3), t_emb)
        e4 = self.attn4(e4)

        d3 = self._run(self.dec3, torch.cat([self.up4(e4), e3], 1), t_emb)
        d2 = self._run(self.dec2, torch.cat([self.up3(d3), e2], 1), t_emb)
        d1 = self._run(self.dec1, torch.cat([self.up2(d2), e1], 1), t_emb)
        d0 = self._run(self.dec0, torch.cat([self.up1(d1), e0], 1), t_emb)

        return self.outc(d0)


# ================================================================================


class Diffusion:
    """
    Wrapper around diffusers schedulers that handles:
       velocity-based training loss     (DDPMScheduler, v_prediction)
       DDIM sampling at inference time (also v_prediction)
    """

    def __init__(self, T: int = 1000, device: torch.device | str = "cuda"):
        self.train_sched = DDPMScheduler(
            num_train_timesteps=T, beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction"
        )

        self.sample_sched_cfg = {"prediction_type": "v_prediction", "timestep_spacing": "trailing"}

    # -------------------------------------------------------- #
    # training loss
    # -------------------------------------------------------- #
    def p_losses(self, model, clean, cond):
        """
        clean : (B,3,256,64)   - ground-truth image
        cond  : (B,3,256,64)   - distorted plate (condition)
        """
        b, *_ = clean.shape
        device = clean.device

        # random timesteps
        t = torch.randint(0, self.train_sched.config.num_train_timesteps, (b,), device=device, dtype=torch.long)

        # forward-diffuse
        noise = torch.randn_like(clean)
        noisy = self.train_sched.add_noise(clean, noise, t)

        # target velocity  v = sqrt(alpha_t) * eps - sqrt(1 - alpha_bar_t) * x_0 
        v_target = self.train_sched.get_velocity(clean, noise, t)

        # scalar sqrt(alpha_bar_t)
        alpha_bar = self.train_sched.alphas_cumprod.to(device)[t]
        noise_lvl = alpha_bar.sqrt()  # (B,)
        v_pred = model(noisy, cond, noise_lvl)  # (B,3,H,W)

        return F.mse_loss(v_pred, v_target)

    # -------------------------------------------------------- #
    # sampling
    # -------------------------------------------------------- #
    @torch.no_grad()
    def sample(self, model, cond, steps: int = 20):
        """
        Deterministic DDIM sampling in v-prediction mode.
        cond : (B,3,256,64)
        """
        sampler = DDIMScheduler.from_config(self.train_sched.config, **self.sample_sched_cfg)
        sampler.set_timesteps(steps, device=cond.device)

        # start from pure Gaussian noise
        x = torch.randn_like(cond)

        for t in sampler.timesteps:
            # compute noise-level scalar
            level = sampler.alphas_cumprod.to(cond.device)[t].sqrt()
            noise_level = level.expand(cond.size(0))  # (B,)
            v_pred = model(x, cond, noise_level)

            x = sampler.step(v_pred, t, x).prev_sample

        return x.clamp(-1, 1)
