"""U-Net and diffusion process of Ho et al. (2020), CIFAR-10 configuration."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


def group_norm(ch):
    return nn.GroupNorm(32, ch)


def zero_init(module):
    module.zero_init = True
    return module


def ho_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight, gain=1e-5 if getattr(m, "zero_init", False) else 1.0)
            nn.init.zeros_(m.bias)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout):
        super().__init__()
        self.norm1, self.conv1 = group_norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2, self.drop = group_norm(cout), nn.Dropout(dropout)
        self.conv2 = zero_init(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm, self.qkv = group_norm(ch), nn.Conv2d(ch, 3 * ch, 1)
        self.proj = zero_init(nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        attn = torch.softmax(q.transpose(1, 2) @ k * c ** -0.5, dim=-1)
        return x + self.proj((v @ attn.transpose(1, 2)).reshape(b, c, h, w))


class Block(nn.Module):
    def __init__(self, cin, cout, tdim, dropout, attn):
        super().__init__()
        self.res = ResBlock(cin, cout, tdim, dropout)
        self.attn = AttnBlock(cout) if attn else nn.Identity()

    def forward(self, x, temb):
        return self.attn(self.res(x, temb))


class Downsample(nn.Conv2d):
    def __init__(self, ch):
        super().__init__(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, temb=None):
        return super().forward(x)


class Upsample(nn.Conv2d):
    def __init__(self, ch):
        super().__init__(ch, ch, 3, padding=1)

    def forward(self, x, temb=None):
        return super().forward(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, ch, ch_mult, num_res_blocks, attn_resolutions, dropout, image_size):
        super().__init__()
        tdim, self.ch = ch * 4, ch
        self.temb = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        skips, cur, res = [ch], ch, image_size
        for i, mult in enumerate(ch_mult):
            for _ in range(num_res_blocks):
                self.down.append(Block(cur, ch * mult, tdim, dropout, res in attn_resolutions))
                cur = ch * mult
                skips.append(cur)
            if i < len(ch_mult) - 1:
                self.down.append(Downsample(cur))
                skips.append(cur)
                res //= 2
        self.mid = nn.ModuleList([Block(cur, cur, tdim, dropout, True), Block(cur, cur, tdim, dropout, False)])
        for i, mult in reversed(list(enumerate(ch_mult))):
            for _ in range(num_res_blocks + 1):
                self.up.append(Block(cur + skips.pop(), ch * mult, tdim, dropout, res in attn_resolutions))
                cur = ch * mult
            if i > 0:
                self.up.append(Upsample(cur))
                res *= 2
        self.out = nn.Sequential(group_norm(cur), nn.SiLU(), zero_init(nn.Conv2d(cur, 3, 3, padding=1)))
        ho_init(self)

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.ch))
        hs = [self.conv_in(x)]
        for layer in self.down:
            hs.append(layer(hs[-1], temb))
        h = hs[-1]
        for layer in self.mid:
            h = layer(h, temb)
        for layer in self.up:
            h = layer(torch.cat([h, hs.pop()], 1), temb) if isinstance(layer, Block) else layer(h)
        return self.out(h)


class Diffusion:
    def __init__(self, T, beta_start, beta_end, device):
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        ab = torch.cumprod(1 - betas, 0)
        ab_prev = F.pad(ab[:-1], (1, 0), value=1.0)
        post_var = betas * (1 - ab_prev) / (1 - ab)
        to = lambda x: x.float().to(device)
        self.T = T
        self.betas, self.ab, self.ab_prev = to(betas), to(ab), to(ab_prev)
        self.var_large = to(torch.cat([post_var[1:2], betas[1:]]))
        self.coef_x0 = to(betas * ab_prev.sqrt() / (1 - ab))
        self.coef_xt = to((1 - ab_prev) * (1 - betas).sqrt() / (1 - ab))

    def loss(self, model, x0):
        t = torch.randint(0, self.T, (len(x0),), device=x0.device)
        noise = torch.randn_like(x0)
        ab = self.ab[t].view(-1, 1, 1, 1)
        return F.mse_loss(model(ab.sqrt() * x0 + (1 - ab).sqrt() * noise, t), noise)

    def steps(self, sampler):
        return self.T if sampler == "ancestral" else int(sampler.split("_")[1])

    def predict_x0(self, model, x, t):
        ab = self.ab[t].view(-1, 1, 1, 1)
        eps = model(x, t)
        return ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1), eps

    @torch.no_grad()
    def sample(self, model, n, image_size, device, sampler="ancestral"):
        x = torch.randn(n, 3, image_size, image_size, device=device)
        if sampler == "ancestral":
            for i in reversed(range(self.T)):
                t = torch.full((n,), i, device=device, dtype=torch.long)
                x0, _ = self.predict_x0(model, x, t)
                x = self.coef_x0[i] * x0 + self.coef_xt[i] * x
                if i > 0:
                    x = x + self.var_large[i].sqrt() * torch.randn_like(x)
            return x
        seq = torch.linspace(0, self.T - 1, self.steps(sampler)).round().long().tolist()
        for i, j in zip(reversed(seq), reversed([-1] + seq[:-1])):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            x0, eps = self.predict_x0(model, x, t)
            ab_prev = self.ab[j] if j >= 0 else torch.tensor(1.0, device=device)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x
