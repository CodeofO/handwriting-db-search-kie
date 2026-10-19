"""필드 이미지 증강(학습) · 정규화(평가). GPU · CPU 텐서 모두 가능.

입력 uint8 (B, h, w) (흰 배경 255) → 출력 float (B, 1, H, W) in [−1, 1] (흰색 = 1).
학습: 회전 ±4° · 기울임 · 배율 · 이동 · 괘선 합성 · 블러 · 밝기/대비/감마 · 노이즈.
keep_ink: 가로 배율 · 이동을 그 이미지 글씨 범위가 캔버스 안에 남는 값으로 제한한다
  (제한이 없으면 이동이 왼쪽 정렬된 글씨 앞부분을, 확대가 캔버스를 채운 긴 줄의 끝을 자른다).
"""
import math

import torch
import torch.nn.functional as F


def augment(x, H, W, train, keep_ink=False):
    x = x.unsqueeze(1).float() / 255.0
    B = x.size(0)
    if not train:
        return (F.interpolate(x, (H, W), mode="bilinear", align_corners=False) - 0.5) / 0.5
    d = x.device
    ang = (torch.rand(B, device=d) * 2 - 1) * math.radians(4)
    shear = (torch.rand(B, device=d) * 2 - 1) * 0.35
    sc = 0.85 + torch.rand(B, device=d) * 0.25
    asp = 0.9 + torch.rand(B, device=d) * 0.2
    tx = (torch.rand(B, device=d) * 2 - 1) * 0.08
    ty = (torch.rand(B, device=d) * 2 - 1) * 0.10
    cos, sin = torch.cos(ang), torch.sin(ang)
    if keep_ink:                                       # 입력 정규화 좌표: 글씨 [s, e] → 출력 x = ((x_in − tx)·z ∓ |s01|·y)/cos 가 [−1, 1] 안
        col = x[:, 0].amin(1) < 0.5                    # (B, w) 잉크 열
        w_ = col.size(1); idx = torch.arange(w_, device=d)
        has = col.any(1)
        s_ = torch.where(col, idx, torch.full_like(idx, w_)).amin(1).float(); e_ = torch.where(col, idx, torch.full_like(idx, -1)).amax(1).float() + 1
        s_ = torch.where(has, 2 * s_ / w_ - 1, torch.full_like(s_, -1.0)); e_ = torch.where(has, 2 * e_ / w_ - 1, torch.full_like(e_, -1.0))
        room = (cos - ((-sin + shear) / 3).abs()).clamp(min=0.3)
        z = (sc * asp).minimum(2 * room / (e_ - s_).clamp(min=1e-3))
        lo, hi = e_ - room / z, s_ + room / z
        tx = torch.where(lo <= hi, torch.maximum(torch.minimum(tx, hi), lo), (lo + hi) / 2)
        asp = z / sc
    theta = torch.zeros(B, 2, 3, device=d)
    theta[:, 0, 0] = cos / (sc * asp); theta[:, 0, 1] = (-sin + shear) / (sc * asp) / 3; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin / sc * 3;       theta[:, 1, 1] = cos / sc;                     theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    x = 1 - F.grid_sample(1 - x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # 바깥 = 흰색
    # 괘선 합성 (가로 p=.3, 세로 p=.2)
    yy = torch.arange(H, device=d).view(1, 1, H, 1); xx = torch.arange(W, device=d).view(1, 1, 1, W)
    hl = torch.rand(B, device=d) < 0.3
    hy = torch.where(torch.rand(B, device=d) < 0.5, torch.rand(B, device=d) * 0.15, 0.85 + torch.rand(B, device=d) * 0.15) * H
    ht = 1 + (torch.rand(B, device=d) * 2).long()
    hmask = ((yy - hy.view(B, 1, 1, 1)).abs() < ht.view(B, 1, 1, 1)) & hl.view(B, 1, 1, 1)
    vl = torch.rand(B, device=d) < 0.2
    vx = torch.where(torch.rand(B, device=d) < 0.5, torch.rand(B, device=d) * 0.06, 0.94 + torch.rand(B, device=d) * 0.06) * W
    vmask = ((xx - vx.view(B, 1, 1, 1)).abs() < 1.5) & vl.view(B, 1, 1, 1)
    lv = 0.2 + torch.rand(B, 1, 1, 1, device=d) * 0.5
    x = torch.where(hmask | vmask, torch.minimum(x, lv), x)
    # 블러 (배치의 절반 확률로, 그 안에서 60% 이미지에 3×3 또는 5×5 가우시안)
    if torch.rand(()) < 0.5:
        k = 5 if torch.rand(()) < 0.4 else 3
        s = 0.6 + float(torch.rand(())) * 0.6
        g = torch.exp(-(torch.arange(k, device=d) - k // 2) ** 2 / (2 * s * s)); g = g / g.sum()
        ker = (g[:, None] * g[None, :]).view(1, 1, k, k)
        bm = (torch.rand(B, 1, 1, 1, device=d) < 0.6)
        x = torch.where(bm, F.conv2d(F.pad(x, [k // 2] * 4, mode="replicate"), ker), x)
    # 밝기 · 대비 · 감마 · 노이즈
    gam = torch.exp((torch.rand(B, 1, 1, 1, device=d) * 2 - 1) * 0.4)
    x = x.clamp(1e-4, 1) ** gam
    con = 0.6 + torch.rand(B, 1, 1, 1, device=d) * 0.6
    bri = (torch.rand(B, 1, 1, 1, device=d) * 2 - 1) * 0.1
    x = (x - 0.5) * con + 0.5 + bri
    x = x + torch.randn_like(x) * (torch.rand(B, 1, 1, 1, device=d) * 0.05)
    return (x.clamp(0, 1) - 0.5) / 0.5
