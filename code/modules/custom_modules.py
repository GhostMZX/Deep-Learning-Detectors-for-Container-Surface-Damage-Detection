import torch
import torch.nn as nn
import math

from ultralytics.nn.modules.block import C3k2, Bottleneck


class LSKblock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class C3k2_LSK(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, False, g, shortcut)
        self.m = nn.ModuleList(
            nn.Sequential(Bottleneck(self.c, self.c, shortcut, g), LSKblock(self.c))
            for _ in range(n)
        )


class CoordAtt(nn.Module):
    def __init__(self, dim: int, reduction: int = 32):
        super().__init__()
        mip = max(8, dim // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(dim, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, dim, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_w * a_h


class C3k2_CA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, False, g, shortcut)
        self.m = nn.ModuleList(
            nn.Sequential(Bottleneck(self.c, self.c, shortcut, g), CoordAtt(self.c))
            for _ in range(n)
        )


class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, max(c // r, 8), 1)
        self.fc2 = nn.Conv2d(max(c // r, 8), c, 1)

    def forward(self, x):
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(self.pool(x)))))
        return x * s


class CBAM(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        hidden = max(c // r, 8)
        self.ch_fc1 = nn.Linear(c, hidden)
        self.ch_fc2 = nn.Linear(hidden, c)
        self.sp_conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        ch = torch.sigmoid(
            self.ch_fc2(torch.relu(self.ch_fc1(x.mean(dim=(2, 3))))
                        + torch.relu(self.ch_fc1(x.amax(dim=(2, 3)))))
        ).unsqueeze(-1).unsqueeze(-1)
        x = x * ch
        sp = torch.sigmoid(self.sp_conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sp


class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        gx = x.reshape(b * self.groups, -1, h, w)
        xh = self.pool_h(gx)
        xw = self.pool_w(gx).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([xh, xw], dim=2))
        xh, xw = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(gx * xh.sigmoid() * xw.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(gx)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1).contiguous())
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1).contiguous())
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        wts = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (gx * wts.sigmoid()).reshape(b, c, h, w)


class C3k2_SE(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, False, g, shortcut)
        self.m = nn.ModuleList(
            nn.Sequential(Bottleneck(self.c, self.c, shortcut, g), SE(self.c))
            for _ in range(n)
        )


class C3k2_CBAM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, False, g, shortcut)
        self.m = nn.ModuleList(
            nn.Sequential(Bottleneck(self.c, self.c, shortcut, g), CBAM(self.c))
            for _ in range(n)
        )


class C3k2_EMA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, False, g, shortcut)
        self.m = nn.ModuleList(
            nn.Sequential(Bottleneck(self.c, self.c, shortcut, g), EMA(self.c))
            for _ in range(n)
        )


class ECBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        k = int(abs(math.log2(c) + 1) // 2 * 2 - 1)
        k = max(3, k if k % 2 == 1 else k + 1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sp_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        se = x.mean(dim=(2, 3), keepdim=False)
        se = self.eca_conv(se.unsqueeze(1)).squeeze(1).sigmoid()
        x = x * se.unsqueeze(-1).unsqueeze(-1)
        sp = torch.sigmoid(self.sp_conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sp


class C3k2_ECBAM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, False, g, shortcut)
        self.m = nn.ModuleList(
            nn.Sequential(Bottleneck(self.c, self.c, shortcut, g), ECBAM(self.c))
            for _ in range(n)
        )
