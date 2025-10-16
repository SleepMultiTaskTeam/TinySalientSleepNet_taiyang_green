import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up1D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv1D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, down_size):
        super(OutConv1D, self).__init__()
        # insize = in_channels * down_size
        # midsize = math.ceil(insize / 2)
        # outsize = out_channels *
        # self.conv = nn.Sequential(
        #     nn.Linear(, out_channels * down_size),
        #     nn.Linear(in_channels * down_size, out_channels * down_size),
        # )
        self.conv = nn.Sequential(
            nn.AvgPool1d(down_size),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, n_channels, n_classes, epoch_num, one_epoch_len, bilinear=True):
        super(UNet1D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv1D(n_channels, 64)
        # self.down1 = Down1D(64, 128)
        # self.down2 = Down1D(128, 256)
        # self.down3 = Down1D(256, 512)
        # self.down4 = Down1D(512, 512)
        # self.up1 = Up1D(1024, 256, bilinear)
        # self.up2 = Up1D(512, 128, bilinear)
        # self.up3 = Up1D(256, 64, bilinear)
        # self.up4 = Up1D(128, 64, bilinear)
        # self.outc = OutConv1D(64, n_classes, one_epoch_len)

        self.inc = DoubleConv1D(n_channels, 32)
        self.down1 = Down1D(32, 64)
        self.down2 = Down1D(64, 128)
        self.down3 = Down1D(128, 256)
        self.down4 = Down1D(256, 256)
        self.up1 = Up1D(512, 128, bilinear)
        self.up2 = Up1D(256, 64, bilinear)
        self.up3 = Up1D(128, 32, bilinear)
        self.up4 = Up1D(64, 16, bilinear)
        self.outc = OutConv1D(16, n_classes, one_epoch_len)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits