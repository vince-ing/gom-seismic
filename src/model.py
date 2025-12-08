"""
src/model.py
Reconstructed to match best_model.pth structure:
- Flattened Decoder (self.up1, self.conv1 separated)
- Symmetric Channels (128 output at layer 3)
- DoubleConv uses internal Sequential
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        # Direct Sequential instead of using DoubleConv class
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- ENCODER ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # --- DECODER (Flattened Style) ---
        # Up 1
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        
        # Up 2
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        
        # Up 3
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        # Up 4
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        # Output
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder Path
        x = self.up1(x5)
        x = self._pad_and_cat(x4, x)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = self._pad_and_cat(x3, x)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = self._pad_and_cat(x2, x)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = self._pad_and_cat(x1, x)
        x = self.conv4(x)
        
        logits = self.outc(x)
        return logits

    def _pad_and_cat(self, target, x):
        diffY = target.size()[2] - x.size()[2]
        diffX = target.size()[3] - x.size()[3]
        if diffX > 0 or diffY > 0:
             x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                           diffY // 2, diffY - diffY // 2])
        return torch.cat([target, x], dim=1)