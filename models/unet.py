import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Double convolution block with Batch Normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    The U-Net architecture with encoder and decoder paths with skip connections.
    """
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNet, self).__init__()
        # Encoder
        self.encoder1 = DoubleConv(in_channels, features)            # Input: 3 x 128 x 512, Output: features x 128 x 512
        self.pool1    = nn.MaxPool2d(kernel_size=2, stride=2)        # Downsample: features x 128 x 512 -> features x 64 x 256
        self.encoder2 = DoubleConv(features, features * 2)           # Input: features x 64 x 256, Output: (features*2) x 64 x 256
        self.pool2    = nn.MaxPool2d(kernel_size=2, stride=2)        # Downsample: (features*2) x 64 x 256 -> (features*2) x 32 x 128
        self.encoder3 = DoubleConv(features * 2, features * 4)       # Input: (features*2) x 32 x 128, Output: (features*4) x 32 x 128
        self.pool3    = nn.MaxPool2d(kernel_size=2, stride=2)        # Downsample: (features*4) x 32 x 128 -> (features*4) x 16 x 64
        self.encoder4 = DoubleConv(features * 4, features * 8)       # Input: (features*4) x 16 x 64, Output: (features*8) x 16 x 64
        self.pool4    = nn.MaxPool2d(kernel_size=2, stride=2)        # Downsample: (features*8) x 16 x 64 -> (features*8) x 8 x 32

        # Bottleneck
        self.bottleneck = DoubleConv(features * 8, features * 16)    # Input: (features*8) x 8 x 32, Output: (features*16) x 8 x 32

        # Decoder
        self.upconv4  = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)    # Upsample: (features*16) x 8 x 32 -> (features*8) x 16 x 64
        self.decoder4 = DoubleConv((features * 8) * 2, features * 8)                                # Input: (features*16) x 16 x 64, Output: (features*8) x 16 x 64
        self.upconv3  = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)     # Upsample: (features*8) x 16 x 64 -> (features*4) x 32 x 128
        self.decoder3 = DoubleConv((features * 4) * 2, features * 4)                                # Input: (features*8) x 32 x 128, Output: (features*4) x 32 x 128
        self.upconv2  = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)     # Upsample: (features*4) x 32 x 128 -> (features*2) x 64 x 256
        self.decoder2 = DoubleConv((features * 2) * 2, features * 2)                                # Input: (features*4) x 64 x 256, Output: (features*2) x 64 x 256
        self.upconv1  = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)         # Upsample: (features*2) x 64 x 256 -> features x 128 x 512
        self.decoder1 = DoubleConv(features * 2, features)                                          # Input: (features*2) x 128 x 512, Output: features x 128 x 512

        # Final output layer
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)  # Output: out_channels x 128 x 512

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)                          # features x 128 x 512
        enc2 = self.encoder2(self.pool1(enc1))           # (features*2) x 64 x 256
        enc3 = self.encoder3(self.pool2(enc2))           # (features*4) x 32 x 128
        enc4 = self.encoder4(self.pool3(enc3))           # (features*8) x 16 x 64

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))   # (features*16) x 8 x 32

        # Decoder path
        dec4 = self.upconv4(bottleneck)                  # (features*8) x 16 x 64
        dec4 = torch.cat((dec4, enc4), dim=1)            # Concatenate with encoder4: (features*16) x 16 x 64
        dec4 = self.decoder4(dec4)                       # (features*8) x 16 x 64

        dec3 = self.upconv3(dec4)                        # (features*4) x 32 x 128
        dec3 = torch.cat((dec3, enc3), dim=1)            # Concatenate with encoder3: (features*8) x 32 x 128
        dec3 = self.decoder3(dec3)                       # (features*4) x 32 x 128

        dec2 = self.upconv2(dec3)                        # (features*2) x 64 x 256
        dec2 = torch.cat((dec2, enc2), dim=1)            # Concatenate with encoder2: (features*4) x 64 x 256
        dec2 = self.decoder2(dec2)                       # (features*2) x 64 x 256

        dec1 = self.upconv1(dec2)                        # features x 128 x 512
        dec1 = torch.cat((dec1, enc1), dim=1)            # Concatenate with encoder1: (features*2) x 128 x 512
        dec1 = self.decoder1(dec1)                       # features x 128 x 512

        # Output layer
        return self.conv(dec1)                           # Output: out_channels x 128 x 512
