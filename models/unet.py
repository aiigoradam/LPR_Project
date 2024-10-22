import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    A module consisting of two consecutive convolutional layers,
    each followed by batch normalization and ReLU activation.
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
    The U-Net architecture with encoder and decoder paths.
    """
    def __init__(self, in_channels=3, out_channels=3, init_features=64):
        super(UNet, self).__init__()
        features = init_features

        # Encoder
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(features * 8, features * 16)

        # Decoder
        self.upconv4, self.decoder4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2), DoubleConv((features * 8) * 2, features * 8)
        self.upconv3, self.decoder3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2), DoubleConv((features * 4) * 2, features * 4)
        self.upconv2, self.decoder2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2), DoubleConv((features * 2) * 2, features * 2)
        self.upconv1, self.decoder1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2), DoubleConv(features * 2, features)

        # Final output layer
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder path
        dec4 = self.decoder4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.decoder3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        # Output layer
        return self.conv(dec1)
