import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super(UNetGenerator, self).__init__()

        # Encoder (downsampling)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, ngf, kernel_size=4, stride=2, padding=1)
        )  # No BatchNorm or ReLU in the first layer

        self.down2 = self.down_block(ngf, ngf * 2)
        self.down3 = self.down_block(ngf * 2, ngf * 4)
        self.down4 = self.down_block(ngf * 4, ngf * 8)
        self.down5 = self.down_block(ngf * 8, ngf * 8)
        self.down6 = self.down_block(ngf * 8, ngf * 8, batch_norm=False)  # Bottleneck (bottom layer)

        # Decoder (upsampling)
        self.up1 = self.up_block(ngf * 8, ngf * 8, dropout=True)
        self.up2 = self.up_block(ngf * 16, ngf * 8, dropout=True)
        self.up3 = self.up_block(ngf * 16, ngf * 4)
        self.up4 = self.up_block(ngf * 8, ngf * 2)
        self.up5 = self.up_block(ngf * 4, ngf)
        self.up6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def down_block(self, in_channels, out_channels, batch_norm=True):
        layers = [nn.LeakyReLU(0.2, inplace=True),
                  nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def up_block(self, in_channels, out_channels, dropout=False):
        layers = [nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(out_channels)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)  # Bottleneck

        # Decoder with skip connections
        u1 = self.up1(d6)
        u1 = torch.cat([u1, d5], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d4], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d3], dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d2], dim=1)

        u5 = self.up5(u4)
        u5 = torch.cat([u5, d1], dim=1)

        output = self.up6(u5)  # Final output layer

        return output


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # Layer 1: No BatchNorm
            nn.Conv2d(in_channels * 2, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output Layer
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, input_image, target_image):
        # Concatenate input and target images along the channel dimension
        x = torch.cat([input_image, target_image], dim=1)
        return self.model(x)
