import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

def circular_pad(x, pad):
    # pad = (left, right, top, bottom)
    return F.pad(x, pad, mode="circular")

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        # 1x1 to 4x4
        self.deconv1 = nn.ConvTranspose2d(z_dim, 256, 4, 1, 0) 
        self.bn1 = nn.BatchNorm2d(256)

        # Instead of ConvTranspose, use Upsample + Conv
        self.block2 = self._make_layer(256, 128) # 4x4 -> 8x8
        self.block3 = self._make_layer(128, 64)  # 8x8 -> 16x16
        self.block4 = self._make_layer(64, 3, final_layer=True) # 16x16 -> 32x32

    def _make_layer(self, in_channels, out_channels, final_layer=False):
        layers = [nn.Upsample(scale_factor=2, mode='nearest')]
        # Use standard Conv2d with circular padding mode
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular'))
        if not final_layer:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
        else:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, z):
        x = torch.relu(self.bn1(self.deconv1(z)))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, padding_mode="circular"),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, padding_mode="circular"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, padding_mode="circular"),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 3, 1, 1, padding_mode="circular")
        )

    def forward(self, x):
        out = self.net(x)              # (B, 1, H, W)
        return out.mean(dim=[2, 3]).view(-1)