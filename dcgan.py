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

        self.deconv1 = nn.ConvTranspose2d(z_dim, 256, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

    def forward(self, z):
        x = self.deconv1(z)
        x = self.bn1(x)
        x = torch.relu(x)

        x = circular_pad(x, (1, 1, 1, 1))
        x = self.deconv2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = circular_pad(x, (1, 1, 1, 1))
        x = self.deconv3(x)
        x = self.bn3(x)
        x = torch.relu(x)

        x = circular_pad(x, (1, 1, 1, 1))
        x = self.deconv4(x)

        return torch.tanh(x)


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