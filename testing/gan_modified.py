import torch
import torch.nn as nn
import torch.nn.functional as F

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        # Initial block: Z_DIM -> 256 @ 4x4
        # Stays ConvTranspose2d here to go from 1x1 latent to spatial feature map
        self.initial_block = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, kernel_size=4, stride=1, padding=0), # Output: (B, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # Upsampling blocks: Use Upsample + Conv2d with circular padding
        # Block 1: 256 @ 4x4 -> 128 @ 8x8
        self.up_block1 = self._make_upsample_block(256, 128) # Output: (B, 128, 8, 8)
        # Block 2: 128 @ 8x8 -> 64 @ 16x16
        self.up_block2 = self._make_upsample_block(128, 64)  # Output: (B, 64, 16, 16)
        # Block 3: 64 @ 16x16 -> 3 @ 32x32
        self.up_block3 = self._make_upsample_block(64, 3, final_layer=True) # Output: (B, 3, 32, 32)

    def _make_upsample_block(self, in_channels, out_channels, final_layer=False):
        layers = [
            nn.Upsample(scale_factor=2, mode='nearest'), # Upscale features
            # Use Conv2d with circular padding for seamlessness
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        ]
        if not final_layer:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
        else:
            layers.append(nn.Tanh()) # Output activation for image in [-1, 1]
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.initial_block(z)      # Z (1,1) -> (4,4)
        x = self.up_block1(x)          # (4,4) -> (8,8)
        x = self.up_block2(x)          # (8,8) -> (16,16)
        x = self.up_block3(x)          # (16,16) -> (32,32)
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # All Conv2d layers use padding_mode='circular'
        # This makes the Discriminator evaluate the image as if it loops around
        self.net = nn.Sequential(
            # Input: (B, 3, 32, 32) -> (B, 64, 16, 16)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, padding_mode="circular"),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 64, 16, 16) -> (B, 128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode="circular"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 128, 8, 8) -> (B, 256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode="circular"),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 256, 4, 4) -> (B, 1, 4, 4)
            # PatchGAN style: outputs a grid of scores, not a single scalar
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, padding_mode="circular") # No padding here for 4x4 output
        )

    def forward(self, x):
        # The Discriminator will output (B, 1, 4, 4)
        # We take the mean across the spatial dimensions to get a single score per image.
        out = self.net(x)
        return out.mean(dim=[2, 3]).view(-1) # Flatten to (B) for BCEWithLogitsLoss