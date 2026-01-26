import torch
import os
from torchvision.utils import save_image

from model import Generator
from config import Z_DIM

NUM_SAMPLES = 100
BEST_EPOCH = 2600

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load generator
G = Generator().to(device)

checkpoint = torch.load(
    f"wgan_with_bc_loss/checkpoints/wgan_epoch_{BEST_EPOCH}.pth",
    map_location=device
)

G.load_state_dict(checkpoint["generator"])
G.eval()

OUT_DIR = f"wgan_with_bc_loss/final_samples_{BEST_EPOCH}_wgan_lambda3"
os.makedirs(OUT_DIR, exist_ok=True)

with torch.no_grad():
    for i in range(NUM_SAMPLES):
        z = torch.randn(1, Z_DIM, 1, 1, device=device)
        img = (G(z) + 1) / 2
        save_image(img, f"{OUT_DIR}/gan_{i:03d}.png")

print("Final GAN samples generated.")
