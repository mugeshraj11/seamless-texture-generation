import torch
from dcgan import Generator
import os
from torchvision.utils import save_image

Z_DIM = 100
NUM_SAMPLES = 100
BEST_EPOCH = 450  # <<< choose based on visual + seam tradeoff

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator(Z_DIM).to(device)

G.load_state_dict(
    torch.load(
        f"outputs/dcgan/checkpoints_dcgan/generator_epoch_{BEST_EPOCH}.pth",
        map_location=device
    )
)

G.eval()

OUT_DIR = f"outputs/dcgan/final_samples_{BEST_EPOCH}"
os.makedirs(OUT_DIR, exist_ok=True)

with torch.no_grad():
    for i in range(NUM_SAMPLES):
        z = torch.randn(1, Z_DIM, 1, 1, device=device)
        img = (G(z) + 1) / 2
        save_image(img, f"{OUT_DIR}/gan_{i:03d}.png")

print("Final GAN samples generated.")
