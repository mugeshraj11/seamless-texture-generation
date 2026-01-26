import torch
from torchvision.utils import save_image

from model import Generator
from config import Z_DIM, DEVICE

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# -------- CHANGE THIS --------
CHECKPOINT_PATH = "wgan/checkpoints/wgan_epoch_1600.pth"
OUTPUT_FILE = "wgan/epoch_samples/generated_1600.png"
# ----------------------------

# Load generator
gen = Generator().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
gen.load_state_dict(checkpoint["generator"])
gen.eval()

# Sample ONE image
z = torch.randn(1, Z_DIM, 1, 1, device=device)
with torch.no_grad():
    fake = gen(z)

save_image(fake, OUTPUT_FILE, normalize=True)
print(f"Saved generated image to {OUTPUT_FILE}")
