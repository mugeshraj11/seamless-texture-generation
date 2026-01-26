import os
import torch
from torchvision.utils import save_image

from model import Generator
from config import Z_DIM, DEVICE

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# ---------------- CONFIG ----------------
CHECKPOINT_DIR = "wgan_with_bc_loss/checkpoints"
OUTPUT_DIR = "wgan_with_bc_loss/epoch_samples"
CHECKPOINT_INTERVAL = 200      # must match training
START_EPOCH = 0
END_EPOCH = 3000               # inclusive
# ---------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed latent vector (VERY IMPORTANT for comparison)
torch.manual_seed(42)
fixed_z = torch.randn(1, Z_DIM, 1, 1, device=device)

gen = Generator().to(device)
gen.eval()

for epoch in range(START_EPOCH, END_EPOCH + 1, CHECKPOINT_INTERVAL):
    ckpt_path = os.path.join(
        CHECKPOINT_DIR,
        f"wgan_epoch_{epoch}.pth"
    )

    if not os.path.exists(ckpt_path):
        print(f"[SKIP] Checkpoint not found: {ckpt_path}")
        continue

    checkpoint = torch.load(ckpt_path, map_location=device)
    gen.load_state_dict(checkpoint["generator"])

    with torch.no_grad():
        fake = gen(fixed_z)

    out_path = os.path.join(
        OUTPUT_DIR,
        f"sample_epoch_{epoch}.png"
    )

    save_image(fake, out_path, normalize=True)
    print(f"[OK] Saved {out_path}")

print("All samples generated.")
