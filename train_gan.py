import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from dataset import TileDataset
from dcgan import Generator, Discriminator

CKPT_DIR = "outputs/dcgan/checkpoints_dcgan"
os.makedirs(CKPT_DIR, exist_ok=True)
def seam_loss(img):
    """
    Penalizes mismatch between opposite edges.
    img: (B, 3, H, W) in [-1, 1]
    """
    left = img[:, :, :, 0]
    right = img[:, :, :, -1]
    top = img[:, :, 0, :]
    bottom = img[:, :, -1, :]
    return ((left - right) ** 2).mean() + ((top - bottom) ** 2).mean()


DATA_DIR = "Textures/floor/seamless_tiles_32"
OUT_DIR = "outputs/dcgan/samples_dcgan"

BATCH_SIZE = 8
EPOCHS = 500
Z_DIM = 100
LR = 2e-4
LAMBDA_SEAM = 10.0   # <<< NEW (try 5, 10, or 15)

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TileDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G = Generator(Z_DIM).to(device)
D = Discriminator().to(device)

opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

criterion = nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    for real in loader:
        real = real.to(device)
        bsz = real.size(0)

        # -------- Train Discriminator --------
        noise = torch.randn(bsz, Z_DIM, 1, 1, device=device)
        fake = G(noise).detach()

        loss_D = (
            criterion(D(real), torch.ones(bsz, device=device)) +
            criterion(D(fake), torch.zeros(bsz, device=device))
        )

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # -------- Train Generator (WITH SEAM LOSS) --------
        noise = torch.randn(bsz, Z_DIM, 1, 1, device=device)
        fake = G(noise)

        adv_loss = criterion(D(fake), torch.ones(bsz, device=device))
        s_loss = seam_loss(fake)                     # <<< NEW
        loss_G = adv_loss + LAMBDA_SEAM * s_loss     # <<< NEW

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    if epoch % 50 == 0:
        with torch.no_grad():
            sample = (G(torch.randn(1, Z_DIM, 1, 1, device=device)) + 1) / 2
            from torchvision.utils import save_image
            save_image(sample, f"{OUT_DIR}/epoch_{epoch}.png")

    # 🔥 SAVE CHECKPOINT
        torch.save(
        G.state_dict(),
        f"{CKPT_DIR}/generator_epoch_{epoch}.pth"
    )

        print(
        f"Epoch {epoch}: "
        f"D={loss_D.item():.3f}, "
        f"G={loss_G.item():.3f}, "
        f"Seam={s_loss.item():.4f}"
    )


torch.save(G.state_dict(), "outputs/dcgan/generator.pth")
print("GAN training complete.")
