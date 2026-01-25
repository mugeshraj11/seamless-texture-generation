import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image

# --- Make sure these imports are correct ---
from dataset import TileDataset
from gan_modified import Generator, Discriminator # <<< IMPORTANT: Use gan_modified

CKPT_DIR = "outputs/gan/checkpoints_seamless" # New checkpoint directory
os.makedirs(CKPT_DIR, exist_ok=True)

# seam_loss: Penalizes mismatch between opposite edges
def seam_loss(img, w_lr=1.0, w_tb=1.0):
    left = img[:, :, :, 0]
    right = img[:, :, :, -1]
    top = img[:, :, 0, :]
    bottom = img[:, :, -1, :]

    lr = ((left - right) ** 2).mean()
    tb = ((top - bottom) ** 2).mean()
    return w_lr * lr + w_tb * tb


DATA_DIR = "C:\\Users\\muges\\GAME DEV\\Textures\\floor\\seamless_tiles_32"
OUT_DIR = "outputs/gan/samples_seamless"      # Output for generated samples
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 2000 # Increased epochs for small dataset
Z_DIM = 100
LR = 2e-4
LAMBDA_SEAM = 20.0 # <<< Increased weight for seam loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TileDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # drop_last for consistent batch size

G = Generator(Z_DIM).to(device)
D = Discriminator().to(device)

opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

criterion = nn.BCEWithLogitsLoss()

print(f"Starting training on {len(dataset)} images with {EPOCHS} epochs.")
print(f"Using device: {device}")

for epoch in range(EPOCHS):
    for batch_idx, real in enumerate(loader):
        real = real.to(device)
        bsz = real.size(0)

        # -------- Train Discriminator --------
        D.zero_grad()
        
        # Real images
        label_real = torch.full((bsz,), 1.0, device=device)
        output_real = D(real)
        loss_D_real = criterion(output_real, label_real)
        
        # Fake images
        noise = torch.randn(bsz, Z_DIM, 1, 1, device=device)
        fake = G(noise) # Generate fake images
        
        label_fake = torch.full((bsz,), 0.0, device=device)
        output_fake = D(fake.detach()) # Detach G from D training
        loss_D_fake = criterion(output_fake, label_fake)
        
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        opt_D.step()

        # -------- Train Generator (WITH SEAM LOSS) --------
        G.zero_grad()
        
        label_gen = torch.full((bsz,), 1.0, device=device) # Generator wants D to think fakes are real
        output_gen = D(fake)
        adv_loss = criterion(output_gen, label_gen)
        
        s_loss = seam_loss(fake, w_lr=1.0, w_tb=1.0) # Calculate seam loss on the generated image
        loss_G = adv_loss + LAMBDA_SEAM * s_loss
        
        loss_G.backward()
        opt_G.step()

    # --- Save Sample and Checkpoint ---
    if epoch % 50 == 0 or epoch == EPOCHS - 1: # Save more often or at end for debugging
        with torch.no_grad():
            G.eval() # Set generator to evaluation mode
            sample_noise = torch.randn(1, Z_DIM, 1, 1, device=device)
            single_generated_tile = (G(sample_noise) + 1) / 2 # Normalize to [0, 1]

            # Create a 2x2 grid of the SAME tile to visually check for seamlessness
            tiled_output_image = single_generated_tile.repeat(1, 1, 8, 8)
            save_image(tiled_output_image, f"{OUT_DIR}/epoch_{epoch}_tiled.png")
            
            G.train() # Set generator back to training mode

        torch.save(G.state_dict(), f"{CKPT_DIR}/generator_epoch_{epoch}.pth")
        print(
            f"Epoch {epoch}/{EPOCHS}: "
            f"D_Loss={loss_D.item():.4f}, "
            f"G_Loss={loss_G.item():.4f}, "
            f"Adv_Loss={adv_loss.item():.4f}, "
            f"Seam_Loss={s_loss.item():.4f}"
        )

torch.save(G.state_dict(), f"{CKPT_DIR}/generator_final.pth")
print("GAN training complete.")