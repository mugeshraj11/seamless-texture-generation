import torch
import os
from torchvision.utils import save_image

# --- Make sure this import is correct ---
from gan_modified import Generator # <<< IMPORTANT: Use gan_modified

Z_DIM = 100
NUM_SAMPLES = 50 # Generate 50 new samples
BEST_EPOCH = 1950 # Choose the epoch that looks best visually from your samples_seamless folder
                  # Or use 'final' for the last saved one: BEST_EPOCH = "final"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator(Z_DIM).to(device)

# Load the desired checkpoint
if isinstance(BEST_EPOCH, int):
    checkpoint_path = f"outputs/gan/checkpoints_seamless/generator_epoch_{BEST_EPOCH}.pth"
else: # assuming "final"
    checkpoint_path = f"outputs/gan/checkpoints_seamless/generator_final.pth"

try:
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded generator from {checkpoint_path}")
except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {checkpoint_path}. Please check BEST_EPOCH or path.")
    exit()
except RuntimeError as e:
    print(f"RuntimeError loading state_dict: {e}")
    print("This often means your Generator class definition does not match the saved checkpoint.")
    print("Please ensure your 'gan_modified.py' matches the code used to train this checkpoint.")
    exit()

G.eval() # Set generator to evaluation mode (important for BatchNorm and Dropout)

OUT_DIR = "outputs/gan/final_seamless_samples" # Output folder for final generated images
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Generating {NUM_SAMPLES} final seamless samples...")
with torch.no_grad():
    for i in range(NUM_SAMPLES):
        z = torch.randn(1, Z_DIM, 1, 1, device=device)
        single_tile = (G(z) + 1) / 2 # Generated image is in [-1, 1], convert to [0, 1]
        
        # Save the single tile
        save_image(single_tile, f"{OUT_DIR}/seamless_gan_{i:03d}.png")
        
        # Optionally, save a tiled version for quick visual verification
        tiled_for_display = single_tile.repeat(1, 1, 4, 4) # 4x4 grid of the same tile
        save_image(tiled_for_display, f"{OUT_DIR}/seamless_gan_{i:03d}_tiled.png")

print("Final GAN samples generated.")