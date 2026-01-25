import torch
import os
from PIL import Image
import torchvision.transforms as T
import numpy as np

from model import Autoencoder

MODEL_PATH = "outputs/baseline_B/autoencoder/autoencoder.pth"
OUTPUT_DIR = "outputs/baseline_B/autoencoder"
NUM_SAMPLES = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

to_tensor = T.ToTensor()
to_img = T.ToPILImage()

# sample latent space by encoding random training tiles
tiles = [
    Image.open(os.path.join("Textures/floor/seamless_tiles_32", f)).convert("RGB")
    for f in os.listdir("Textures/floor/seamless_tiles_32")
]

for i in range(NUM_SAMPLES):
    img = to_tensor(tiles[i % len(tiles)]).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)[0]

    to_img(out.cpu()).save(f"{OUTPUT_DIR}/ae_{i:03d}.png")

print("Autoencoder samples generated.")
