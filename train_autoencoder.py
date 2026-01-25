import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from dataset import TileDataset
from model import Autoencoder

DATA_DIR = "Textures/floor/seamless_tiles_32"
OUTPUT_DIR = "outputs/baseline_B/autoencoder"
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-3

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TileDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    total_loss = 0
    for x in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), f"{OUTPUT_DIR}/autoencoder.pth")
print("Training complete.")
