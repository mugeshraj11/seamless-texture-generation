import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class Random90Rotation:
    def __call__(self, img):
        angle = random.choice([0,90,180,270])
        return img.rotate(angle)

class TileDataset(Dataset):
    def __init__(self, root):
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(".png")
        ]
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            Random90Rotation(),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)  # [0,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)
