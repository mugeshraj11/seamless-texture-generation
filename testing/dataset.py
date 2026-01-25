import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class TileDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        if transform is None:
            self.transform = transforms.Compose([
                # --- Aggressive Augmentation for Small Dataset ---
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # End of Augmentations

                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image

if __name__ == '__main__':
    # Example usage and verification
    DATA_DIR = "C:\\Users\\muges\\GAME DEV\\Textures\\floor\\seamless_tiles_32" # Make sure this path is correct
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' not found. Please create it and add images.")
    else:
        dataset = TileDataset(DATA_DIR)
        print(f"Loaded {len(dataset)} images from {DATA_DIR}")
        sample_img = dataset[0]
        print(f"Sample image tensor shape: {sample_img.shape}, dtype: {sample_img.dtype}")
        print(f"Sample image min/max pixel value: {sample_img.min():.2f}/{sample_img.max():.2f}")

        # Check for image size
        if sample_img.shape[1] != 32 or sample_img.shape[2] != 32:
            print("WARNING: Images might not be 32x32. GAN architecture assumes 32x32 output.")