from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(data_dir, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    return loader
