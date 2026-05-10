# dataset.py
import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
 
EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]
 
def get_transforms(train=True):
    base = [
        transforms.Grayscale(num_output_channels=3),  # 1ch -> 3ch
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ]
    if train:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)
 
def get_loaders(data_dir="data", batch_size=32):
    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=get_transforms(train=True)
    )
    full_test = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=get_transforms(train=False)
    )
    val_size = len(full_test) // 2
    test_size = len(full_test) - val_size
    val_ds, test_ds = random_split(full_test, [val_size, test_size])
 
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                             shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                             shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                             shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader, train_ds.classes
