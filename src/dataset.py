from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATA_DIR = Path("data/raw/bloodcells_dataset")
IMAGE_SIZE = 224
BATCH_SIZE = 32
RANDOM_SEED = 42


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def create_dataloaders():
    train_dataset_full = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(DATA_DIR, transform=validation_transform)

    total_size = len(train_dataset_full)
    train_size = int(total_size * 0.8)

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_dataset_full, train_indices)
    validation_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, validation_loader, train_dataset_full.classes


if __name__ == "__main__":
    train_loader, validation_loader, class_names = create_dataloaders()
    print(f"Classes: {class_names}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")