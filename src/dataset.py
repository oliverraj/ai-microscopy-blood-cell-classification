from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DATA_DIR = Path("data/raw/bloodcells_dataset")

IMAGE_SIZE = 224
BATCH_SIZE = 32
RANDOM_SEED = 42


train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


validation_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def create_dataloaders():

    dataset = datasets.ImageFolder(
        root=DATA_DIR,
        transform=train_transform,
    )

    class_names = dataset.classes

    total_size = len(dataset)

    train_size = int(total_size * 0.8)
    validation_size = total_size - train_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=None,
    )

    validation_dataset.dataset.transform = validation_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return (
        train_loader,
        validation_loader,
        class_names,
    )


if __name__ == "__main__":

    train_loader, validation_loader, class_names = create_dataloaders()

    print(f"Classes: {class_names}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")