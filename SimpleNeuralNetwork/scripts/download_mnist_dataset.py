import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


def get_mnist_train_data():
    return datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=Lambda(
            lambda y:
                torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
        )
    )


def get_mnist_test_data():
    return datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )


def create_dataloader(batch_size):
    # Download data for training and testing model
    training_data = get_mnist_train_data()
    test_data = get_mnist_test_data()

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader
