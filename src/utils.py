from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import DataLoader

def load_train_data(train_root, batch_size=128):



        train_transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.RandomCrop((128, 128)),
            transforms.RandomRotation(degrees=15), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])

        data = ImageFolder(root = train_root, transform=train_transform)

        train_size = int(0.8 * len(data))
        val_size = len(data) - train_size

        train_data, val_data = random_split(data, [train_size, val_size])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        return train_loader, val_loader

def load_test_data(test_root, batch_size=128):
    test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    test_data = ImageFolder(root = test_root, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return test_loader
