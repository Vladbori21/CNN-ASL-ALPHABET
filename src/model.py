import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_part = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.mlp_part = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self._conv_output(), out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=29)
        )

    def forward(self, x):
        x = self.conv_part(input)
        x = self.mlp_part(x)
        return x


    def _conv_output(self):
        x = torch.randn(1, 3, 128, 128)
        output = self.conv_part(x)

        _, c, h, w = output.shape

        return c*h*w

    def my_train(self, epochs, print_every=1):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.0001)

        train_loader = DataLoader(self.train_data, batch_size=128, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=128)

        for epoch in range(epochs):
            train_losses = []
            train_acc = []
            val_acc = []

            self.train()
            for images, labels in train_loader:
                output = self(images)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = torch.argmax(output)
                acc = torch.mean(preds == labels).item()
                train_acc.append(acc)
                train_losses.append(loss.item())


            if (epoch + 1) % print_every == 0:

                self.eval()
                for images, labels in val_loader:
                    output = self(images)

                    preds = torch.argmax(output)
                    acc = torch.mean(preds == labels).item()
                    val_acc.append(acc)

                t_acc = np.mean(train_acc)
                v_acc = np.mean(val_acc)
                t_loss = np.mean(train_losses)
                print(f'epoch: {epoch+1} loss: {t_loss:.4f} train acc: {t_acc:.4f} val acc: {v_acc:.4f}')





    def load_data(self, train_root, test_root):



        train_transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.RandomCrop((128, 128)),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        self.test_data = ImageFolder(root=test_root, transform=test_transform)
        data = ImageFolder(root = train_root, transform=train_transform)

        train_size = int(0.8 * len(data))
        val_size = len(data) - train_size

        self.train_data, self.val_data = random_split(data, [train_size, val_size])
