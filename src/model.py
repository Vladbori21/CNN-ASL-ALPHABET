import torch
from torch import nn
import torch.optim as optim
import numpy as np


class CNN(nn.Module):

    def __init__(self, device=torch.device('cpu')):
        super().__init__()
        self.device = device

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

        self.to(self.device)

    def forward(self, x):
        x = self.conv_part(x)
        x = self.mlp_part(x)
        return x


    def _conv_output(self):
        x = torch.randn(1, 3, 128, 128)
        output = self.conv_part(x)

        _, c, h, w = output.shape

        return c*h*w
    
    def test(self, test_loader):
        test_acc = []

        self.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self(images)

                preds = torch.argmax(output, dim=1)
                acc = torch.mean((preds == labels).float()).item()
                test_acc.append(acc)

        t_acc = np.mean(test_acc)
        print(f'test accuracy: {t_acc}')


    def my_train(self, train_loader, val_loader, epochs, print_every=1):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.0001)

        

        for epoch in range(epochs):
            train_losses = []
            train_acc = []
            

            self.train()
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self(images)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = torch.argmax(output, dim=1)
                acc = torch.mean((preds == labels).float()).item()
                train_acc.append(acc)
                train_losses.append(loss.item())


            if (epoch + 1) % print_every == 0:
                val_acc = []

                self.eval()
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        output = self(images)

                        preds = torch.argmax(output, dim=1)
                        acc = torch.mean((preds == labels).float()).item()
                        val_acc.append(acc)

                t_acc = np.mean(train_acc)
                v_acc = np.mean(val_acc)
                t_loss = np.mean(train_losses)
                print(f'epoch: {epoch+1} loss: {t_loss:.4f} train acc: {t_acc:.4f} val acc: {v_acc:.4f}')