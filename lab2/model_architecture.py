import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = (64, 64)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=15, padding=15 // 2).cuda(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=11, padding=11 // 2).cuda(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=7, padding=7 // 2).cuda(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, padding=5 // 2).cuda(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=3 // 2).cuda(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.linear_layers = 64 * 2 * 2

        # linear layers
        self.fc = nn.Sequential(
            nn.Linear(self.linear_layers, 4096).cuda(),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 128).cuda(),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10).cuda(),
        )

        self.printed_size = False

    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.conv(x)

        if not self.printed_size:
            print(x.size())
            self.printed_size = True
        x = x.flatten(start_dim=1)

        # linear layers
        x = self.fc(x)
        return x
