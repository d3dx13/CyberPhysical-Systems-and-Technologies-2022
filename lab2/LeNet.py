import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = (28, 28)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=5 // 2).cuda(),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0).cuda(),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
        )

        self.linear_layers = 16 * 5 * 5

        # linear layers
        self.fc = nn.Sequential(
            nn.Linear(self.linear_layers, 120).cuda(),
            nn.Sigmoid(),
            nn.Linear(120, 84).cuda(),
            nn.Sigmoid(),
            nn.Linear(84, 10).cuda(),
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
