import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes: int = 10, drop_rate: float = 0.0):
        super(Net, self).__init__()

        self.dropout = nn.Dropout(p=drop_rate)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        # DONT MODIFY THE ORDER OF TRAINABLE LAYERS - in order to keep track of layer execution
        # CNN layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 16, kernel_size=5, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(4 * 4 * self.conv2.out_channels, 120)
        self.fc2 = nn.Linear(self.fc1.out_features, 84)
        self.fc3 = nn.Linear(self.fc2.out_features, num_classes)


    def forward(self, x):

        x = self.avgpool1(self.relu(self.conv1(x)))
        x = self.avgpool2(self.relu(self.conv2(x)))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu((self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    m = Net()
    print(count_parameters(m))
