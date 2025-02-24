import os
import pdb
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch_intermediate_layer_getter import IntermediateLayerGetter
from torchvision.datasets import MNIST
from tqdm import tqdm

from utils.util import z_score_normalization


class Lenet5(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.0):
        super(Lenet5, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc_1 = nn.Linear(4 * 4 * 16, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.relu = nn.ReLU()
        self.fc_3 = nn.Linear(84, num_classes)
        self.dropout = drop_rate
        self.num_classes = 10
        print(f"Model Name: LeNet-5.")
        self.model_path = os.path.join("models/lenet5/lenet5_mnist.pt")

    def forward(self, x):
        """
        :param x: [b, 1, 28, 28]
        :return:
        """
        batch_size = x.shape[0]
        x = self.conv_1(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)

        x = self.conv_2(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)
        x = x.view(batch_size, 4 * 4 * 16)

        x = self.fc_1(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)

        x = self.fc_2(self.relu(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout)

        x = self.fc_3(self.relu(x))

        return x

    def intermediate_outputs(self, data):
        """
        generates outputs for the selected intermediate layers and save the layers names and number of neurons as class
        variable.
        NOTE: This function should be called in each run at the beginning for the fault detection to work.
        expected_intermediate_layers = ['conv_1', 'conv_2', 'fc_1', 'fc_2', 'fc_3']
        expected_params = [6, 16, 120, 84, 10]
        :param data: any input data
        :return: intermediate layer outputs
        """
        layers = self.named_children()
        intermediate_layers = OrderedDict()
        self.num_neurons = []
        for layer in layers:
            p = list(layer[1].parameters())
            if not len(p) == 0:
                self.num_neurons.append(list(layer[1].parameters())[-1].size(0))
                intermediate_layers.update({layer[0]: layer[0]})
        self.intermediate_layers = list(intermediate_layers.keys())
        assert len(self.num_neurons) == len(self.intermediate_layers), "Intermediate layer identification going wrong"
        intermediate_getter = IntermediateLayerGetter(model=self, return_layers=intermediate_layers, keep_output=True)
        mids, _ = intermediate_getter(data)
        return mids

    def good_seed_detection(self, test_loader, predictor, config):
        """
        from the test set detect seeds which can be considered as initial test seeds for fault detection
        (ones with correct prediction)
        :param test_loader:
        :param predictor: method for prediction
        :return: indexes of seeds to be considered
        """
        idx_increment = 0
        idx_good = np.array([], dtype=np.int)
        for data in test_loader:
            test_set, labels = data
            data = z_score_normalization(test_set)
            _, result = predictor(self, data, config.network_type, config, config.normalize_data)
            result = result.to("cpu")
            idx_good = np.append(idx_good, np.where(labels == result)[0] + idx_increment)
            idx_increment += test_loader.batch_size
        return idx_good


def train(model, train_loader):
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    opt = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-3)

    model.train()
    for epoch in range(50):
        total_loss, total_acc = 0, 0
        for i, batch in enumerate(tqdm(train_loader)):
            data, label = batch
            data, label = data.to(device), label.to(device)

            pred = model(data)
            loss = loss_fn(input=pred, target=label)
            total_loss += loss.item()

            acc = (label == torch.argmax(pred, dim=-1)).float().mean()
            total_acc += acc.item()

            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f"Epoch : {epoch}, Loss : {total_loss / i}, Accuracy : {total_acc / i}")

    torch.save(model.state_dict(), "lenet5_mnist.pt")


def test(model, test_loader):
    model.eval()
    x_test, y_test = next(iter(test_loader))
    with torch.no_grad():
        images, y_test = x_test.to(device), y_test.to(device)
        outputs = model(images)
        # layers_outputs = model.intermediate_outputs(images)
        labels = torch.argmax(outputs, dim=-1)
        acc = (labels == y_test).float().mean() * 100
        print(f"Accuracy: {acc} %")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # using algo mnist dataset
    transform = transforms.ToTensor()
    train_set = MNIST(root="../../../dataset/", train=True, transform=transform, download=False)
    test_set = MNIST(root="../../../dataset/", train=False, transform=transform, download=False)

    pdb.set_trace()
    # train(model, train_loader)

    model = Lenet5()
    model = model.to(device=device)
    model.load_state_dict(torch.load("lenet5_mnist.pt", map_location=device))

    test_loader = DataLoader(dataset=test_set, batch_size=len(test_set))
    train_loader = DataLoader(dataset=train_set, batch_size=256 * 4, shuffle=True, num_workers=8, pin_memory=True)

    test(model, test_loader)
