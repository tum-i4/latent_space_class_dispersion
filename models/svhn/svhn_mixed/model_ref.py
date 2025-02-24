import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.default_model import ClassificationModel


class SVHN_mixed(ClassificationModel):
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()

        self.model_name = "pytorch_classification_svhn_mixed"
        self.model_path = os.path.join("models/svhn/svhn_mixed/weights", "{}.pth".format(self.model_name))
        print(self.model_path)

        # CNN layers
        self.conv1 = nn.Conv2d(input_channels, 100, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(250)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250 * 2 * 2, 350)
        self.fc2 = nn.Linear(350, num_classes)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(nn.Linear(10 * 4 * 4, 32), nn.ReLU(True), nn.Linear(32, 3 * 2))

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform forward pass
        x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)), 2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)), 2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)), 2))
        x = self.conv_drop(x)
        x = x.view(-1, 250 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        feature_vector = self.fc2(x)
        out = F.log_softmax(feature_vector, dim=1)
        return out, feature_vector

    def inference(self, x, req_feature_vec=False):
        x_pred_softmax, feature_vector = self.forward(x)
        feature_vector = feature_vector[0].cpu().numpy()
        _, x_pred_tags = torch.max(x_pred_softmax, dim=1)
        pred_probs = torch.exp(x_pred_softmax)
        pred_probs = pred_probs.cpu().numpy()
        x_pred_tags = x_pred_tags.cpu().numpy()
        x_pred_prob = pred_probs[0][x_pred_tags]

        if req_feature_vec:
            return x_pred_tags, x_pred_prob, pred_probs, feature_vector
        else:
            return x_pred_tags, x_pred_prob, pred_probs

    def good_seed_detection(self, test_loader, predictor, config):
        """
        from the test set detect seeds which can be considered as initial test seeds for fault detection
        (ones with correct prediction)
        :param test_loader:
        :param predictor: method for prediction
        :return: indexes of seeds to be considered
        """
        gt_labels, pred_labels, pred_op_prob, pred_probs = [], [], [], []

        for i in tqdm(range(len(test_loader))):
            data = test_loader[i]
            images, labels = data
            # labels = labels.numpy()
            layers_outputs, detections, output_dict, feature_vector = predictor(
                self, data, config.network_type, config
            )
            gt_labels.extend(labels)
            pred_labels.extend(detections)
            pred_op_prob.extend(output_dict["op_class_prob"])
            pred_probs.extend(output_dict["op_probs"])

        return gt_labels, pred_labels, pred_op_prob, pred_probs
