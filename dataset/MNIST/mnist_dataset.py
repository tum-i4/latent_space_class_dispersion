import os
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from dataset.base_dataset import BaseDataset


class MNISTDataset(BaseDataset):
    """
    A dataset class for the SVHN dataset.
    """

    def __init__(self, config, image_set, mode, augmentation) -> None:
        super().__init__(config, image_set, mode, augmentation)

    def _get_image_set(self):
        """
        Retrieves the string name of the current image set.
        """
        return self._image_set

    def _get_dataset_root_path(self) -> str:
        """
        Returns the path to the root folder for this dataset.
        1) If data doesn't exists downloads and stores it.
        2) Manually took out validation set from train and stored it in .pth format. All train, val, test

        """
        if any(os.listdir(self._config.data_path)):
            return self._config.data_path
        else:
            trans = transforms.ToTensor()
            test_set = MNIST(root=self._config.data_path, train=False, transform=trans, download=True)
            return self._config.data_path
        # return self._config.data_path

    def _get_class_names(self) -> List[str]:
        """
        Returns the list of class names for the given dataset.
        """
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def _load_image_ids(self) -> List:
        """
        Returns a list of strings with image ids that are part of the dataset.
        The image ids usually indicated the image file name.
        """
        ids = []
        trans = transforms.ToTensor()

        if self._mode == "train":
            # self.train_set = MNIST(root=self._config.data_path, train=True, transform=None, download=False)
            self.train_set = torch.load(os.path.join(self._root_path, "mnist_train.pth"))
            ids.append([str(i) for i in range(len(self.train_set))])

        elif self._mode == "val":
            # self.val_set = MNIST(root=self._config.data_path, train=False, transform=None, download=False)
            self.val_set = torch.load(os.path.join(self._root_path, "mnist_validation.pth"))
            ids.append([str(i) for i in range(len(self.val_set))])

        else:
            # self.test_set = MNIST(root=self._config.data_path, train=False, transform=None, download=False)
            self.test_set = torch.load(os.path.join(self._root_path, "mnist_test.pth"))
            ids.append([str(i) for i in range(len(self.test_set))])

        return ids[0]

    def _load_image_sizes(self) -> Tuple[List[int], List[int]]:
        """
        Retrieves the width and height of all images in the given dataset and
        returns two lists with all integer widths and all integer heights.
        """
        img_widths, img_heights = [], []
        for img in self._image_paths:
            img_widths.append(28)
            img_heights.append(28)

        return img_widths, img_heights

    def _load_image_paths(self) -> List[str]:
        """
        Returns a list of file paths for each of the images.
        """
        ids = self.image_ids

        if self._mode == "train":
            all_images = self.train_set  # self.train_set.train_data

        elif self._mode == "val":
            all_images = self.val_set

        else:
            all_images = self.test_set

        image_paths = [
            img[0].unsqueeze(dim=0) for img in all_images
        ]  # 1) Filename doesn't have .png unsqueeze(dim=0) 2) List has tuple with 1st entry as image [0,1] and label.
        image_paths = torch.cat(image_paths, dim=0)

        return image_paths

    def _load_annotations(self) -> Tuple[List, List, List, List]:
        """
        Loads the categories for each detection for each image and loads the
        bounding box information for each detection for each image.
        """
        classes, coords = [], []

        if self._mode == "train":
            all_images = self.train_set  # self.train_set.train_labels
        elif self._mode == "val":
            all_images = self.val_set
        else:
            all_images = self.test_set

        classes_list = [img[1] for img in all_images]

        for cl in classes_list:
            cl = np.array([int(cl)])
            co = np.array([])
            classes.append(cl)
            coords.append(co)  # Coordiantes are not useful for classification.

        return classes, coords

    def _load_difficulties(self) -> List[str]:
        """
        Returns a list of difficulties for each of the images based on config mode.
        """
        difficulties = []
        try:
            difficulties = self._dataset.difficulties
        except AttributeError:
            for index in range(len(self.image_ids)):
                difficulties.append(np.array([False] * len(self.classes[index])))

        return difficulties

    def __len__(self) -> int:
        """
        Returns the number of images within this dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Loads an image (input) and its class and coordinates (targets) based on
        the index within the list of image ids. That means that the image
        information that will be returned belongs to the image at given index.

        :param index: The index of the image within the list of image ids that
                you want to get the information for.

        :return: A quadruple of an image, its class, its coordinates and the
                path to the image itself.
        """
        image = self._image_paths[index]  # Use only when using original format / 255.0
        classes = deepcopy(self.classes[index])

        preprocess = transforms.Compose(
            [
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            ]
        )

        image = preprocess(image)

        return image, classes
