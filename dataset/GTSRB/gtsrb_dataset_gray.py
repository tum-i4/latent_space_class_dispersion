import csv
import os
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dataset.base_dataset import BaseDataset


class GTSRBDataset_gray(BaseDataset):
    """
    A dataset class for the GTSRB dataset.
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
        """
        return self._config.data_path

    def _get_class_names(self) -> List[str]:
        """
        Returns the list of class names for the given dataset.
        """

        # source: https://github.com/naokishibuya/car-traffic-sign-classification/blob/7d7ee1f58eff86765a2471fb9b06b5783b7f1260/sign_names.csv
        return [
            "Speed limit (20km/h)",
            "Speed limit (30km/h)",
            "Speed limit (50km/h)",
            "Speed limit (60km/h)",
            "Speed limit (70km/h)",
            "Speed limit (80km/h)",
            "End of speed limit (80km/h)",
            "Speed limit (100km/h)",
            "Speed limit (120km/h)",
            "No passing",
            "No passing for vehicles over 3.5 metric tons",
            "Right-of-way at the next intersection",
            "Priority road",
            "Yield",
            "Stop",
            "No vehicles",
            "Vehicles over 3.5 metric tons prohibited",
            "No entry",
            "General caution",
            "Dangerous curve to the left",
            "Dangerous curve to the right",
            "Double curve",
            "Bumpy road",
            "Slippery road",
            "Road narrows on the right",
            "Road work",
            "Traffic signals",
            "Pedestrians",
            "Children crossing",
            "Bicycles crossing",
            "Beware of ice/snow",
            "Wild animals crossing",
            "End of all speed and passing limits",
            "Turn right ahead",
            "Turn left ahead",
            "Ahead only",
            "Go straight or right",
            "Go straight or left",
            "Keep right",
            "Keep left",
            "Roundabout mandatory",
            "End of no passing",
            "End of no passing by vehicles over 3.5 metric tons",
        ]

    def _load_image_ids(self) -> List:
        """
        Returns a list of strings with image ids that are part of the dataset.
        The image ids usually indicated the image file name.
        """
        ids = []

        # Read from created .pth files.
        image_set_path = os.path.join(self._root_path, "gtsrb_lenet_gray_equalized.pth")
        self.data = torch.load(image_set_path)

        if self._mode == "train":
            paths_all = self.data["path_train"]
            ids.append([path.split(".")[0] for path in paths_all])

        elif self._mode == "val":
            paths_all = self.data["path_valid"]
            ids.append([path.split(".")[0] for path in paths_all])

        else:
            paths_all = self.data["path_test"]
            ids.append([path.split(".")[0] for path in paths_all])

        return ids[0]

    def _load_image_sizes(self) -> Tuple[List[int], List[int]]:
        """
        Retrieves the width and height of all images in the given dataset and
        returns two lists with all integer widths and all integer heights.
        """
        img_widths, img_heights = [], []  # Not required as all images saved are constant size.

        return img_widths, img_heights

    def _load_image_paths(self) -> List[str]:
        """
        Returns a list of file paths for each of the images. Here, it has images stored so its rather input image instead of path.
        """
        # image_paths = []

        if self._mode == "train":
            image_paths = self.data["img_train"]

        elif self._mode == "val":
            image_paths = self.data["img_valid"]

        else:
            image_paths = self.data["img_test"]

        return image_paths

    def _load_annotations(self) -> Tuple[List, List, List, List]:
        """
        Loads the categories for each detection for each image and loads the
        bounding box information for each detection for each image.
        """
        classes, coords = [], []

        if self._mode == "train":
            classes_list = self.data["lbl_train"]
        elif self._mode == "val":
            classes_list = self.data["lbl_valid"]
        else:
            classes_list = self.data["lbl_test"]

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
        image = self._image_paths[index]
        # image = Image.open(image_path, mode="r").convert("RGB")
        classes = deepcopy(self.classes[index])
        
        preprocess = transforms.Compose(
            [
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            ]
        )

        image = preprocess(image)

        return image, classes
