import csv
import os
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from dataset.base_dataset import BaseDataset


class GTSRBDataset(BaseDataset):
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

        # Run dataset_creation.py first to get proper train/val/test splits.
        image_set_path = os.path.join(self._root_path, "{}.csv".format(self._mode))

        if self._mode == "train":
            with open(image_set_path, newline="") as csv_train:
                train_data_reader = csv.DictReader(csv_train)
                for row in train_data_reader:
                    png_name = row["Path"].split("/")[-1]
                    ids.append(png_name.split(".")[0])  # Train/20/00020_00000_00000.png

        elif self._mode == "val":
            with open(image_set_path, newline="") as csv_val:
                val_data_reader = csv.DictReader(csv_val)
                for row in val_data_reader:
                    png_name = row["Path"].split("/")[-1]
                    ids.append(png_name.split(".")[0])  # Train/20/00020_00000_00000.png

        else:
            with open(image_set_path, newline="") as csv_test:
                test_data_reader = csv.DictReader(csv_test)
                for row in test_data_reader:
                    png_name = row["Path"].split("/")[-1]
                    ids.append(png_name.split(".")[0])  # Test/00020.png

        return ids

    def _load_image_sizes(self) -> Tuple[List[int], List[int]]:
        """
        Retrieves the width and height of all images in the given dataset and
        returns two lists with all integer widths and all integer heights.
        """
        img_widths, img_heights = [], []
        image_set_path = os.path.join(self._root_path, "{}.csv".format(self._mode))

        if self._mode == "train":
            with open(image_set_path, newline="") as csv_train:
                train_data_reader = csv.DictReader(csv_train)
                for row in train_data_reader:
                    img_widths.append(int(row["Width"]))
                    img_heights.append(int(row["Height"]))
        elif self._mode == "val":
            with open(image_set_path, newline="") as csv_val:
                val_data_reader = csv.DictReader(csv_val)
                for row in val_data_reader:
                    img_widths.append(int(row["Width"]))
                    img_heights.append(int(row["Height"]))
        else:
            with open(image_set_path, newline="") as csv_test:
                test_data_reader = csv.DictReader(csv_test)
                for row in test_data_reader:
                    img_widths.append(int(row["Width"]))
                    img_heights.append(int(row["Height"]))

        return img_widths, img_heights

    def _load_image_paths(self) -> List[str]:
        """
        Returns a list of file paths for each of the images.
        """
        image_paths = []
        image_set_csv = os.path.join(self._root_path, "{}.csv".format(self._mode))
        with open(image_set_csv, newline="") as image_csv:
            train_data_reader = csv.DictReader(image_csv)
            for row in train_data_reader:
                path = os.path.join(self._root_path, row["Path"])
                image_paths.append(path)
        return image_paths

    def _load_annotations(self) -> Tuple[List, List, List, List]:
        """
        Loads the categories for each detection for each image and loads the
        bounding box information for each detection for each image.
        """
        classes, coords = [], []

        if self._mode == "train":
            file_name = "train.csv"
        elif self._mode == "val":
            file_name = "val.csv"
        else:
            file_name = "test.csv"

        with open(os.path.join(self._config.data_path, file_name), newline="") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            # skip headerline
            next(csvreader)
            for row in csvreader:
                image_width, image_height, x0, y0, x1, y1, class_id, _ = row
                image_width = int(image_width)
                image_height = int(image_height)

                cl = np.array([int(class_id)])
                co = np.array([[x0, y0, x1, y1]])  # Top Left and Bottom Right format.
                co = co.astype(int)
                co[co < 0] = 0
                co[co[:, 0] > image_width, 0] = image_width
                co[co[:, 1] > image_height, 1] = image_height
                co[co[:, 2] > image_width, 2] = image_width
                co[co[:, 3] > image_height, 3] = image_height
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
