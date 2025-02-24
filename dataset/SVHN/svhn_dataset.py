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
from tqdm import tqdm

from dataset.base_dataset import BaseDataset


class SVHNDataset(BaseDataset):
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
        """
        return self._config.data_path

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

        if self._mode == "train":
            ip_path = os.path.join(self._root_path, f"{self._mode}_remix", "digitStruct.feather")
            self._dataset_info = pd.read_feather(path=ip_path)
            ids = self._dataset_info["img_name"].to_list()

        elif self._mode == "val":
            ip_path = os.path.join(self._root_path, "validation_remix", "digitStruct.feather")
            self._dataset_info = pd.read_feather(path=ip_path)
            ids = self._dataset_info["img_name"].to_list()

        else:
            ip_path = os.path.join(self._root_path, f"{self._mode}_remix", "digitStruct.feather")
            self._dataset_info = pd.read_feather(path=ip_path)
            ids = self._dataset_info["img_name"].to_list()

        return ids

    def _load_image_sizes(self) -> Tuple[List[int], List[int]]:
        """
        Retrieves the width and height of all images in the given dataset and
        returns two lists with all integer widths and all integer heights.
        """
        img_widths, img_heights = [], []
        for path in self._image_paths:
            image = Image.open(path, mode="r")
            img_widths.append(image.width)
            img_heights.append(image.height)
        return img_widths, img_heights

    def _load_image_paths(self) -> List[str]:
        """
        Returns a list of file paths for each of the images.
        """
        ids = self.image_ids

        if self._mode == "train":
            ip_path = os.path.join(self._root_path, f"{self._mode}_remix")

        elif self._mode == "val":
            ip_path = os.path.join(self._root_path, "validation_remix")

        else:
            ip_path = os.path.join(self._root_path, f"{self._mode}_remix")

        image_paths = [os.path.join(ip_path, img_id) for img_id in ids]  # Filename doesn't have .png

        return image_paths

    def _load_annotations(self) -> Tuple[List, List, List, List]:
        """
        Loads the categories for each detection for each image and loads the
        bounding box information for each detection for each image.
        """
        classes, coords = [], []

        class_data = self._dataset_info["label"].to_list()

        for class_id in class_data:
            gt_class = np.array([int(class_id)])
            classes.append(gt_class)

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
