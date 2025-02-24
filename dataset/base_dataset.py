from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as FT
from PIL import Image
from torch.utils import data
from torchvision import transforms

# from dataset.preprocess import preprocess  # can be used for other dataset or based on requirements.


class BaseDataset(data.Dataset):
    """
    This dataset class serves as a base class for any dataset that should be fed
    into the SSD model. You simply need to implement the following functions to
    load the dataset in a format that is compatible with our training, inference
    and evaluation scripts:
    """

    def __init__(self, config, image_set, mode, augmentation) -> None:
        super().__init__()
        self._mode = mode
        self._config = config
        self._image_set = image_set
        self.augmentation = augmentation
        self.class_names = self._get_class_names()
        self._root_path = self._get_dataset_root_path()
        self.image_ids = self._load_image_ids()
        self._image_paths = self._load_image_paths()
        self.img_widths, self.img_heights = self._load_image_sizes()
        self.classes, self.coordinates = self._load_annotations()
        self.difficulties = self._load_difficulties()

        self.norm_mean = getattr(self._config, "norm_mean_" + self._image_set)
        self.norm_std = getattr(self._config, "norm_std_" + self._image_set)

    def _get_image_set(self):
        """
        Retrieves the string name of the current image set.
        """
        raise NotImplementedError

    def _get_dataset_root_path(self) -> str:
        """
        Returns the path to the root folder for this dataset. For PascalVOC this
        would be the path to the VOCdevkit folder.
        """
        raise NotImplementedError

    def _get_class_names(self) -> List[str]:
        """
        Returns the list of class names for the given dataset.
        """
        raise NotImplementedError

    def _load_image_ids(self) -> List:
        """
        Returns a list of strings with image ids that are part of the dataset.
        The image ids usually indicated the image file name.
        """
        raise NotImplementedError

    def _load_image_sizes(self) -> Tuple[List[int], List[int]]:
        """
        Retrieves the width and height of all images in the given dataset and
        returns two lists with all integer widths and all integer heights.
        """
        raise NotImplementedError

    def _load_image_paths(self) -> List[str]:
        """
        Returns a list of file paths for each of the images.
        """
        raise NotImplementedError

    def _load_annotations(self) -> Tuple[List, List, List, List]:
        """
        Loads the categories for each detection for each image and loads the
        bounding box information for each detection for each image.
        """
        raise NotImplementedError

    def _load_difficulties(self) -> List[str]:
        """
        Returns a list of difficulties for each of the images based on config mode.
        """
        raise NotImplementedError

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
        image_path = self._image_paths[index]
        if self._image_set == "gtsrb-gray" or self._image_set == "mnist":
            image = Image.open(image_path, mode="r").convert("L")
        else:
            image = Image.open(image_path, mode="r").convert("RGB")
        classes = deepcopy(self.classes[index])

        # preprocessing according to
        # https://mailto-surajk.medium.com/a-tutorial-on-traffic-sign-classification-using-pytorch-dabc428909d7
        preprocess = transforms.Compose(
            [
                transforms.Resize([self._config.input_height, self._config.input_width]),
                transforms.ToTensor(),
                # transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            ]
        )

        image = preprocess(image)

        return image, classes

    # Responsible for batching and padding inputs
    @staticmethod
    def collate_fn(samples_in_batch):
        """
        Helps to create a real batch of inputs and targets. The function
        receives a list of single samples and combines the inputs as well as the
        targets to single lists.

        :param samples_in_batch: A list of quadruples of an image, its class,
            its coordinates and file path.

        :return: A batch of samples.
        """
        images = [sample[0] for sample in samples_in_batch]
        paths = [sample[3] for sample in samples_in_batch]

        classes = [sample[1] for sample in samples_in_batch]
        coords = [sample[2] for sample in samples_in_batch]

        images = torch.stack(images, dim=0)

        return images, classes, coords, paths
