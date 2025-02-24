import os

import numpy as np
import pandas as pd

from copy import deepcopy
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from typing import Tuple, List
from easydict import EasyDict

from tqdm import tqdm


class SVHNDataset(Dataset):
    """
    A dataset class for the SVHN dataset.
    """

    def __init__(self, config: EasyDict, image_set, mode, augmentation=None) -> None:
        self._config = config
        self._image_set = image_set
        self._mode = mode
        self._augmentation = augmentation
        self._root_path = config[f'{self._mode}_dataset_path']

        self._dataset_info = pd.read_feather(path=os.path.join(self._root_path, config.dataset_info))
        self._image_paths = self._load_image_paths()
        # self.img_widths, self.img_heights = self._load_image_sizes()
        # self.subimg_widths, self.subimg_heights = self._load_subimage_sizes()
        self.img_boxes = self._load_boxes()
        self.labels = self._dataset_info['label'].values

    def get_image_size(self) -> Tuple[List[int], List[int]]:
        return self.img_widths, self.img_heights

    def get_subimage_size(self) -> Tuple[List[int], List[int]]:
        return self.subimg_widths, self.subimg_heights

    def _load_image_sizes(self) -> Tuple[List[int], List[int]]:
        img_widths, img_heights = [], []
        for path in tqdm(self._image_paths):
            image = Image.open(path, mode='r')

            img_widths.append(image.width)
            img_heights.append(image.height)

        return img_widths, img_heights

    def _load_subimage_sizes(self) -> Tuple[List[int], List[int]]:
        img_widths = self._dataset_info['width'].astype(dtype=int).values
        img_heights = self._dataset_info['height'].astype(dtype=int).values

        return img_widths, img_heights

    def _load_image_paths(self) -> List[str]:
        file_names = self._dataset_info['img_name'].values

        return [f'{self._root_path}/{file_name}' for file_name in file_names]

    def _load_boxes(self) -> List[Tuple[int, int, int, int]]:
        boxes = []
        for height, left, top, width in self._dataset_info[['height', 'left', 'top', 'width']].values:
            box = (left+1, top+1, left+width, top+height)
            boxes.append(box)

        return boxes

    def _transformation(self, img, item) -> torch.Tensor:
        preprocess = transforms.Compose([
            transforms.Resize([self._config.input_width, self._config.input_height]),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])

        img = img.crop(self.img_boxes[item])

        return preprocess(img)

    def __len__(self) -> int:
        if self._augmentation:
            return len(self._image_paths) * 2
        else:
            return len(self._image_paths)

    def _get_image_label(self, item) -> Tuple[torch.Tensor, int]:
        image_path = self._image_paths[item]
        image = Image.open(image_path, mode='r').convert('RGB')
        label = deepcopy(self.labels[item])

        image = self._transformation(image, item)

        return image, label

    def __getitem__(self, item) -> Tuple[torch.Tensor, int]:
        if item >= len(self._image_paths):
            if self._augmentation:
                item = item - len(self._image_paths)
                image, label = self._get_image_label(item)

                image = self._augmentation.transform(image)

                return image, label
            else:
                raise ''

        return self._get_image_label(item)


def test():
    print(os.getcwd())

    config = {
        'dataset_path': 'train',
        'dataset_info': 'digitStruct.feather',
        'input_width': 32,
        'input_height': 32
    }

    config = EasyDict(config)

    dataset = SVHNDataset(config=config,
                          image_set='train',
                          mode='train',
                          augmentation=None)

    img_width, img_height = dataset.get_image_size()
    subimg_width, subimg_height = dataset.get_subimage_size()

    print('-'*10)
    print(f'Image Sizes: width: {min(img_width)}-{max(img_width)} | {np.mean(img_width)}, height: {min(img_height)}-{max(img_height)} | {np.mean(img_height)}')
    print(f'Subimage Sizes: width: {min(subimg_width)}-{max(subimg_width)} | {np.mean(subimg_width)}, height: {min(subimg_height)}-{max(subimg_height)} | {np.mean(subimg_height)}')
    print('-'*10)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

    print(f'Dataset length: {len(dataset)}')

    for i, (img, label) in enumerate(dataloader):
        print(f'Index: {i}, Label: {label[0]}')

    pass


if __name__ == '__main__':
    test()
