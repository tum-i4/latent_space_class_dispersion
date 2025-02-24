import json
import os
import random
import sys
from pathlib import Path

import toml
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomChoice
from tqdm import tqdm

sys.path.append("/home/vekariya/Documents/practicum_dl_testing/testing_framework_fortiss_classifiers")

from dataset.GTSRB.gtsrb_custom import GTSRB_Custom


def prepare_dataset(cls, cls_args, data_folder, data_transforms, extra_transforms):
    train_val_set = cls(data_folder, **cls_args, download=True, transform=data_transforms)

    generator1 = torch.Generator().manual_seed(3176550861)
    train_set, val_set = torch.utils.data.random_split(train_val_set, [0.8, 0.2], generator=generator1)

    extra_sets = [cls(data_folder, **cls_args, transform=t) for t in extra_transforms]
    extra_sets.insert(0, train_set)
    train_set = torch.utils.data.ConcatDataset(extra_sets)

    return train_set, val_set


def create_transform(*extra_transforms, size: int = 32):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((size, size)),
            *extra_transforms,
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.RandomEqualize(p=1.0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5317,), (0.3174,)),
        ]
    )


def to_tensor(dataset, shuffle: bool):
    combined_list = [(img, lbl) for img, lbl in tqdm(dataset)]
    all_images_tensor = torch.stack([i[0] for i in combined_list])
    labels_tensor = torch.tensor([i[1] for i in combined_list], dtype=torch.uint8)

    return all_images_tensor, labels_tensor


def export(root_folder: Path, filename: str, train: Dataset, val: Dataset, shuffle: bool, num_classes: int = 43):
    train_img, train_lbl = to_tensor(train, shuffle)
    valid_img, valid_lbl = to_tensor(val, shuffle)

    file = root_folder / filename

    torch.save(
        {
            "img_train": train_img,
            "img_valid": valid_img,
            "lbl_train": train_lbl,
            "lbl_valid": valid_lbl,
            "num_classes": num_classes,
        },
        file,
    )

    print("Saved", file, train_img.shape, train_img.dtype, valid_img.shape)


def main(data_folder: Path):
    torch.manual_seed(3176550861)
    random.seed(3176550861)

    # transforms = [
    #     create_transform(torchvision.transforms.ColorJitter(brightness=5)),
    #     create_transform(torchvision.transforms.ColorJitter(saturation=5)),
    #     create_transform(torchvision.transforms.ColorJitter(contrast=5)),
    #     create_transform(torchvision.transforms.ColorJitter(hue=0.4)),
    #     create_transform(torchvision.transforms.RandomRotation(15)),
    #     create_transform(
    #         torchvision.transforms.RandomHorizontalFlip(1),
    #         torchvision.transforms.RandomVerticalFlip(1),
    #     ),
    #     create_transform(torchvision.transforms.RandomAffine(degrees=15, shear=2)),
    #     create_transform(torchvision.transforms.RandomAffine(degrees=15, translate=(0.1, 0.1))),
    #     create_transform(torchvision.transforms.CenterCrop(32), size=36),
    # ]

    data_transforms = create_transform()

    root_folder = data_folder / "pth_shuffled_gray_without_norm_without_equalize"
    root_folder.mkdir(exist_ok=True, parents=True)

    gtsrb_train, gtsrb_val = prepare_dataset(GTSRB_Custom, {"split": "train"}, data_folder, data_transforms, [])
    export(root_folder, "gtsrb_random_plain.pth", gtsrb_train, gtsrb_val, shuffle=True)


if __name__ == "__main__":
    main(Path("data"))
