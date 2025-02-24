from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import toml
import torch
import torchvision
import sys

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torchvision import transforms

import path_setup
from dataset.corner_case_dataset import CornerCaseDataset
from mutagen.all_classes import *

sys.path.append("/home/go68zas/Documents/mutation_testing_classifiers/")

from dataset.corner_case_dataset import CornerCaseDataset
from mutagen.all_classes import *
from dotenv import load_dotenv

load_dotenv(override=True)


def arguments():
    parser = argparse.ArgumentParser(
        description="Pre-Process Fuzzing Data for Calculations"
    )
    parser.add_argument(
        "-config_file",
        help="choose configuration with which to run",
        default=f"config/gtsrb.toml",  # {os.getenv('DATASET_NAME', 'mnist')}
    )
    return parser.parse_args()


def export(root_folder: Path, filename: str, dataset_list: list, num_classes: int = 43):

    file = root_folder / filename

    obj = {
        "img_cc_nc": dataset_list[0],
        "lbl_cc_nc": dataset_list[1],
        "path_cc_nc": dataset_list[2],
        "img_cc_kmnc": dataset_list[3],
        "lbl_cc_kmnc": dataset_list[4],
        "path_cc_kmnc": dataset_list[5],
        "img_cc_nbc": dataset_list[6],
        "lbl_cc_nbc": dataset_list[7],
        "path_cc_nbc": dataset_list[8],
        "img_cc_lscd": dataset_list[9],
        "lbl_cc_lscd": dataset_list[10],
        "path_cc_lscd": dataset_list[11],
        "num_classes": num_classes,
    }

    torch.save(obj, file)
    print("Saved", file)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch using device:", device)

    args = arguments()
    config = json.loads(json.dumps(toml.load(args.config_file)), object_hook=obj)
    config.device = device
    torch.set_grad_enabled(False)

    root_folder = Path("data") / config.data
    root_folder.mkdir(exist_ok=True, parents=True)

    op_dir = os.getenv("OP_DIR", "gtsrb_1")

    experiment_paths = [
        "/srv/vivek/experiment_data_dh_prob/gtsrb_1_nc_full_valid_099/",
        "/srv/vivek/experiment_data_dh_prob/gtsrb_1_kmnc_full_valid_099/",
        "/srv/vivek/experiment_data_dh_prob/gtsrb_1_nbc_full_valid_099/",
        "/srv/vivek/experiment_data_dh_prob/gtsrb_1_lscd_full_valid_099/",
    ]
    splits = ["cc_nc", "cc_kmnc", "cc_nbc", "cc_lscd"]
    total_dataset_list = []

    norm_mean = getattr(config, "norm_mean_" + config.detection_model.image_set)
    norm_std = getattr(config, "norm_std_" + config.detection_model.image_set)

    print(norm_mean, norm_std)

    for i, selected_path in enumerate(experiment_paths):
        imgs_loaded = []
        if (
            config.data == "mnist"
            or config.data == "svhn"
            or config.data == "gtsrb-gray"
            or config.data == "gtsrb"
        ):
            config.experiment_path = [selected_path]
            dataset = CornerCaseDataset(
                config=config,
                image_set=config.detection_model.image_set,
                mode=splits[i],
                augmentation=False,
            )

            preprocess = torchvision.transforms.Compose(
                [
                    transforms.Resize([config.input_height, config.input_width]),
                    transforms.ToTensor(),
                    torchvision.transforms.Normalize((norm_mean), (norm_std)),
                ]
            )

            gt_img_paths = dataset._image_paths

            for path in tqdm(gt_img_paths):
                if config.data == "gtsrb-gray" or config.data == "mnist":
                    image = Image.open(Path(path), mode="r").convert("L")
                else:
                    image = Image.open(Path(path), mode="r").convert("RGB")
                imgs_loaded.append(preprocess(image))

            normalized_imgs = torch.stack(
                [imgs_loaded[i] for i in tqdm(range(len(imgs_loaded)))]
            )
            gt_labels = torch.tensor(dataset.classes).squeeze().to(torch.uint8)
            img_paths = dataset.image_ids
            img_paths = torch.tensor(
                [int(img_paths[i].split("_")[1]) for i in tqdm(range(len(img_paths)))],
                dtype=torch.int64,
            )  # id_000000_27 (e.g.)
        else:
            raise NotImplementedError(
                "Please extend it to input dataset or use supported dataset."
            )
        print(
            "Split:",
            splits[i],
            normalized_imgs.shape,
            gt_labels.shape,
            img_paths.shape,
        )

        total_dataset_list.append(normalized_imgs)
        total_dataset_list.append(gt_labels)
        total_dataset_list.append(img_paths)

    export(
        root_folder=root_folder,
        filename=config.data + "_cc_data_normalized_prob_valid.pth",
        dataset_list=total_dataset_list,
        num_classes=config.num_classes,
    )
