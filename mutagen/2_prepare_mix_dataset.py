from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import random
import numpy as np
import toml
import torch

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)


class obj:
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def arguments():
    parser = argparse.ArgumentParser(description="Pre-Process Data for Combined Data Calculations")
    parser.add_argument(
        "-config_file",
        help="choose configuration with which to run",
        default=f"config/mnist.toml", #{os.getenv('DATASET_NAME', 'mnist')}.
    )
    parser.add_argument(
        "-dataset",
        help="dataset name",
        default="mnist", #os.getenv('DATASET_NAME', 'mnist')
    )
    return parser.parse_args()


def export(root_folder: Path, filename: str, dataset_list: list, num_classes: int = 10):

    file = root_folder / filename
    
    stacked_imgs = torch.stack([dataset_list[0][i] for i in tqdm(range(len(dataset_list[0])))])

    obj = {
        "img_merged": stacked_imgs,
        "lbl_merged": torch.tensor(dataset_list[1]),
        "path_merged": torch.tensor(dataset_list[2]),
        "num_classes": num_classes,
    }

    torch.save(obj, file)
    print("Saved", file)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch using device:", device)
    np.random.seed(9)

    args = arguments()
    config = json.loads(json.dumps(toml.load(args.config_file)), object_hook=obj)
    config.device = device
    torch.set_grad_enabled(False)

    root_folder = Path("data") / config.data
    root_folder.mkdir(exist_ok=True, parents=True)

    splits = ["cc_nc", "cc_kmnc", "cc_nbc", "cc_lscd"]
    keywords = ["img_", "lbl_", "path_"]
    img_list, lbl_list, path_list = [], [], []

    norm_mean = getattr(config, "norm_mean_" + config.detection_model.image_set)
    norm_std = getattr(config, "norm_std_" + config.detection_model.image_set)
    print(norm_mean, norm_std)
    
    org_dataset_path = Path("data", args.dataset, args.dataset+"_org_data_normalized.pth")
    cc_dataset_path = Path("data", args.dataset, args.dataset+"_cc_data_normalized_prob_valid.pth")
    
    org_data = torch.load(org_dataset_path, "cpu")
    cc_data = torch.load(cc_dataset_path, "cpu")

    weight_factor_org_data = 0.3

    test_imgs = org_data[keywords[0]+"test"]
    print("Length of test dataset:", len(test_imgs))
    random_indices_test = random.sample(range(len(test_imgs)), int(len(test_imgs)*weight_factor_org_data))
    print("Selected Data:", len(random_indices_test))

    img_list.extend(org_data["img_test"][random_indices_test])
    lbl_list.extend(org_data["lbl_test"][random_indices_test])
    path_list.extend(org_data["path_test"][random_indices_test]) 

    print((np.array(random_indices_test)== np.array(org_data["path_test"][random_indices_test])).sum())

    for i, split in enumerate(splits):
        print(split)

        split_imgs = cc_data[keywords[0]+splits[i]]
        print("Length of cc dataset:", len(split_imgs))
        
        weight_factor_cc_data = (1-weight_factor_org_data) / 4
        random_indices_cc = random.sample(range(len(split_imgs)), int(len(split_imgs)*weight_factor_cc_data))
        print("Selected Data:", len(random_indices_cc))

        print((np.array(random_indices_cc)== np.array(cc_data["path_" + split][random_indices_cc])).sum())

        img_list.extend(cc_data["img_" + split][random_indices_cc])
        lbl_list.extend(cc_data["lbl_"+ split][random_indices_cc])
        path_list.extend(cc_data["path_"+ split][random_indices_cc]) 


    export(
        root_folder=root_folder,
        filename=config.data + "_mixed_data_normalized_valid.pth",
        dataset_list=[img_list, lbl_list, path_list],
        num_classes=config.num_classes,
    )
