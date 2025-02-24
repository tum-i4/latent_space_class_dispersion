from __future__ import absolute_import, division, print_function

import argparse
import json
from pathlib import Path
import toml
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

import path_setup
from dataset.MNIST.mnist_dataset import MNISTDataset
from dataset.SVHN.svhn_dataset import SVHNDataset
from dataset.GTSRB.gtsrb_dataset_gray import GTSRBDataset_gray
from dataset.GTSRB.gtsrb_dataset import GTSRBDataset

class obj:
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def arguments():
    parser = argparse.ArgumentParser(description="Pre-Process Data for Calculations")
    parser.add_argument(
        "-config_file",
        help="choose configuration with which to run",
        default="config/gtsrb.toml",
    )
    return parser.parse_args()


def export(root_folder: Path, filename: str, dataset_list: list, num_classes: int = 43):

    file = root_folder / filename

    obj = {
        "img_train": dataset_list[0],
        "lbl_train": dataset_list[1],
        "path_train": dataset_list[2],
        "img_valid": dataset_list[3],
        "lbl_valid": dataset_list[4],
        "path_valid": dataset_list[5],
        "img_test": dataset_list[6],
        "lbl_test": dataset_list[7],
        "path_test": dataset_list[8],
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

    splits = ["train", "val", "test"]
    total_dataset_list = []

    norm_mean = getattr(config, "norm_mean_" + config.detection_model.image_set)
    norm_std = getattr(config, "norm_std_" + config.detection_model.image_set)
    
    print(norm_mean, norm_std)

    for selected_split in splits:
        imgs_loaded = []
        if config.data == "mnist":
            dataset = MNISTDataset(
                config=config,
                image_set=config.detection_model.image_set,
                mode=selected_split,
                augmentation=False,
            )
            preprocess = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Normalize(
                        (norm_mean), (norm_std)
                    )
                ]
            )
            gt_imgs = dataset._image_paths  # Dataloader loads images to this attribute. 
            normalized_imgs = torch.stack(
                [preprocess(gt_imgs[i]) for i in tqdm(range(gt_imgs.shape[0]))]
            )
            gt_labels = torch.tensor(dataset.classes).squeeze().to(torch.uint8)
            img_paths = dataset.image_ids # IDs in ascending order.
            img_paths = torch.tensor(
                [int(img_paths[i]) for i in tqdm(range(len(img_paths)))], dtype=torch.int64
            )
        elif config.data == "svhn":
            dataset = SVHNDataset(
                config=config,
                image_set=config.detection_model.image_set,
                mode=selected_split,
                augmentation=False,
            )
            preprocess = torchvision.transforms.Compose(
                [
                    transforms.Resize([config.input_height, config.input_width]),
                    transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (norm_mean), (norm_std)
                    )
                ]
            )
            gt_img_paths = dataset._image_paths

            for path in tqdm(gt_img_paths):
                image = Image.open(Path(path), mode="r").convert("RGB")
                imgs_loaded.append(preprocess(image))

            normalized_imgs = torch.stack(
                [imgs_loaded[i] for i in tqdm(range(len(imgs_loaded)))]
            )
            gt_labels = torch.tensor(dataset.classes).squeeze().to(torch.uint8)
            img_paths = dataset.image_ids
            img_paths = torch.tensor(
                [int(img_paths[i]) for i in tqdm(range(len(img_paths)))], dtype=torch.int64
            )
        
        elif config.data == "gtsrb-gray":
            dataset = GTSRBDataset_gray(
                config=config,
                image_set=config.detection_model.image_set,
                mode=selected_split,
                augmentation=False,
            )
            preprocess = torchvision.transforms.Compose(
                [

                    torchvision.transforms.Normalize(
                         (norm_mean), (norm_std)
                    )
                ]
            )
            gt_imgs = dataset._image_paths  # Dataloader loads images to this attribute. 
            normalized_imgs = torch.stack(
                [preprocess(gt_imgs[i]) for i in tqdm(range(gt_imgs.shape[0]))]
            )
            gt_labels = torch.tensor(dataset.classes).squeeze().to(torch.uint8)
            img_paths = dataset.image_ids # IDs in ascending order.
            img_paths = torch.tensor(
                [int(img_paths[i]) for i in tqdm(range(len(img_paths)))], dtype=torch.int64
            )
        
        elif config.data == "gtsrb":
            dataset = GTSRBDataset(
                config=config,
                image_set=config.detection_model.image_set,
                mode=selected_split,
                augmentation=False,
            )
            preprocess = torchvision.transforms.Compose(
                [
                    transforms.Resize([config.input_height, config.input_width]),
                    transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (norm_mean), (norm_std)
                    )
                ]
            )
            gt_img_paths = dataset._image_paths

            for path in tqdm(gt_img_paths):
                image = Image.open(Path(path), mode="r").convert("RGB")
                imgs_loaded.append(preprocess(image))

            normalized_imgs = torch.stack(
                [imgs_loaded[i] for i in tqdm(range(len(imgs_loaded)))]
            )
            gt_labels = torch.tensor(dataset.classes).squeeze().to(torch.uint8)
            img_paths = dataset.image_ids
            img_paths = torch.tensor(
                [int(img_paths[i]) for i in tqdm(range(len(img_paths)))], dtype=torch.int64
            )
        else:
            raise NotImplementedError("Please extend it to input dataset or use supported dataset.")
        
        
        print(
            "Split:",
            selected_split,
            normalized_imgs.shape,
            gt_labels.shape,
            img_paths.shape,
        )

        total_dataset_list.append(normalized_imgs)
        total_dataset_list.append(gt_labels)
        total_dataset_list.append(img_paths) # MNIST=ids, SVHN=ids (path to be used to fetch images)

    export(
        root_folder=root_folder,
        filename=config.data + "_org_data_normalized.pth",
        dataset_list=total_dataset_list,
        num_classes=config.num_classes,
    )
