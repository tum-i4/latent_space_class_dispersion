import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from dataset.base_dataset import BaseDataset

class_name_dict = {
    "gtsrb": [
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
    ],
    "svhn": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "mnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
}

class CornerCaseDataset(BaseDataset):
    def __init__(self, config, image_set, mode, augmentation) -> None:
        self.difficulties = []
        super().__init__(config, image_set, mode, augmentation)

    def _get_image_set(self):
        return self._image_set

    def _get_dataset_root_path(self) -> List[str]:
        return self._config.experiment_path

    def _get_class_names(self) -> list:
        """
        Returns the list of class names for the given dataset.
        """

        if self._config.data == "gtsrb" or self._config.data == "gtsrb-gray":
            class_name = class_name_dict["gtsrb"]
        else:
            class_name = class_name_dict[self._config.data]
        
        return class_name
    
    def _load_image_ids(self) -> list:
        img_names_id, img_name = [], []

        for images_root_path in self._config.experiment_path:
            self._root_path = images_root_path
            splits = self._get_sequence_splits()

            for split in splits:
                file_names, file_names_json = [], []
                selected_dir = splits[split]
                print("Reading files from Sequence:", selected_dir)

                file_names += [file for file in os.listdir(selected_dir) if file.endswith(".png")]
                file_names_json += [file for file in os.listdir(selected_dir) if file.endswith(".json")]

                image_files = sorted(file_names)
                json_files = sorted(file_names_json)
                img_names_id += [img.split(".")[0] for img in image_files]
                img_name += [img.split("_")[-1] for img in img_names_id]

        return img_names_id

    def _load_image_sizes(self) -> Tuple[list, list]:
        img_widths, img_heights = [], []
        for path in self._image_paths:
            image = Image.open(path, mode="r")
            img_widths.append(image.width)
            img_heights.append(image.height)  # fix constant value w.r.t. original image size
        return img_widths, img_heights

    def _get_sequence_splits(self):
        splits = dict(crashes_rec=[])
        for split in ["crashes_rec"]:
            splits[split] = os.path.join(self._root_path, split)

        return splits

    def _load_image_paths(self) -> list:
        image_paths = []
        crash_types = ["crashes_rec"]

        for images_root_path in self._config.experiment_path:
            for selected_crash in crash_types:
                selected_folder = os.path.join(images_root_path, selected_crash)
                paths = [file for file in os.listdir(selected_folder) if file.endswith(".png")]
                for i in range(len(paths)):
                    path = os.path.join(selected_folder, paths[i])
                    image_paths.append(path)

        def custom_key(path):
            # Split the path by '/' and extract the attribute after the last '/'
            attribute = path.split("/")[-1].split("_")[1]
            return int(attribute)

        return sorted(image_paths, key=custom_key)

    def _load_annotations(self) -> Tuple[list, list, list, list]:
        classes, coords, occls, box_ids = [], [], [], []

        for index, single_path in enumerate(self._image_paths):
            ann_path = single_path[:-4] + ".json"

            with open(ann_path) as json_file:
                data = json.load(json_file)

                cl = data["__gt_labels__"][0]["__gt_class__"]

            cl = np.array([int(cl)])

            classes.append(cl)
            coords.append([0])

        return classes, coords

    def _load_difficulties(self) -> Tuple[list]:
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
