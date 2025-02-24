import glob
import math
import os
import random
import shutil
import sys
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

sys.path.append("/home/vekariya/Documents/practicum_dl_testing/testing_framework_fortiss_classifiers")

__SVHN_DATASET_PATH: str = "/data/disk2/svhn_dataset"
__ANNOTATIONS_FILE: str = "digitStruct.mat"


"""
#### MatLab code for reverence ####

load digitStruct.mat
for i = 1:length(digitStruct)
    im = imread([digitStruct(i).name]);
    for j = 1:length(digitStruct(i).bbox)
        [height, width] = size(im);
        aa = max(digitStruct(i).bbox(j).top+1,1);
        bb = min(digitStruct(i).bbox(j).top+digitStruct(i).bbox(j).height, height);
        cc = max(digitStruct(i).bbox(j).left+1,1);
        dd = min(digitStruct(i).bbox(j).left+digitStruct(i).bbox(j).width, width);

        imshow(im(aa:bb, cc:dd, :));
        fprintf('%d\n',digitStruct(i).bbox(j).label );
        pause;
    end
end
"""


def create_df_from_HDF5(annots: h5py.File) -> pd.DataFrame:
    bbox = "digitStruct/bbox"

    rows = []
    for img_id in tqdm(range(0, annots[bbox].shape[0])):
        ref = annots[bbox][()][img_id][0]

        keys = np.asarray(annots[ref])
        shape = annots[ref][keys[0]].shape

        for sub_img_id in range(shape[0]):
            row_dict = {}
            for k in keys:
                if shape[0] == 1:
                    row_dict.update({k: annots[ref][k][()][sub_img_id][0]})
                else:
                    sub_ref = annots[ref][k][()][sub_img_id][0]
                    row_dict.update({k: annots[sub_ref][()][0][0]})

            row_dict.update({"img_name": f"{img_id + 1}.png", "sub_img_id": sub_img_id})
            rows.append(row_dict)

    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype(int)
    df["label"] = df["label"].map(lambda x: 0 if x == 10 else x)

    return df


def create_mix_set():
    for mode in ["test", "train"]:
        op_dir = os.path.join(__SVHN_DATASET_PATH, f"{mode}_remix")
        if not os.path.exists(op_dir):
            os.mkdir(op_dir)

        remix_path = os.path.join(__SVHN_DATASET_PATH, f"{mode}_32x32_remix.mat")
        annots = sio.loadmat(remix_path)

        image_names = []
        for i in tqdm(range(annots["X"].shape[-1])):
            img = Image.fromarray(annots["X"][:, :, :, i])
            img.save(os.path.join(op_dir, str(i)), "png")
            image_names.append(f"{i}")

        df = pd.DataFrame(data=annots["y"][:, 0], columns=["label"])
        df["img_name"] = image_names
        df.to_feather(path=os.path.join(op_dir, "digitStruct.feather"))

        print(df)
        print("Data stored at: {}".format(os.path.join(op_dir, "digitStruct.feather")))


def create_validation_set(path: str, source: str, destination: str, train_split: float, indices: List[int] = None):
    source_path = os.path.join(path, source)
    destination_path = os.path.join(path, destination)

    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    if indices is None:
        images = glob.glob(f"{source_path}/*")

        indices = []
        val_size = math.ceil(len(images) * (1 - train_split))
        for image in random.sample(images, k=val_size):
            os.rename(image, destination_path + "/" + image.split("/")[-1])

            indices.append(image)

        np.save(destination_path + "/validation_indices.npy", indices)

    else:
        for image in indices:
            os.rename(str(image), destination_path + "/" + str(image).split("/")[-1])

    df = pd.read_feather(path=os.path.join(source_path, "digitStruct.feather"))

    duplicates_idx = np.asarray([source_path + "/" + image in indices for image in df["img_name"].values])
    val_df = df[duplicates_idx]
    train_df = df[~duplicates_idx]

    os.remove(os.path.join(source_path, "digitStruct.feather"))
    train_df, val_df = train_df.reset_index(), val_df.reset_index()
    val_df.to_feather(path=os.path.join(destination_path, "digitStruct.feather"))
    train_df.to_feather(path=os.path.join(source_path, "digitStruct.feather"))


def main():
    os.chdir("../../")
    print(os.getcwd())

    for mode in ["extra", "test", "train"]:
        path = os.path.join(__SVHN_DATASET_PATH, f"{mode}")
        annots = h5py.File(os.path.join(path, f"{__ANNOTATIONS_FILE}"), "r")
        df = create_df_from_HDF5(annots)

        print(df)

        df.to_csv(path_or_buf=os.path.join(path, "digitStruct.csv"), index=False)
        df.to_feather(path=os.path.join(path, "digitStruct.feather"))


if __name__ == "__main__":
    # TODO: your path to the validation_indices.npy file
    # PATH = "/home/jan/TUM/WS2023_24/ATDLM/train_test_ssd_ws_22_23/classification_dnns/dataset/SVHN"
    # main()
    create_mix_set()

    indices_path = os.path.join(__SVHN_DATASET_PATH, "validation_indices.npy")
    indices = np.load(indices_path)
    new_local_path = os.path.join(__SVHN_DATASET_PATH, "validation_indices_ref.npy")
    shutil.copy(indices_path, new_local_path)

    prefix_to_replace = "/home/jan/TUM/WS2023_24/ATDLM/train_test_ssd_ws_22_23/classification_dnns/dataset/SVHN"
    replace_with = __SVHN_DATASET_PATH
    new_arr = np.char.replace(indices, prefix_to_replace, replace_with)

    np.save(indices_path, new_arr)
    local_indices = np.load(indices_path)

    create_validation_set(
        __SVHN_DATASET_PATH,
        source="train_remix",
        destination="validation_remix",
        train_split=0.85,
        indices=local_indices,
    )
