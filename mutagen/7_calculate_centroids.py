import os
import sys
import torch
import pandas as pd

from pathlib import Path

import path_setup
sys.path.append(os.path.join(os.getcwd(), "mutagen/utils")) # required for 'from plot_utils import *' to work (inside mutagen.utils.lscd_ms_utils)

from utils.lscd_ms_utils import *

from dotenv import load_dotenv
load_dotenv(override=True)

is_post_training: bool = os.getenv("POST_TRAINING", "True").lower() == "true"


op_dir = Path(os.getenv("OP_DIR", "gtsrb_1"))
num_classes = os.getenv("NUM_CLASSES", 43)

results_directory = "results" / op_dir
# new_directory = "results" / op_dir /"discarded_mutants"

mutant_folder_suffix = "post_training_mutants" if is_post_training else "trained_mutants"
trained_mutant_folders = sorted(
    [
        folder
        for folder in (results_directory / mutant_folder_suffix).iterdir()
        if folder.is_dir()
    ]
)

eval_folder_suffix = "evaln/post_training_mutants/dataset=train" if is_post_training else "evaln/dataset=train"
evaluation_folders = sorted(
    [
        folder
        for folder in (results_directory / eval_folder_suffix).iterdir()
        if folder.is_dir()
    ]
)

print(len(evaluation_folders), len(trained_mutant_folders))

centroid_folder_suffix = "init_centroids/post_training_mutants" if is_post_training else "init_centroids"

def create_hive_folder(results_directory: Path, sut_name: str):
    data_path = results_directory / centroid_folder_suffix / f"sut_name={sut_name}"
    data_path.mkdir(exist_ok=True, parents=True)


operator_list = [
    str(selected_mutant).split("/")[-1]
    for selected_mutant in trained_mutant_folders
]

for mutant_operator in operator_list:   
    create_hive_folder(results_directory=results_directory, sut_name=mutant_operator)


centroid_folders = sorted(
    [
        folder
        for folder in (results_directory / centroid_folder_suffix).iterdir()
        if folder.is_dir()
    ]
)

# dnn_under_test = Net(num_classes=10, drop_rate=0)
parquet_file_path_suffix = "train.parquet" if is_post_training else "sut_training=0/train.parquet"
for i, mutant_folder in enumerate(trained_mutant_folders):
    print(operator_list[i])

    if operator_list[i] == "AAA_Original_000":
        # weights_path = os.path.join(mutant_folder, "model_fuzzing.pth")
        parquet_file_path = os.path.join(
            evaluation_folders[i],
            parquet_file_path_suffix,
        )
    else:
        # weights_path = os.path.join(mutant_folder, "model.pth")
        parquet_file_path = os.path.join(
            evaluation_folders[i],
            parquet_file_path_suffix,
        )

    # dnn_under_test.load_state_dict(torch.load(weights_path))
    print("Loaded Feature Vectors from:", parquet_file_path)

    input_data = pd.read_parquet(parquet_file_path, engine="fastparquet")
    gt_labels, output_labels = input_data["label"], input_data["output"]

    all_centroids = calculate_initial_centroid_radius(
        gt_labels=gt_labels, output_labels=output_labels, input_data_frame=input_data
    )

    if len(all_centroids) != num_classes:
        print(operator_list[i])
    
    centroid_info_dict = {"all_centroids": all_centroids}
    torch.save(centroid_info_dict, Path(centroid_folders[i],"init_centroids.pickle"))
    print("Initial Centroids & radius threshold values are saved @ ", centroid_folders[i])
