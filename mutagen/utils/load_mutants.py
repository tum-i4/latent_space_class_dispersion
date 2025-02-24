import torch
import pandas as pd
from pathlib import Path
import datetime

def load_mutant_model(results_directory, selected_splits, num_classes=10, is_post_training=False):
    mutant_folder_suffix = (
        "post_training_mutants" if is_post_training else "trained_mutants"
    )
    trained_mutant_folders = sorted(
        [
            folder
            for folder in (results_directory / mutant_folder_suffix).iterdir()
            if folder.is_dir()
        ]
    )

    operator_list = [
        str(selected_mutant).split("/")[-1] for selected_mutant in trained_mutant_folders
    ]

    centroid_folders = sorted(
        [
            folder
            for folder in (results_directory / "init_centroids/").iterdir()
            if folder.is_dir()
        ]
    )

    act_traces_folders = sorted(
        [
            folder
            for folder in (results_directory / "act_trace/").iterdir()
            if folder.is_dir()
        ]
    )

    for split in selected_splits:
        start_time = datetime.datetime.now()
        evaluation_folders = sorted(
            [
                folder
                for folder in (results_directory / f"evaln/dataset={split}").iterdir()
                if folder.is_dir()
            ]
        )

        selected_folder = evaluation_folders[0]

        centroid_path = Path(centroid_folders[0], "init_centroids.pickle")
        centroid_loaded = torch.load(centroid_path)

        act_traces_path = Path(act_traces_folders[0], "at_train_data.pickle")
        act_traces_loaded = torch.load(act_traces_path)

        parquet_file_path = Path(selected_folder, "sut_training=0", f"{split}.parquet")

        org_data = pd.read_parquet(parquet_file_path, engine="fastparquet")

        acc_ref = round(
            (org_data["label"] == org_data["output"]).sum() / org_data.shape[0] * 100,
            2,
        )

        print(
            "Accuracy on {} is {} % using {} dataset.".format(
                str(selected_folder).split("/")[-1], acc_ref, split
            )
        )


    # To filter out not properly trained mutants.... (Can be optimized)
    evaluation_folders_updated = {}
    for split in selected_splits:
        evaluation_folders = sorted(
            [
                folder
                for folder in (results_directory / f"evaln/dataset={split}").iterdir()
                if folder.is_dir()
            ]
        )
        centroid_folders = sorted(
            [
                folder
                for folder in (results_directory / "init_centroids/").iterdir()
                if folder.is_dir()
            ]
        )

        act_traces_folders = sorted(
            [
                folder
                for folder in (results_directory / "act_trace/").iterdir()
                if folder.is_dir()
            ]
        )

        for i, selected_folder in enumerate(evaluation_folders):

            centroid_path = Path(centroid_folders[i], "init_centroids.pickle")
            centroid_loaded = torch.load(centroid_path)

            if len(list(centroid_loaded["all_centroids"].keys())) != num_classes:
                print("Centroid doesn't exist for all output classes in ", selected_folder)
                evaluation_folders.pop(i)
                centroid_folders.pop(i)
                act_traces_folders.pop(i)
                pass
            else:
                pass

        evaluation_folders_updated.update({split: evaluation_folders})

    return evaluation_folders_updated, centroid_folders, act_traces_folders