import os
import sys
import math

from pathlib import Path
import torch
import numpy as np
import pandas as pd

import path_setup

sys.path.append(
    os.path.join(os.getcwd(), "mutagen/utils")
)  # required for 'from plot_utils import *' to work (inside mutagen.utils.lscd_ms_utils)

import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

from utils.lscd_ms_utils import *
from utils.plot_utils import *
from utils.surprise_adequacy_utils import *
from utils.load_mutants import load_mutant_model

np.random.seed(0)

is_post_training: bool = os.getenv("POST_TRAINING", "True").lower() == "true"

dataset = os.getenv("DATASET", "svhn")
dataset_plot = "SVHN"
op_dir = os.getenv("OP_DIR", "svhn_3")
op_file_name = os.getenv("OP_DIR", "svhn_1_sc")
multi_threading = os.getenv("MUTLI_THREADING", "True").lower() == "true"
num_classes = int(os.getenv("NUM_CLASSES", 10))
outlier_methods = ["IQR"]

results_directory = Path.cwd() / "results" / op_dir

log_filepath = results_directory / Path(op_file_name)

selected_splits = ["test", "cc_nc", "cc_kmnc", "cc_nbc", "mixed"]
selected_splits_analyze = ["test", "cc_nc", "cc_kmnc", "cc_nbc", "mixed"]

evaluation_folders_updated, centroid_folders, act_traces_folders = load_mutant_model(
    results_directory=results_directory,
    selected_splits=selected_splits,
    num_classes=num_classes,
    is_post_training=is_post_training,
)

# To calculate MS, LSCD and SC for all mutatnts in all splits...
ms_dict, lscd_dict, acc_dict, sc_dict = {}, {}, {}, {}
ms_dict_plot, lscd_dict_plot, acc_dict_plot, sc_dict_plot = {}, {}, {}, {}

index_last, index_current = 0, 0

start_time_1 = datetime.datetime.now()
for split in selected_splits_analyze:
    ms_list, lscd_list, scaled_lscd_list, acc_list = [], [], [], []

    evaluation_folders_current = evaluation_folders_updated[split]

    for i, selected_folder in enumerate(evaluation_folders_current):

        centroid_path = Path(centroid_folders[i], "init_centroids.pickle")
        centroid_loaded = torch.load(centroid_path)

        parquet_file_path_suffix = "" if is_post_training else "sut_training=0"
        parquet_file_path = Path(
            selected_folder, parquet_file_path_suffix, f"{split}.parquet"
        )
        ref_parquet_file_path = Path(
            evaluation_folders_current[0], parquet_file_path_suffix, f"{split}.parquet"
        )

        input_data = pd.read_parquet(parquet_file_path, engine="fastparquet")
        ref_data = pd.read_parquet(ref_parquet_file_path, engine="fastparquet")

        print("Calculating MS for:", parquet_file_path)
        print("Refernce file for MS is:", ref_parquet_file_path)

        if str(selected_folder).split("/")[-1].split("=")[-1] == "AAA_Original":
            if multi_threading:
                lscd_values_dict, avg_lscd_value = calculate_lscd_multithreading(
                    centroid_loaded, input_data, num_classes, num_workers=8
                )
            else:
                lscd_values_dict, avg_lscd_value = calculate_lscd(
                    centroid_loaded, input_data, num_classes
                )

            accuracy_current = round(
                (input_data["label"] == input_data["output"]).sum()
                / input_data.shape[0]
                * 100,
                2,
            )
            print(
                "Accuracy on {} is {} % using {} dataset.".format(
                    str(selected_folder).split("/")[-1], accuracy_current, split
                )
            )
            print("LSCD Score for original datasets:", avg_lscd_value)

        else:
            mutation_score, cl_ms_score_dict = calculate_mutation_score(
                input_data=input_data,
                reference_data=ref_data,
                num_classes=num_classes,
                split=split,
            )
            if multi_threading:
                lscd_values_dict, avg_lscd_value = calculate_lscd_multithreading(
                    centroid_loaded, input_data, num_classes, num_workers=8
                )
            else:
                lscd_values_dict, avg_lscd_value = calculate_lscd(
                    centroid_loaded, input_data, num_classes
                )

            accuracy_current = round(
                (input_data["label"] == input_data["output"]).sum()
                / input_data.shape[0]
                * 100,
                2,
            )

            acc_list.append(accuracy_current)
            ms_list.append(round(mutation_score, 3))
            lscd_list.append(avg_lscd_value)

            print(
                "Accuracy on {} is {} % using {} dataset.".format(
                    str(selected_folder).split("/")[-1], accuracy_current, split
                )
            )

            print("Mutation Score:", mutation_score)
            print("LSCD Score:", avg_lscd_value)

        ms_dict.update({split: ms_list})
        lscd_dict.update({split: lscd_list})
        acc_dict.update({split: acc_list})

print("\n MS v/s LSCD calculation took", datetime.datetime.now() - start_time_1)

start_time_2 = datetime.datetime.now()
for split in selected_splits_analyze:
    sc_list = []

    evaluation_folders_current = evaluation_folders_updated[split]

    for i, selected_folder in enumerate(evaluation_folders_current):

        centroid_path = Path(centroid_folders[i], "init_centroids.pickle")
        centroid_loaded = torch.load(centroid_path)

        act_trace_path = Path(act_traces_folders[i], "at_train_data.pickle")
        act_trace_loaded = torch.load(act_trace_path)

        parquet_file_path = Path(selected_folder, "sut_training=0", f"{split}.parquet")
        ref_parquet_file_path = Path(
            evaluation_folders_current[0], "sut_training=0", f"{split}.parquet"
        ) 

        input_data = pd.read_parquet(parquet_file_path, engine="fastparquet")
        ref_data = pd.read_parquet(ref_parquet_file_path, engine="fastparquet")

        print("Calculating MS for:", parquet_file_path)
        print("Refernce file for MS is:", ref_parquet_file_path)

        if str(selected_folder).split("/")[-1].split("=")[-1] == "AAA_Original":
            if multi_threading:
                distance_sa, surprise_coverage, surprise_coverage_dict = (
                    calculate_surprise_adequacy(
                        train_act_traces=act_trace_loaded,
                        input_data=input_data,
                        num_classes=num_classes,
                        multi_threading=multi_threading,
                    )
                )
            else:
                distance_sa, surprise_coverage, surprise_coverage_dict = (
                    calculate_surprise_adequacy(
                        train_act_traces=act_trace_loaded,
                        input_data=input_data,
                        num_classes=num_classes,
                    )
                )

            accuracy_current = round(
                (input_data["label"] == input_data["output"]).sum()
                / input_data.shape[0]
                * 100,
                2,
            )
            print(
                "Accuracy on {} is {} % using {} dataset.".format(
                    str(selected_folder).split("/")[-1], accuracy_current, split
                )
            )
            print("SC Score for original datasets:", surprise_coverage)
        else:
            mutation_score, cl_ms_score_dict = calculate_mutation_score(
                input_data=input_data,
                reference_data=ref_data,
                num_classes=num_classes,
                split=split,
            )
            if multi_threading:
                distance_sa, surprise_coverage, surprise_coverage_dict = (
                    calculate_surprise_adequacy(
                        train_act_traces=act_trace_loaded,
                        input_data=input_data,
                        num_classes=num_classes,
                        multi_threading=multi_threading,
                    )
                )
            else:
                distance_sa, surprise_coverage, surprise_coverage_dict = (
                    calculate_surprise_adequacy(
                        train_act_traces=act_trace_loaded, input_data=input_data, num_classes=num_classes
                    )
                )

            accuracy_current = round(
                (input_data["label"] == input_data["output"]).sum()
                / input_data.shape[0]
                * 100,
                2,
            )

            sc_list.append(surprise_coverage)

            print(
                "Accuracy on {} is {} % using {} dataset.".format(
                    str(selected_folder).split("/")[-1], accuracy_current, split
                )
            )

            print("Mutation Score:", mutation_score)
            print("Surprise Coverage:", surprise_coverage)

        sc_dict.update({split: sc_list})


print("\n MS v/s SC calculation took", datetime.datetime.now() - start_time_2)

print(
    len(sc_dict["test"]),
    len(ms_dict["test"]),
    len(acc_dict["test"]),
    len(lscd_dict["test"]),
)

for filter_method in outlier_methods:
    ms_final_list, lscd_final_list, acc_final_list, sc_final_list = [], [], [], []
    for split in selected_splits_analyze:
        ms_list, lscd_list, acc_list, sc_list = (
            ms_dict[split],
            lscd_dict[split],
            acc_dict[split],
            sc_dict[split],
        )
        data = pd.DataFrame(
            {
                "ms_list": ms_list,
                "lscd_list": lscd_list,
                "acc_list": acc_list,
                "sc_list": sc_list,
            }
        )

        data_clean = filter_data(data, filter_method)

        print("Original Data Shape:", data.shape)
        print("Cleaned Data Shape:", data_clean.shape)

        ms_dict_plot.update({split: data_clean["ms_list"]})
        lscd_dict_plot.update({split: data_clean["lscd_list"]})
        acc_dict_plot.update({split: data_clean["acc_list"]})
        sc_dict_plot.update({split: data_clean["sc_list"]})

        for i in range(len(data_clean["ms_list"])):
            if math.isnan(lscd_dict[split][i]):
                data_clean["ms_list"].pop(i)
                data_clean["lscd_list"].pop(i)
                data_clean["acc_list"].pop(i)
                data_clean["sc_list"].pop(i)
                pass
            else:
                ms_final_list.append(data_clean["ms_list"][i])
                lscd_final_list.append(float(data_clean["lscd_list"][i]))
                acc_final_list.append(data_clean["acc_list"][i])
                sc_final_list.append(data_clean["sc_list"][i])

    plot_ms_lscd_scatter(
        ms_dict_plot,
        lscd_dict_plot,
        splits=split,
        dataset=dataset_plot,
        filter=filter_method,
    )
    plot_ms_sc_scatter(
        ms_dict_plot, sc_dict_plot, splits=split, dataset=dataset_plot, filter=filter_method
    )

    filtered_dict = correlation_analysis(
        ms_list=ms_final_list,
        lscd_list=lscd_final_list,
        acc_list=acc_final_list,
        sc_list=sc_final_list,
        log_filepath=log_filepath,
        filter_method=filter_method,
    )
