from itertools import combinations
from collections import defaultdict
import os
import torch
from pathlib import Path
import numpy as np
import json
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
from models.mnist.lenet5.model import Net
import pandas as pd
from tqdm import tqdm
from plot_utils import *
from scipy import stats
from sklearn.ensemble import IsolationForest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed 

def calculate_classwise_distances(cl_centroid, vector):
        diff = torch.abs(torch.tensor(vector) - cl_centroid)
        euc_dist = torch.norm(diff, p=2)
        return euc_dist # if euc_dist else torch.tensor([0.0]) # handle empty all_dist

def calculate_lscd_for_class(cl, centroid_loaded, label_pairs_dict, num_classes):
    if cl not in centroid_loaded["all_centroids"]:
        cl_centroid = torch.zeros(num_classes)
    else:
        cl_centroid = centroid_loaded["all_centroids"][cl]

    cl_feature_vectors = label_pairs_dict[cl]
    if not cl_feature_vectors:
        return cl, torch.tensor([0.0]), []  # Return empty list for distances

    all_dist = []
    try:
        for vector in cl_feature_vectors:  # Still need to loop to store individual distances
            diff = torch.abs(torch.tensor(vector) - cl_centroid)
            euc_dist = torch.norm(diff, p=2)
            all_dist.append(euc_dist)

        lscd_value = torch.mean(torch.tensor(all_dist)) if all_dist else torch.tensor([0.0])
        return cl, lscd_value, all_dist  # Return distances as well

    except Exception as e:
        print(f"Error in class calculation: {e}")
        return cl, torch.tensor([0.0]), []  # Return empty list in case of error
        

def calculate_initial_centroid_radius(gt_labels, output_labels, input_data_frame):
    centroid_dictionary = {}
    sut_name = input_data_frame["sut_name"]
    print(
        "Calculating the initial centroid positioning & radius thresholds from the train dataset for .",
        sut_name[0],
    )

    accuracy = round((gt_labels == output_labels).sum() / gt_labels.shape[0] * 100, 2)
    fp_samples = int((gt_labels != output_labels).sum())

    print(
        "Accuracy on {} is {} % using training dataset.".format(sut_name[0], accuracy)
    )
    # feature_vectors = input_data["latent_space"]
    mask = input_data_frame["label"] == input_data_frame["output"]
    input_data_frame_new = input_data_frame[mask]
    input_data_frame_new.reset_index(drop=True, inplace=True)

    print("FPs: ", fp_samples)
    print(
        "New Dataframe resized to:",
        input_data_frame_new.shape[0],
    )

    grouped = input_data_frame_new.groupby("label")
    label_pairs_dict = defaultdict(list)

    for label, group in grouped:
        features = group["latent_space"].tolist()
        label_pairs_dict[label] = features
        # print(len(features))

    samples_ignored = 0  # TP values with nan entries.

    for key, class_wise_feature_vectors in label_pairs_dict.items():
        try:
            features_cl_train = torch.tensor(class_wise_feature_vectors)
            if (features_cl_train.size != 0) and not torch.any(
                torch.isnan(features_cl_train)
            ):
                # print(key, features_cl_train.shape)
                mean_feature = torch.mean(features_cl_train, dim=0)
                centroid_dictionary[key] = mean_feature
            else:
                samples_ignored += 1
        except:
            samples_ignored += 1
    print("Samples Ignored:", samples_ignored)
    return centroid_dictionary


def calculate_mutation_score(input_data, reference_data, num_classes, split):
    cl_ms_score_dict = {}

    input_data_op = input_data["output"]
    ref_data_op = reference_data["output"]
    mutation_score_DMplus = (
        (input_data_op != ref_data_op).sum() / input_data_op.shape[0]
    ) 

    input_data_frame_new = input_data.copy(deep=True)
    ref_data_frame_new = reference_data.copy(deep=True)
    ip_grouped = input_data_frame_new.groupby("label")
    ref_grouped = ref_data_frame_new.groupby("label")
    label_pairs_dict = defaultdict(list)

    for label, group in ip_grouped:
        ip_cl_ops = group["output"].to_numpy()
        ref_cl_ops = ref_grouped.get_group(label)["output"].to_numpy()
        cl_ms_score_dict[label] = float(
            ((ip_cl_ops != ref_cl_ops).sum() / ip_cl_ops.shape[0]) * 100
        )

    return mutation_score_DMplus, cl_ms_score_dict


def get_lsc(lower, upper, k, sa):
    """Surprise Coverage - Latent Space

    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
        k (int): The number of buckets.
        sa (list): List of lsa or dsa.

    Returns:
        cov (int): Surprise coverage.
    """
    buckets = np.digitize(sa, np.linspace(lower, upper, k))
    return len(list(set(buckets))) / float(k)


def calculate_lscd(centroid_loaded, input_data_frame, num_classes):
    sut_name = input_data_frame["sut_name"]

    print(
        "Calculating the LSCD values for ",
        sut_name[0],
    )

    all_distance_dict, lscd_values_dict, avg_lscd_value = ({}, {}, 0.0)
    # lsc_sc_dict, avg_lsc_value = ({},0.0)

    input_data_frame_new = input_data_frame.copy(deep=True)
    grouped = input_data_frame_new.groupby("label")
    # input_data_frame_new.reset_index(drop=True, inplace=True)
    label_pairs_dict = defaultdict(list)

    for label, group in grouped:
        features = group["latent_space"].tolist()
        label_pairs_dict[label] = features

    for cl in range(num_classes):
        all_dist, all_dist_2 = [], []
        if cl in centroid_loaded["all_centroids"]:
            cl_centroid = centroid_loaded["all_centroids"][cl]
        else:
            cl_centroid = torch.zeros(num_classes)

        cl_feature_vectors = torch.tensor(label_pairs_dict[cl])

        try:
            diffs = torch.abs(cl_feature_vectors - cl_centroid)
            euc_dists = torch.norm(diffs, p=2, dim=1)
        except:
            pass

        lscd_values_dict[cl] = torch.mean(torch.tensor(euc_dists))
        all_distance_dict[cl] = euc_dists.tolist()

    for cl in range(num_classes):
        avg_lscd_value += lscd_values_dict[cl]


    avg_lscd_value = round(float(avg_lscd_value / num_classes),4)

    return lscd_values_dict, avg_lscd_value 


def calculate_lscd_multithreading(centroid_loaded, input_data_frame, num_classes, num_workers=8):
    sut_name = input_data_frame["sut_name"]

    print(
        "Calculating the LSCD values for ",
        sut_name[0],
    )

    all_distance_dict, lscd_values_dict, avg_lscd_value = ({}, {}, 0.0)
    # lsc_sc_dict, avg_lsc_value = ({},0.0)

    input_data_frame_new = input_data_frame.copy(deep=True)
    grouped = input_data_frame_new.groupby("label")
    # input_data_frame_new.reset_index(drop=True, inplace=True)
    label_pairs_dict = defaultdict(list)

    for label, group in grouped:
        features = group["latent_space"].tolist()
        label_pairs_dict[label] = features
        

    for cl in range(num_classes):
        if cl in centroid_loaded["all_centroids"]:
            cl_centroid = centroid_loaded["all_centroids"][cl]
        else:
            cl_centroid = torch.zeros(num_classes)

        cl_feature_vectors = torch.tensor(label_pairs_dict[cl])

        # Using Multi-threading for each class using ThreadPoolExecutor.
        all_dist = []
        cl_feature_vectors = torch.tensor(label_pairs_dict[cl])
        valid_vectors = cl_feature_vectors[~torch.isnan(cl_feature_vectors).any(dim=1)]
        with ThreadPoolExecutor() as executor:
                futures = [executor.submit(calculate_classwise_distances, cl_centroid, valid_vectors[i]) for i in range(len(valid_vectors))]

                for future in as_completed(futures):
                    euc_dist = future.result()
                    all_dist.append(euc_dist)
        
        lscd_values_dict[cl] = torch.mean(torch.tensor(all_dist))
        all_distance_dict[cl] = all_dist

    for cl in range(num_classes):
        avg_lscd_value += lscd_values_dict[cl]

    avg_lscd_value = round(float(avg_lscd_value / num_classes), 4)

    return lscd_values_dict, avg_lscd_value


def filter_data(data, filter_method):

    if filter_method == "IQR":
        # Calculate the IQR (interquartile range) & Filter out the outliers
        Q1 = data.quantile(0.20)
        Q3 = data.quantile(0.80)
        IQR = Q3 - Q1
        data_clean = data[
            ~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
        ]
        data_clean = data_clean.reset_index()

    else:
        data_clean = data

    return data_clean

def correlation_analysis (ms_list, lscd_list, acc_list, sc_list, log_filepath, filter_method):
    np.random.seed(0)
    correlation_methods = ["Spearman", "Pearson"]
    filtered_dict, filtered_dict_list = {}, {}
    
    data_clean = pd.DataFrame({
        'ms_list': ms_list,
        'lscd_list': lscd_list,
        'acc_list': acc_list,
        'sc_list': sc_list
    })

    rng = np.random.default_rng()

    for i, correlation_method in enumerate(correlation_methods):
        if correlation_method == "Spearman":
            cor_1, p_value_1 = spearmanr(data_clean['ms_list'], data_clean['lscd_list'])
            cor_2, p_value_2 = spearmanr(data_clean['acc_list'], data_clean['ms_list'])
            cor_3, p_value_3 = spearmanr(data_clean['acc_list'], data_clean['lscd_list'])
            cor_4, p_value_4 = spearmanr(data_clean['ms_list'], data_clean['sc_list'])
            cor_5, p_value_5 = spearmanr(data_clean['sc_list'], data_clean['acc_list'])
        elif correlation_method == "Pearson":
            # method = stats.MonteCarloMethod(rvs=(rng.uniform, rng.uniform))
            method = stats.PermutationMethod(n_resamples=50, random_state=rng)
            cor_1, p_value_1 = pearsonr(data_clean['ms_list'], data_clean['lscd_list'], method=method)
            cor_2, p_value_2 = pearsonr(data_clean['acc_list'], data_clean['ms_list'], method=method)
            cor_3, p_value_3 = pearsonr(data_clean['acc_list'], data_clean['lscd_list'], method=method)
            cor_4, p_value_4 = pearsonr(data_clean['ms_list'], data_clean['sc_list'])
            cor_5, p_value_5 = pearsonr(data_clean['sc_list'], data_clean['acc_list'])
        else:
            pass
        
        print("Correlation MS v/s LSCD: ", cor_1, p_value_1)
        print("Correlation Acc v/s MS: ", cor_2, p_value_2)
        print("Correlation Acc v/s LSCD: ", cor_3, p_value_3)
        print("Correlation MS v/s SC: ", cor_4, p_value_4)
        print("Correlation SC v/s Acc: ", cor_5, p_value_5)

        
        analysis_data = dict(
                    Outlier_method = filter_method,
                    Correlation_Coeffiecient=correlation_method,
                    Correlation_Data=["MS v/s LSCD", ["MS v/s Accuracy"], ["LSCD v/s Accuracy"], ["MS v/s SC"], ["SC v/s Accuracy"]],
                    Correlation_Values=[round(float(cor_1),3), round(float(cor_2),3), round(float(cor_3),3), round(float(cor_4),3), round(float(cor_5),3)],
                    p_value=[round(float(p_value_1),6), round(float(p_value_2),6), round(float(p_value_3),6), round(float(p_value_4),3), round(float(p_value_5),3)])
        
        full_data = dict(
                    MS_list = list(data_clean["ms_list"]),
                    LSCD_list = list(data_clean["lscd_list"]),
                    ACC_list = list(data_clean["acc_list"]),
                    SC_list = list(data_clean["sc_list"])
                    )

        data_key = str(i) + "_" + filter_method + "_" + correlation_method
        list_key = str(i) + "_" + filter_method + "_"+ correlation_method + "_data" 
        
        filtered_dict.update({data_key: analysis_data})
        filtered_dict_list.update({list_key: full_data})

    data = {}
    log_filepath_data = Path(str(log_filepath) + filter_method + ".json")
    log_filepath_list = Path(str(log_filepath) + filter_method + "list.json")

    if os.path.exists(log_filepath_data):
        with open(log_filepath_data) as json_file:
            data = json.load(json_file)
            del data
            data = filtered_dict
    else:
            data = filtered_dict

    with open(log_filepath_data, "w") as json_file:
        json.dump(data, json_file, indent=4)
        print("Results written to:", log_filepath_data)        
    
    if os.path.exists(log_filepath_list):
        with open(log_filepath_list) as json_file:
            data = json.load(json_file)
            del data
            data = filtered_dict_list
    else:
            data = filtered_dict_list

    with open(log_filepath_list, "w") as json_file:
        json.dump(data, json_file, indent=4)
        print("Results written to:", log_filepath_list)  
    
    return filtered_dict 