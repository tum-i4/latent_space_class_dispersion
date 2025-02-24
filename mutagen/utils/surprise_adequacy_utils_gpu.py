import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
import os
import torch
from pathlib import Path
import numpy as np
import json
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
from lscd_ms_utils import *
from models.mnist.lenet5.model import Net
import pandas as pd
from tqdm import tqdm
from plot_utils import *
from scipy import stats
from torch.utils.data import DataLoader, Dataset 
import torch.multiprocessing as mp

class CustomDataset(Dataset):
    def __init__(self, layers_outputs, output_classes):
        self.layers_outputs = layers_outputs
        self.output_classes = output_classes

    def __len__(self):
        return len(self.output_classes)

    def __getitem__(self, idx):
        return self.layers_outputs[idx], self.output_classes[idx]
    

def calculate_activation_traces(gt_labels, output_labels, input_data_frame):
    at_dictionary = {}
    sut_name = input_data_frame["sut_name"]
    print(
        "Calculating the activation traces for .",
        sut_name[0],
    )

    latent_space_vectors = np.vstack([a for a in input_data_frame["latent_space"]])
    predictions = np.array(input_data_frame["output"])

    print("Length of ATs:", latent_space_vectors.shape)

    at_dictionary.update({"AT": latent_space_vectors})
    at_dictionary.update({"predictions": predictions})

    return at_dictionary


def get_target_pred(layers_outputs):
    # Identify the last layer dynamically
    last_layer_name = list(layers_outputs.keys())[-1]
    # Extract the output of the final fully connected layer
    last_layer_output = layers_outputs[last_layer_name]

    # Apply softmax to get probabilities (optional, depending on if the model already applies it)
    probabilities = F.softmax(last_layer_output, dim=1)
    # Get the class prediction (index of the highest probability)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class  # .numpy()


def find_closest_at(at, train_ats):
    """The closest distance between subject AT and training ATs.

    Args:
        at (list): List of activation traces of an input.
        train_ats (list): List of activation traces in training set (filtered)

    Returns:
        dist (int): The closest distance.
        at (list): Training activation trace that has the closest distance.
    """

    # dist = np.linalg.norm(at - train_ats, axis=1)
    # return (min(dist), train_ats[np.argmin(dist)])
    dist = torch.norm(at - train_ats, dim=1)
    min_dist, min_index = torch.min(dist, dim=0)
    closest_train_at = train_ats[min_index]

    return (min_dist, closest_train_at)

def cal_dsa_worker(train_ats, train_pred, layers_output, target_pred, device):
    layer_output = torch.tensor([layers_output]).to(device)

    if layer_output[0].ndim == 3:  # For convolutional layers
        layer_matrix = torch.mean(layer_output, dim=(-1, -2)).to(device)  # Aggregated to (channels,)
    else:  # For fcn layers
        layer_matrix = layer_output.to(device)

    target_ats = layer_matrix

    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label.item() not in class_matrix:
            class_matrix[label.item()] = []
        class_matrix[label.item()].append(i)
        all_idx.append(i)

    dsa_list_img_wise = []
    for i, at in enumerate(target_ats):
        label = target_pred
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[int(label)]])
        b_dist, _ = find_closest_at(a_dot, train_ats[list(set(all_idx) - set(class_matrix[int(label)]))])
        dsa_list_img_wise.append(a_dist / b_dist)

    return dsa_list_img_wise

def cal_dsa(train_act_traces, layers_outputs, output_classes):
    """Surprise-adequacy coverage for 1 input
    Args:
        layers_outputs(OrderedDict): The outputs of internal layers for a batch of mutants
    Returns:
        ptr(Tensor): array that records the coverage information
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ats = torch.tensor(train_act_traces["AT"]).to(device) # (#samples, 10 (op_layers))
    train_pred = torch.tensor(train_act_traces["predictions"]).to(device) # (#samples,)
    # Get the last 10 columns
    # train_ats = train_ats[:, -10:]
    
    dataset = CustomDataset(layers_outputs, output_classes)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    
    dsa_list_img_wise = []
    with mp.Pool(processes=4) as pool:
        results = []
        for layers_output, target_pred in tqdm(dataloader):
            layers_output = layers_output[0].numpy()
            target_pred = target_pred[0].item()
            result =  pool.apply_async(cal_dsa_worker, (train_ats, train_pred, layers_output, target_pred, device))
            results.append(result)

        for result in results:
            dsa_list_img_wise.extend(result.get())

    return dsa_list_img_wise

    # dsa_list_img_wise = []
    # for idx in tqdm(range(len(output_classes))):
    #     layer_output = torch.tensor([layers_outputs[idx]]).to(device)
    #     # CALCULATE TARGET PREDICTION
    #     target_ats = None
    #     target_pred = output_classes[idx]

    #     if layer_output[0].ndim == 3: # For convolutional layers
    #         # layer_matrix = np.array(torch.mean(layer_output, dim=(-1, -2)))  # Aggregated to (channels,)
    #         layer_matrix = torch.mean(layer_output, dim=(-1, -2)) # Aggregated to (channels,)
    #     else: # For fcn layers
    #         # layer_matrix = np.array(layer_output)
    #         layer_matrix = torch.tensor(layer_output)

    #     if target_ats is None:
    #         target_ats = layer_matrix
    #     else:
    #         target_ats = np.append(target_ats, layer_matrix, axis=1)
    #         target_ats = torch.cat((target_ats, layer_matrix), axis=1)
    #         layer_matrix = None

    #     # Fetch Distance-based surprise adequacy
    #     class_matrix = {}
    #     all_idx = []
    #     for i, label in enumerate(train_pred):
    #         label = int(label)
    #         if label not in class_matrix:
    #             class_matrix[label] = []
    #         class_matrix[label].append(i)
    #         all_idx.append(i)

    #     # dsa = []

    #     #print("Fetching DSA")
    #     # Convert target_pred to numpy array
    #     for i, at in enumerate(target_ats):
    #         label = target_pred # [i]
    #         a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[int(label)]])
    #         b_dist, _ = find_closest_at(
    #             a_dot, train_ats[list(set(all_idx) - set(class_matrix[int(label)]))]
    #         )
    #         dsa_list_img_wise.append(a_dist / b_dist)
        
    # # dsa_img = np.array([dsa_list_img_wise])

    # return dsa_list_img_wise 


def get_sc(lower, upper, k, sa):
    """Surprise Coverage

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


def calculate_surprise_adequacy(train_act_traces, input_data):

    dsa_total = cal_dsa(
        train_act_traces, input_data["latent_space"], input_data["output"]
    )
    surprise_coverage = get_sc(
        lower=np.amin(dsa_total), upper=np.amax(dsa_total), k=100, sa=dsa_total
    )
    return dsa_total, surprise_coverage


def correlation_analysis_sc(
    ms_list, acc_list, sc_list, log_filepath, filter_method
):
    np.random.seed(0)
    # outlier_methods = ["IQR", "IsolationForest"]
    correlation_methods = ["Spearman", "Pearson"]
    filtered_dict, filtered_dict_list = {}, {}

    data_clean = pd.DataFrame(
        {"ms_list": ms_list, "sc_list": sc_list, "acc_list": acc_list}
    )

    rng = np.random.default_rng()

    for i, correlation_method in enumerate(correlation_methods):
        if correlation_method == "Spearman":
            cor_1, p_value_1 = spearmanr(data_clean["ms_list"], data_clean["sc_list"])
            cor_2, p_value_2 = spearmanr(data_clean["acc_list"], data_clean["ms_list"])
            cor_3, p_value_3 = spearmanr(data_clean["acc_list"], data_clean["sc_list"])
        elif correlation_method == "Pearson":
            # method = stats.MonteCarloMethod(rvs=(rng.uniform, rng.uniform))
            method = stats.PermutationMethod(n_resamples=50, random_state=rng)
            cor_1, p_value_1 = pearsonr(
                data_clean["ms_list"], data_clean["sc_list"], method=method
            )
            cor_2, p_value_2 = pearsonr(
                data_clean["acc_list"], data_clean["ms_list"], method=method
            )
            cor_3, p_value_3 = pearsonr(
                data_clean["acc_list"], data_clean["sc_list"], method=method
            )
        else:
            pass

        print("Correlation MS v/s SC: ", cor_1, p_value_1)
        print("Correlation Acc v/s MS: ", cor_2, p_value_2)
        print("Correlation Acc v/s SC: ", cor_3, p_value_3)

        # keys = ["ms_list", "lscd_list",  "acc_list"]

        # filtered_dict.update({"outlier_filter_method:": filter_method})
        # filtered_dict.update({"Correlation Coeffiecient:": correlation_method})
        # filtered_dict.update({"Correlation Values: ": round(float(cor_1),3)})

        analysis_data = dict(
            Outlier_method=filter_method,
            Correlation_Coeffiecient=correlation_method,
            Correlation_Data=["MS v/s SC", ["MS v/s Accuracy"], ["SC v/s Accuracy"]],
            Correlation_Values=[
                round(float(cor_1), 3),
                round(float(cor_2), 3),
                round(float(cor_3), 3),
            ],
            p_value=[
                round(float(p_value_1), 6),
                round(float(p_value_2), 6),
                round(float(p_value_3), 6),
            ],
        )

        full_data = dict(
            MS_list=list(data_clean["ms_list"]),
            SC_list=list(data_clean["sc_list"]),
            ACC_list=list(data_clean["acc_list"]),
        )

        # for key in keys:
        #     filtered_dict[key] = []
        #     filtered_dict[key].extend(list(data_clean[key]))

        data_key = str(i) + "_" + filter_method + "_" + correlation_method
        list_key = str(i) + "_" + filter_method + "_" + correlation_method + "_data"

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
