from pathlib import Path
import torch
import numpy as np
import pandas as pd
import sys
import path_setup
import datetime
from dotenv import load_dotenv
import pickle
import os
import math

load_dotenv(override=True)

sys.path.append(os.path.join(os.getcwd(), "mutagen/utils"))

from utils.lscd_ms_utils import *
from utils.plot_utils import *
from utils.surprise_adequacy_utils import *
from utils.load_mutants import load_mutant_model

outlier_methods = ["IQR"]  # , "IsolationForest"
selected_splits = ["test", "cc_nc", "cc_kmnc", "cc_nbc", "mixed"]
selected_splits_analyze = ["test", "cc_nc", "cc_kmnc", "cc_nbc", "mixed"]


def correlation_analysis(ms_list, lscd_list, acc_list, sc_list):
    np.random.seed(0)
    correlation_methods = ["Pearson"]

    data_clean = pd.DataFrame(
        {
            "ms_list": ms_list,
            "lscd_list": lscd_list,
            "acc_list": acc_list,
            "sc_list": sc_list,
        }
    )

    rng = np.random.default_rng()

    for i, correlation_method in enumerate(correlation_methods):
        if correlation_method == "Spearman":
            cor_1, p_value_1 = spearmanr(data_clean["ms_list"], data_clean["lscd_list"])
            cor_2, p_value_2 = spearmanr(data_clean["acc_list"], data_clean["ms_list"])
            cor_3, p_value_3 = spearmanr(
                data_clean["acc_list"], data_clean["lscd_list"]
            )
            cor_4, p_value_4 = spearmanr(data_clean["ms_list"], data_clean["sc_list"])
            cor_5, p_value_5 = spearmanr(data_clean["sc_list"], data_clean["acc_list"])
        elif correlation_method == "Pearson":
            # method = stats.MonteCarloMethod(rvs=(rng.uniform, rng.uniform))
            method = stats.PermutationMethod(n_resamples=50, random_state=rng)
            cor_1, p_value_1 = pearsonr(
                data_clean["ms_list"], data_clean["lscd_list"], method=method
            )
            cor_2, p_value_2 = pearsonr(
                data_clean["acc_list"], data_clean["ms_list"], method=method
            )
            cor_3, p_value_3 = pearsonr(
                data_clean["acc_list"], data_clean["lscd_list"], method=method
            )
            cor_4, p_value_4 = pearsonr(data_clean["ms_list"], data_clean["sc_list"])
            cor_5, p_value_5 = pearsonr(data_clean["sc_list"], data_clean["acc_list"])
        else:
            pass

        print("Correlation MS v/s LSCD: ", cor_1, p_value_1)
        print("Correlation Acc v/s MS: ", cor_2, p_value_2)
        print("Correlation Acc v/s LSCD: ", cor_3, p_value_3)
        print("Correlation MS v/s SC: ", cor_4, p_value_4)
        print("Correlation SC v/s Acc: ", cor_5, p_value_5)

        return cor_4, cor_5


def get_sc_correlation(ms_dict, lscd_dict, acc_dict, sc_dict, log_filepath):
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

        sc_ms_cor, sc_acc_cor = correlation_analysis(
            ms_list=ms_final_list,
            lscd_list=lscd_final_list,
            acc_list=acc_final_list,
            sc_list=sc_final_list,
        )
    
    return sc_ms_cor, sc_acc_cor
