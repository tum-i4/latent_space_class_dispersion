import os

import matplotlib.pyplot as plt
import seaborn
from pathlib import Path
import numpy as np

colors = ["steelblue", "orchid", "lightsalmon", "cyan", "burlywood", "magenta"]
splits_list = ["Test Data", "Corner Cases \n (NC)", "Corner Cases\n (k-MNC)", "Corner Cases \n (NBC)", "Combined Data"]

params = {'font.size': 21,
          'figure.figsize': (12, 9), #10,8 GTSRB
         'axes.labelsize': 21,
         'axes.titlesize':24}
plt.rcParams.update(params)

def plot_ms_lscd_scatter(ms_dict, lscd_dict, splits, dataset, filter):
    plt.clf()
    fig, ax = plt.subplots()
    for i in range(len(ms_dict)):
        key = list(ms_dict.keys())[i]
        ms_list = ms_dict[key]
        lscd_list = lscd_dict[key]
        plt.scatter(ms_list, lscd_list, color=colors[i], label=splits_list[i])
    
    # plt.title('Mutation Score v/s LSCD values \n for {} dataset'.format(dataset), loc="center")
    plt.title('{} Dataset'.format(dataset), loc="center")
    plt.xlabel('Mutation Score')
    plt.ylabel('LSCD')
    plt.legend(loc='upper left', markerscale=2)
    # ax.set_xlim(-0.1, 1.1)
    # plt.xticks(np.arange(-0.1, 1.1, 0.2))
    plt.tight_layout()
    file_name = Path.cwd() / "plots/new" /("lscd_ms1_" + dataset + "_" + filter +".pdf")
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    plt.savefig(file_name, dpi=600, bbox_inches="tight")


def plot_ms_sc_scatter(ms_dict, sc_dict_plot, splits, dataset, filter):
    plt.clf()
    fig, ax = plt.subplots()
    for i in range(len(ms_dict)):
        key = list(ms_dict.keys())[i]
        ms_list = ms_dict[key]
        sc_list = sc_dict_plot[key]
        plt.scatter(ms_list, sc_list, color=colors[i], label=splits_list[i])
    
    # plt.title('Mutation Score v/s Surprise Coverage (SC) values \n for {} dataset'.format(dataset), loc="center")
    plt.title('{} Dataset'.format(dataset), loc="center")
    plt.xlabel('Mutation Score')
    plt.ylabel('Distance-based SC')
    plt.legend(loc='upper left', markerscale=2)
    # ax.set_xlim(-0.1, 1.1)
    # plt.xticks(np.arange(-0.1, 1.1, 0.2))
    plt.tight_layout()
    file_name = Path.cwd() / "plots/new" /("sc_ms1_" + dataset + "_" + filter +".pdf")
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    plt.savefig(file_name, dpi=600, bbox_inches="tight")