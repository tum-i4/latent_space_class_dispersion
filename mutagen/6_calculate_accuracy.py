import argparse
import configparser
import json
import time
from typing import List, Protocol, Union, Iterable, Tuple, NamedTuple, Optional
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import duckdb
from collections import defaultdict
import datetime  
import eval_util
from all_classes import *
import importlib
import sys

def load_config(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser(default_section="General")
    config.read(next(path.glob("*.ini")))
    return config

def load_module(module_name: str, filepath: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, filepath.resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    print("Loaded module", module_name)

    return module


def load_mutant(mutant_folder: Path) -> Mutant:
    config = load_config(mutant_folder)
    sut_module = load_module(
        ".".join([mutant_folder.name, "eval"]), Path(config["General"]["eval"])
    )
    sut = sut_module.SUT(mutant_folder, device="cuda")
    model_train_data = Path((mutant_folder / "data.link").read_text())

    sut_name, sut_training = mutant_folder.name.rsplit("_", maxsplit=1)
    sut_training = int(sut_training)

    with (mutant_folder / "training.json").open() as fp:
        obj = json.load(fp)

    return Mutant(
        sut_name,
        sut_training,
        mutant_folder,
        model_train_data,
        config,
        sut,
        obj["duration"],
    )


def create_tensor_dataset(path: Path, split: str, has_path: bool = True):
    data = torch.load(path, "cpu")
    if has_path:
        return Data(
            data[f"img_{split}"].to("cuda"), data[f"lbl_{split}"], data[f"path_{split}"]
        )
    return Data(
        data[f"img_{split}"].to("cuda"),
        data[f"lbl_{split}"],
        torch.arange(len(data[f"lbl_{split}"])),
    )


def create_hive_folder(
    base_path: Path, sut_name: str, sut_training: int, dataset_name: str
):
    data_path = (
        base_path
        / f"dataset={dataset_name}"
        / f"sut_name={sut_name}"
        / f"sut_training={sut_training}"
    )
    data_path.mkdir(exist_ok=True, parents=True)

    return data_path, data_path / f"{dataset_name}.parquet"


def print_progresses(*items: Tuple[int, int], width: int = 20):
    for current, total in items:
        percentage = current / total
        print(f"[{'=' * round(percentage * width):<{width}}]", end="")


def batched(data: Data, batch_size: int):
    length = len(data.lbl)
    for i in range(0, length, batch_size):
        end = min(length, i + batch_size)

        yield data.img[i:end], data.lbl[i:end], data.ids[i:end], i, end


def run_eval(mutant: Mutant, data: Data, dataset_name: str):
    start = time.time()
    print(
        f"[{mutant.training:03d}][{mutant.name:<30}] {dataset_name:15}",
        end="",
        flush=True,
    )

    p_classes = torch.empty_like(data.lbl)
    comparison = torch.empty_like(data.lbl, dtype=torch.bool)
    probabilities = torch.empty_like(data.lbl, dtype=torch.float32)
    latent_space_vectors = torch.empty((len(data.lbl), num_classes), dtype=torch.float32)

    with mutant.sut:
        for img, label, sample_ids, i, end in batched(data, batch_size=2048):
            p, classes, outputs = mutant.sut.execute_raw(img)
            classes = classes.cpu()

            p_classes[i:end] = classes
            comparison[i:end] = label == classes
            probabilities[i:end] = p
            latent_space_vectors[i:end] = outputs
    
    print("Accuracy:", comparison.sum()/data.lbl.shape[0] * 100)
    
    test_df = pd.DataFrame.from_dict(
        {
            "sut_name": mutant.name,
            "sut_training": mutant.training,
            "dataset": dataset_name,
            "sample_id": data.ids,
            "label": data.lbl,
            "output": p_classes,
            "result": comparison,
            "confidence": probabilities,
            "latent_space": [v.tolist() for v in latent_space_vectors],
        }
    )
    dur = time.time() - start
    test_df["training_time"] = mutant.training_time
    test_df["evaluation_time"] = dur
    test_df["is_duplicate"] = False

    print(f"took {dur: 7.2f}s", end=" ")
    return test_df


def write_dataframe(
    df: pd.DataFrame,
    eval_path: Path,
    mutant: Mutant,
    dataset_name: str,
    duplicate: Optional[str] = None,
):
    start = time.time()

    sub_folder, outfile = create_hive_folder(
        eval_path, mutant.name, mutant.training, dataset_name
    )

    with duckdb.connect() as db_con:
        db_con.sql(eval_util.TEST_RESULT_TABLE_CREATION)
        db_con.execute("INSERT INTO test_results SELECT * FROM df")
        db_con.sql("SELECT * FROM test_results").write_parquet(
            str(outfile), compression=None
        )

        print(f"| export took {time.time() - start: 7.2f}s", end="")
        start = time.time()

        if duplicate:
            sub_folder, outfile = create_hive_folder(
                eval_path, mutant.name, mutant.training, duplicate
            )

            db_con.execute(
                "UPDATE test_results SET dataset = $1, is_duplicate=true;",
                parameters=[duplicate],
            )
            db_con.sql("SELECT * FROM test_results").write_parquet(
                str(outfile), compression=None
            )

            print(f" | duplicate took {time.time() - start: 7.2f}s", end="")

    print()


def eval(result_path_1: Path, dataset: str, dataset_path: str, dataset_cc_path:str, dataset_mixed_path:str, dataset_mixed_1_path:str):
    result_path = Path.cwd() / 'results' / result_path_1 
    folders = sorted(
        [
            folder
            for folder in (result_path / "trained_mutants").iterdir()
            if folder.is_dir()
        ]
    )
    original_data_set = Path(load_config(folders[0])["General"]["data"])
    mutants = [load_mutant(folder) for folder in folders]
    
    if dataset == "mnist" or dataset == "svhn" or dataset == "gtsrb_gray" or dataset == "gtsrb":
        data_sets = {
            "train": (
                dataset_path,
                "train",
            ),
            "valid": (
                dataset_path,
                "valid",
            ),
            "test": (
                dataset_path,
                "test",
            ),
            "cc_nc": (
                dataset_cc_path,
                "cc_nc",
            ),

            "cc_kmnc": (
                dataset_cc_path,
                "cc_kmnc",
            ),

            "cc_nbc": (
                dataset_cc_path,
                "cc_nbc",
            ),

            "mixed": (
                dataset_mixed_path,
                "merged",
            ),

            # "mixed_1": (
            #     dataset_mixed_1_path,
            #     "merged",
            # ),
        }
    eval_path = result_path / "evaln"
    eval_path.mkdir(exist_ok=True, parents=True)

    mutants_datasets = defaultdict(list)

    print(type(mutants_datasets))

    for di, (dataset_name, data_set_args) in enumerate(data_sets.items()):
        data_set = create_tensor_dataset(*data_set_args) # Path, split_name

        for i, mutant in enumerate(mutants):
            print_progresses((di, len(data_sets)), (i, len(mutants)))

            if create_hive_folder(
                eval_path, mutant.name, mutant.training, dataset_name
            )[1].exists():
                print(
                    f"[{mutant.training:03d}][{mutant.name:<30}] {dataset_name:15}exists"
                )
                continue

            dupl = ""
            if dataset_name == "otrain":
                if mutant.train_set_path == original_data_set:
                    dupl = "mtrain"
                else:
                    mutants_datasets[mutant.train_set_path].append(mutant)

            results = run_eval(mutant, data_set, dataset_name)
            write_dataframe(results, eval_path, mutant, dataset_name, dupl)

        del data_set

    for di, (dataset_path, mutant_list) in enumerate(mutants_datasets.items()):
        data_set = create_tensor_dataset(dataset_path, split="train", has_path=False)

        for i, mutant in enumerate(mutant_list):
            print_progresses((di, len(mutants_datasets)), (i, len(mutant_list)))

            if create_hive_folder(eval_path, mutant.name, mutant.training, "mtrain")[
                1
            ].exists():
                print(f"[{mutant.training:03d}][{mutant.name:<30}] {'mtrain':15}exists")
                continue

            results = run_eval(mutant, data_set, "mtrain")
            write_dataframe(results, eval_path, mutant, "mtrain")

        del data_set


def arguments():
    op_dir = "gtsrb_1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        default=op_dir,
        help="Result Path",
    )
    parser.add_argument("--model", default="gtsrb", help="Model Type")

    return parser.parse_args()


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    args = arguments()
    dataset = "gtsrb"
    num_classes = 43
    dataset_path = Path("data", dataset, dataset+"_org_data_normalized.pth")
    dataset_cc_path = Path("data", dataset, dataset+"_cc_data_normalized_prob_valid.pth")
    dataset_mixed_path = Path("data", dataset, dataset+"_mixed_data_normalized_valid.pth")
    # dataset_mixed_1_path = Path("data", dataset, dataset+"_mixed_1_data_normalized.pth")
    eval(result_path_1=args.result_dir, dataset=dataset, dataset_path=dataset_path, dataset_cc_path=dataset_cc_path, dataset_mixed_path= dataset_mixed_path, dataset_mixed_1_path= None)

    print("\nFinished! Took", datetime.datetime.now() - start_time)
