import argparse
import configparser
import importlib
import importlib.util
import shutil
import sys
import json
import datetime
from pathlib import Path

import torch
import sys

def load_module(module_name: str, filepath: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, filepath.resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    print("Loaded module", module_name)

    return module


def run_training(result_path_1: Path, num_training: int, op_dir_name: str):
    result_path = Path.cwd() / 'results' / result_path_1 
    output_folder = result_path / "trained_mutants"
    output_folder.mkdir(exist_ok=True)

    items = [folder for folder in (result_path / "raw_mutants").iterdir() if folder.is_dir()]

    orig_conf = json.loads((result_path / "raw_mutants" / "AAA_Original" / "meta.json").read_text())
    original_train_hash = orig_conf['hashes']['train']

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Currently using:", device)

    for mutant_folder in sorted(items):
        config = configparser.ConfigParser(default_section="General")
        config.read(next(mutant_folder.glob("*.ini")))

        files = {k: mutant_folder / Path(config["General"][k]).name for k in ("model", "eval", "train")}

        train_module = load_module(".".join([mutant_folder.name, "train"]), files['train'])
        model_module = load_module(".".join([mutant_folder.name, "model"]), files['model'])

        with (mutant_folder / "meta.json").open("r") as fp:
            conf = json.load(fp)

        pretrain = Path.cwd() / 'data' / 'pretrain' / f'{conf["hashes"]["model"]}_{original_train_hash}.pth'
        if not pretrain.exists():
            pretrain = None

        data_file = (mutant_folder / 'data.link').read_text()
        data = torch.load(data_file, device)

        for i in range(num_training):
            print(f"[{i:04d}] {mutant_folder.name}")
            training_folder = output_folder / f"{mutant_folder.name}_{i:03d}"
            if training_folder.exists():
                if all((training_folder / f).exists() for f in ("training.json", "model.pth")):
                    print("Mutant", training_folder.name, "already trained")
                    continue

                shutil.rmtree(training_folder)
            training_folder.mkdir()

            copied = [shutil.copy(f, training_folder) for f in mutant_folder.iterdir() if f.is_file()]

            net = model_module.Net() # Creates an instance of NN defined in model_module.
            net.to(device)
            train_module.train(net, data, training_folder / 'model.pth', pretrain=pretrain)
            del net
        del data
        torch.cuda.empty_cache()
        print("=" * 120)


if __name__ == '__main__':

    start_time = datetime.datetime.now()
    op_dir = "gtsrb_1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default=op_dir)
    parser.add_argument('--num_trainings', default=1, type=int)

    args = parser.parse_args()

    run_training(Path(args.result_dir), args.num_trainings, op_dir_name= op_dir)

    print("\nFinished! Took", datetime.datetime.now() - start_time)
