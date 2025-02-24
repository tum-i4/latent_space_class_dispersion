import argparse
import copy
import importlib
import importlib.util
import random
import shutil
import sys
import os
import inspect
import configparser
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Type, Dict, List
import ast

import torch
import torch.nn as nn

import path_setup
from dotenv import load_dotenv
load_dotenv(override=True)


def load_model_from_script(scripted_model: torch.jit.ScriptModule, model_file: Path) -> nn.Module:
    spec = importlib.util.spec_from_file_location("model_module", model_file)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = model_module
    spec.loader.exec_module(model_module)

    model_class = next(cls for name, cls in inspect.getmembers(model_module, inspect.isclass) if issubclass(cls, nn.Module))
    model_instance = model_class()
    model_instance.load_state_dict(scripted_model.state_dict())
    return model_instance

class MutOp(NamedTuple):
    name: str
    typ: str
    op_cls: Type
    init_kwargs: Dict
    
class Mutant:
    name: str
    files: Dict[str, Path] # model, model_params, eval, config - collect from ini config file
    model: nn.Module
    history: List[MutOp]
    ast_model_py: ast.Module
    
    def __init__(self, name: str, files: Dict[str, Path]):
        self.name = name
        self.files = files
        scriptModel = torch.jit.load(str(files['model_params']))
        self.model = load_model_from_script(scriptModel, files['model'])
        self.history = []
        self.ast_model_py = ast.parse(files['model'].read_text())
        
    def create_copy(self, new_name: str) -> "Mutant":
        copied = copy.deepcopy(self)
        copied.name = new_name
        return copied
    
    def apply(self, *mut_ops: MutOp):
        for mut_op in mut_ops:
            self.history.append(mut_op)
            self.model, self.ast_model_py = mut_op.op_cls(**mut_op.init_kwargs).mutate(self.model, self.ast_model_py)
    
    def save(self, mutation_folder: Path):
        # print("Saving", self.name)

        own_folder = mutation_folder / self.name
        own_folder.mkdir(parents=True, exist_ok=True)

        for mut_op in self.history:
            print(f"> Applied mutation op {mut_op.name} with args: {mut_op.init_kwargs.items()}")

        # Save the model as a TorchScript module
        # scripted_model = torch.jit.script(self.model)
        # scripted_model.save(str(own_folder / 'model_script.pth'))
        torch.save(self.model.state_dict(), str(own_folder / 'model.pth'))
        
        shutil.copy(self.files['config'], own_folder)
        shutil.copy(self.files['eval'], own_folder)
        #shutil.copy(self.files['model'], own_folder)
        with open(own_folder / 'model.py', 'w', encoding='utf-8') as f:
            f.write(ast.unparse(self.ast_model_py))

def get_mutation_ops() -> Dict[str, MutOp]:
    all_ops = []
    path = Path("mutagen/mutation_ops") if not post_training else Path("mutagen/mutation_ops/post_training")
    for file in path.glob("*.py"):
        module_name = f"mutop.{file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file.resolve())
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        classes = [v for k, v in inspect.getmembers(module, inspect.isclass) if v.__module__ == module_name and not v.__name__.startswith("_")]
        all_ops.extend(MutOp(c.__name__, c.mutation_typ, c, {}) for c in classes) # init kwargs will be filled in main

    return {m.name: m for m in all_ops}

def main(config_file: str, op_dir_name: str):
    ops = get_mutation_ops()
    print("Found mutation ops: ", ", ".join(ops.keys()))

    print("Reading config file", config_file)
    config = configparser.ConfigParser()
    config.read(config_file)
    
    files = {k: Path(config["General"][k]) for k in ("model", "eval", "model_params")}
    files['config'] = config_file
    
    seed = config['General'].getint('seed', fallback=None) # TODO: want to change the fallback?
    if seed:
        random.seed(seed)
        torch.manual_seed(seed)
        
    mutations: Dict[str, List[configparser.SectionProxy]] = defaultdict(list)
    for name, section in config.items():
        if name == "General" or name == "DEFAULT":
            continue

        if "." not in name: # if you want to apply multiple operators on the same Mutant => use [MutationName.operatorName] in config # operatorName won't be used
            mutations[name] = [section]
            continue
        
        parent_name, sub_name = name.split(".", maxsplit=1)
        mutations[parent_name].append(section)

    folder_suffix = "trained_mutants" if not post_training else "post_training_mutants"
    output_folder = Path.cwd() / 'results' / op_dir_name / folder_suffix
    output_folder.mkdir(parents=True, exist_ok=True)
    
    original = Mutant("AAA_Original", files) # files are the files we want to copy into the mutant folder
    original.save(output_folder)
    
    for name, mutation_sections in mutations.items():
        all_mutation: List[MutOp] = []

        for section in mutation_sections: # sections that have the same name until the first dot => they are applied together to the same mutant
            mut_op = ops[section["mutation"]]
            for k, v in section.items():
                if k != "mutation": 
                    mut_op.init_kwargs[k] = v # all other fields are passed as kwargs to the mutation class constructor
            
            all_mutation.append(mut_op)
     
        new_mutant: Mutant = original.create_copy(f"{name}")
        print("Applying", new_mutant.name)

        num_params_before = sum(p.numel() for p in new_mutant.model.parameters())
        print(f"> Number of parameters before mutation: {num_params_before}")

        new_mutant.apply(*all_mutation)

        num_params_after = sum(p.numel() for p in new_mutant.model.parameters())
        print(f"> Number of parameters after mutation: {num_params_after}")

        new_mutant.save(output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_file', default=os.getenv("MUT_CONFIG_FILE", "mutagen/configs/mnist/lenet_post.ini"))

    args = parser.parse_args()

    op_dir = os.getenv('OP_DIR', 'mnist_1')
    
    post_training = os.getenv('POST_TRAINING', 'True').lower() == 'true'

    main(args.config_file, op_dir_name = op_dir)