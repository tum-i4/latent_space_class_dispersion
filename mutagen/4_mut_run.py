import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import random
import shutil
import sys
import inspect
import itertools
import configparser
import io
import torch
import sys

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Callable, Union, Type, Dict, List
from mut_base import MutationTransformer


class MutOp(NamedTuple):
    name: str
    typ: str
    op: Union[Type, Callable]

    def get_default_values(self):
        return self.default_values_method() if self.typ == 'data' else self.default_values_class()

    def default_values_class(self):
        annotations = inspect.get_annotations(self.op)
        default_values = {k: getattr(self.op, k) for k in annotations if hasattr(self.op, k)}

        return annotations, default_values

    def default_values_method(self):
        annotations = inspect.get_annotations(self.op)
        annotations.pop('data')

        signature = inspect.signature(self.op)
        default_values = {k: v.default for k, v in signature.parameters.items() if v.default != inspect.Signature.empty}

        return annotations, default_values


class MutOpInstance(NamedTuple):
    name: str
    op: MutOp
    init_kwargs: Dict

    def to_json(self) -> Dict:
        return {
            "name": self.name,
            "operator": {"name": self.op.op.__name__, "typ": self.op.typ},
            "kwargs": self.init_kwargs
        }


@lru_cache
def hash_file_content_cached(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Mutant:
    name: str
    trees: Dict[str, ast.AST]
    files: Dict[str, Path]
    data: Dict[str, torch.Tensor]

    history: Dict[str, List[MutOpInstance]]
    _content_hash: Dict[str, str]

    def __init__(self, name: str, files: Dict[str, Path]):
        self.name = name
        self.files = files
        self.trees = {k: ast.parse(files[k].read_text()) for k in ("model", "eval", "train")}
        self.data = {}

        self.history = {k: [] for k in ("model", "eval", "train", "data")}
        self._content_hash = {}

    def content_hash(self) -> Dict[str, str]:
        if self._content_hash:
            return self._content_hash

        full_hash = hashlib.sha256()

        def hash_ast(tree: ast.AST):
            text = ast.unparse(tree).encode("utf8")
            full_hash.update(text)
            return hashlib.sha256(text).hexdigest()

        all_hashes = {k: hash_ast(tree) for k, tree in self.trees.items()}

        if self.data:
            fp = io.BytesIO()
            self.save_data(fp)
            data_hash = hashlib.sha256(fp.getbuffer()).hexdigest()
        else:
            data_hash = hash_file_content_cached(self.files['data'].resolve())

        full_hash.update(data_hash.encode("utf-8"))
        all_hashes['data'] = data_hash
        all_hashes['full'] = full_hash.hexdigest()

        self._content_hash = all_hashes
        return all_hashes

    def load_data(self):
        if not self.data:
            self.data = torch.load(self.files['data'], 'cpu')

    def create_copy(self, new_name: str) -> "Mutant":
        copied = copy.deepcopy(self)
        copied.name = new_name
        return copied

    def apply(self, *mut_op_instances: MutOpInstance):
        self._content_hash = {}
        for mut_op in mut_op_instances:
            self.history[mut_op.op.typ].append(mut_op)

            if mut_op.op.typ == 'data':
                self.apply_data_mutation(mut_op)
            else:
                self.apply_ast_mutation(mut_op)

    def apply_ast_mutation(self, mut_op_instance: MutOpInstance):
        tree = self.trees[mut_op_instance.op.typ]
        output_ast = mut_op_instance.op.op(mut_op_instance.init_kwargs).visit(tree)
        self.trees[mut_op_instance.op.typ] = ast.fix_missing_locations(output_ast)

    def apply_data_mutation(self, mut_op_instance: MutOpInstance):
        self.load_data()
        mut_op_instance.op.op(self.data, **mut_op_instance.init_kwargs)

    def save_data(self, data_file: Union[Path, io.BytesIO]):
        torch.save(self.data, data_file)

    def save(self, mutation_folder: Path, cache_folder: Path):
        print("Saving", self.name)

        own_folder = mutation_folder / self.name
        own_folder.mkdir(parents=True, exist_ok=False)

        for k in self.trees:
            out_file = own_folder / self.files[k].name

            comment = [
                f"# Applied mutation op {mut_op.op.name} with args: {mut_op.init_kwargs.items()}"
                for mut_op in self.history[k]
            ]

            out_text = "\n".join(comment) + "\n\n"
            out_text += ast.unparse(self.trees[k])
            out_file.write_text(out_text)

        shutil.copy(self.files['config'], own_folder)

        hashes = self.content_hash()

        data_file = own_folder / "data.link"
        if self.data:
            cache_file = cache_folder / f"{hashes['data']}.pth"
            data_file.write_text(cache_file.as_posix())

            if not cache_file.exists():
                self.save_data(cache_file)
        else:
            data_file.write_text(self.files['data'].as_posix())

        with (own_folder / "meta.json").open("w") as fp:
            json_history = {k: [moi.to_json() for moi in moi_list] for k, moi_list in self.history.items()}
            json.dump({"hashes": hashes, "mutation_history": json_history}, fp, indent=2)

        self.data = {}


def get_mutation_ops() -> Dict[str, MutOp]:
    all_ops = []

    for file in Path("mutagen/mutation_ops").glob("*.py"):
        module_name = f"mutop.{file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file.resolve())
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        classes = [v for k, v in inspect.getmembers(module, inspect.isclass) if v.__module__ == module_name]
        all_ops.extend(MutOp(c.__name__, c.mutation_typ, c) for c in classes if issubclass(c, MutationTransformer))

        funcs = [v for k, v in inspect.getmembers(module, inspect.isfunction) if not k.startswith("_")]
        all_ops.extend(MutOp(f.__name__, "data", f) for f in funcs)

    return {m.name: m for m in all_ops}


def main(config_file: str, op_dir_name: str):
    ops = get_mutation_ops()
    print("Found mutation ops: ", ", ".join(ops.keys()))

    print("Reading config file", config_file)
    config = configparser.ConfigParser(default_section="General")
    config.read(config_file)

    files = {k: Path(config["General"][k]) for k in ("model", "eval", "train", "data")}
    files['config'] = config_file

    seed = config['General'].getint('seed', fallback=688446987)
    if seed:
        random.seed(seed)
        torch.manual_seed(seed)

    mutations: Dict[str, List[configparser.SectionProxy]] = {}
    for name, section in config.items():
        if name == "General":
            continue

        if "." not in name:
            mutations[name] = [section]
            continue

        parent_name, sub_name = name.split(".", maxsplit=1)
        mutations[parent_name].append(section)

    output_folder = Path.cwd() / 'results' / op_dir_name / 'raw_mutants'
    output_folder.mkdir(parents=True)

    cache_folder = Path.cwd() / 'data' / 'cache'
    cache_folder.mkdir(exist_ok=True, parents=True)
    cache_folder_relative = cache_folder.relative_to(Path.cwd())

    original = Mutant("AAA_Original", files)
    original.save(output_folder, cache_folder_relative)
    mutants = {original.content_hash()['full']: original}

    for name, mutation_sections in mutations.items():
        all_mutation: List[List[MutOpInstance]] = []
        max_len = -1

        for section in mutation_sections:
            print(section)
            if section["mutation"] == 'high_order':
                max_len = section.getint("sample", fallback=-1)
                continue

            mut_op = ops[section["mutation"]]
            annotations, default_values = mut_op.get_default_values()

            all_values = {}
            for parameter_name, parameter_type in annotations.items():
                if parameter_name in section:
                    values = [parameter_type(str_v.strip()) for str_v in section[parameter_name].split(",")]
                else:
                    values = [default_values[parameter_name]]
                all_values[parameter_name] = values

            sub_mutations = []
            for i, values in enumerate(itertools.product(*all_values.values())):
                init_kwargs = dict(zip(all_values.keys(), values))
                sub_mutations.append(MutOpInstance(section.name, mut_op, init_kwargs))
            all_mutation.append(sub_mutations)

        sub_mutant_count = 0

        full_product = [m for m in itertools.product(*all_mutation)]
        random.shuffle(full_product)

        for mutations in full_product:
            new_mutant: Mutant = original.create_copy(f"{name}_{sub_mutant_count}")
            new_mutant.apply(*mutations)

            full_hash = new_mutant.content_hash()['full']
            if full_hash in mutants:
                continue
            mutants[full_hash] = new_mutant

            new_mutant.save(output_folder, cache_folder_relative)

            sub_mutant_count += 1
            if 0 < max_len <= sub_mutant_count:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_file', default="mutagen/configs/gtsrb_new/gtsrb_new_merged.ini")

    args = parser.parse_args()

    op_dir = "gtsrb_1"

    main(args.config_file, op_dir_name = op_dir)
