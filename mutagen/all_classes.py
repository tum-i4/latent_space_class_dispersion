import torch
from typing import List, Protocol, Union, Iterable, Tuple, NamedTuple, Optional
import numpy as np
from pathlib import Path
import argparse
import configparser

class SUT_Proto(Protocol):
    def execute(self, images: Union[torch.Tensor, Iterable[np.array]]) -> List[Tuple[int, float, List[float]]]: ...
    def execute_raw(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, value, traceback): ...

class Mutant(NamedTuple):
    name: str
    training: int

    mutant_folder: Path
    train_set_path: Path

    config: configparser.ConfigParser
    sut: SUT_Proto

    training_time: float

class Data(NamedTuple):
    img: torch.Tensor
    lbl: torch.Tensor
    ids: torch.Tensor

class FuzzingItem(NamedTuple):
    sample: torch.Tensor
    label: int
    original_image_id: int
    ssim: float
    mse: float
    l_0_norm: float
    l_2_norm: float
    l_inf_norm: float
    transformation: str
    combined_id: int

class obj:
    def __init__(self, dict_):
        self.__dict__.update(dict_)
