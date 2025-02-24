import csv
import json
import pathlib
from typing import Tuple, Optional, Callable, NamedTuple, Literal, List, Any
from pathlib import Path
import torch
from torchvision.datasets import VisionDataset
import PIL
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive

TEST_RESULT_TABLE_CREATION_TEMPLATE = """
CREATE TABLE test_results (
    sut_name VARCHAR,
    sut_training INT,
    dataset TEXT,
    sample_id INT8,
    label INT,
    output INT,
    result BOOLEAN,
    confidence FLOAT,
    latent_space FLOAT[{num_classes}],
    training_time FLOAT,
    evaluation_time FLOAT,
    is_duplicate BOOLEAN,
)
"""

TEST_RESULT_TABLE_CREATION = TEST_RESULT_TABLE_CREATION_TEMPLATE.format(num_classes=43)

def get_test_result_table_creation_sql(num_classes: int) -> str:
    return TEST_RESULT_TABLE_CREATION_TEMPLATE.format(num_classes=num_classes)


class FuzzingDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        model: str,
        method: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._target_folder = Path(root) / "fuzzing" / model / method / "crashes_rec"

        self._images = sorted(self._target_folder.glob("*.png"))
        self._infos = [f.with_suffix(".json") for f in self._images]

    def __len__(self) -> int:
        return len(self._infos)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, int, int]:
        # sample = PIL.Image.open(self._images[index]).convert("RGB")
        sample = PIL.Image.open(self._images[index])

        with self._infos[index].open("r") as fp:
            target = json.load(fp)["__ground_truth_metrics__"][0]["op_class"]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        _, *nums = self._infos[index].stem.split("_")
        nums = [int(n) for n in nums]

        return sample, target, nums[0] << 32 | nums[1]


class FuzzingFairItem(NamedTuple):
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


class FuzzingDatasetFair(VisionDataset):
    def __init__(
        self,
        root: str,
        model: str,
        method: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._target_folder = Path(root) / "fuzzing" / model / method / "crashes_rec"

        self._images = sorted(self._target_folder.glob("*.png"))
        self._infos = [f.with_suffix(".json") for f in self._images]

    def __len__(self) -> int:
        return len(self._infos)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, int, int]:
        # sample = PIL.Image.open(self._images[index]).convert("RGB")
        sample = PIL.Image.open(self._images[index])

        if self.transform is not None:
            sample = self.transform(sample)

        with self._infos[index].open("r") as fp:
            conf = json.load(fp)
        target = conf["__ground_truth_metrics__"][0]["op_class"]
        qm = conf["__image_quality_metrics__"][0]

        if self.target_transform is not None:
            target = self.target_transform(target)

        _, *nums = self._infos[index].stem.split("_")
        nums = [int(n) for n in nums]

        return FuzzingFairItem(
            sample,
            target,
            int(qm["org_image"]),
            qm["SSIM"],
            qm["MSE"],
            qm["l0_norm"],
            qm["l2_norm"],
            qm["linf_norm"],
            qm["transformations"],
            nums[0] << 32 | nums[1],
        )


class GTSRB_Custom(VisionDataset):
    """`German Traffic Sign Recognition Benchmark (GTSRB) <https://benchmark.ini.rub.de/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        split: str = Literal["train", "validation", "test"],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        validation_prefix: Optional[List[str]] = None,
        old_train: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test", "validation"))
        self._base_folder = pathlib.Path(root) / "gtsrb"

        self.old_train = old_train
        folder_name = "Training" if old_train else "Final_Training/Images"

        self._target_folder = (
            self._base_folder
            / "GTSRB"
            / (
                folder_name
                if self._split in {"train", "validation"}
                else "Final_Test/Images"
            )
        )

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        if self._split in {"train", "validation"}:
            samples = make_dataset(str(self._target_folder), extensions=(".ppm",))
            samples.sort(key=lambda x: x[0])

            if validation_prefix:
                if self._split == "train":
                    samples = [
                        (p, l)
                        for p, l in samples
                        if pathlib.Path(p).name.split("_")[0] not in validation_prefix
                    ]
                else:
                    samples = [
                        (p, l)
                        for p, l in samples
                        if pathlib.Path(p).name.split("_")[0] in validation_prefix
                    ]
            else:
                samples = make_dataset(str(self._target_folder), extensions=(".ppm",))
                samples.sort(key=lambda x: x[0])
        else:
            with open(self._base_folder / "GT-final_test.csv") as csv_file:
                samples = [
                    (str(self._target_folder / row["Filename"]), int(row["ClassId"]))
                    for row in csv.DictReader(
                        csv_file, delimiter=";", skipinitialspace=True
                    )
                ]

        self._samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self._split == "test":
            sample_id = int(Path(path).stem)
            return sample, target, sample_id
        else:
            return sample, target

    def _check_exists(self) -> bool:
        return self._target_folder.is_dir()

    def download(self) -> None:
        if self._check_exists():
            return

        base_url = (
            "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
        )

        if self._split in {"train", "validation"}:
            download_and_extract_archive(
                (
                    f"{base_url}GTSRB-Training_fixed.zip"
                    if self.old_train
                    else f"{base_url}GTSRB_Final_Training_Images.zip"
                ),
                download_root=str(self._base_folder),
                # md5="513f3c79a4c5141765e10e952eaa2478",
            )
        else:
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_Images.zip",
                download_root=str(self._base_folder),
                md5="c7e4e6327067d32654124b0fe9e82185",
            )
            download_and_extract_archive(
                f"{base_url}GTSRB_Final_Test_GT.zip",
                download_root=str(self._base_folder),
                md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5",
            )
