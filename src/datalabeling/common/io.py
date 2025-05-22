import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import albumentations as A
import lightning as L
import pandas as pd
import numpy as np
import torch
import yaml
from animaloc.data.transforms import (
    FIDT,
    DownSample,
    MultiTransformsWrapper,
    PointsToMask,
)
from animaloc.datasets import CSVDataset, FolderDataset
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import random
from itertools import chain
from functools import partial
from multiprocessing.pool import ThreadPool

from .config import DataConfig


# =================
# Data Handling
# =================
logger = logging.getLogger(__name__)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)
    return cfg


def get_images_paths(
    images_dir: str,
    patterns: tuple = ("*.JPG", "*.jpg", "*.png", "*.PNG", "*.jpeg", "*.JPEG"),
):
    images_paths = chain.from_iterable([Path(images_dir).glob(p) for p in patterns])
    images_paths = list(set(images_paths))
    return images_paths


def save_yaml(cfg: dict, save_path: str, mode="w"):
    with open(save_path, mode, encoding="utf-8") as file:
        yaml.dump(cfg, file)


def save_json(data: dict | list, save_path: str):
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_yolo_yaml_cfg(
    root_dir: str,
    labels_map: dict,
    yolo_train: list | str,
    yolo_val: list | str,
    save_path: str,
    mode="w",
) -> None:
    cfg_dict = {
        "path": root_dir,
        "names": labels_map,
        "train": yolo_train,
        "val": yolo_val,
        "nc": len(labels_map),
    }

    save_yaml(cfg=cfg_dict, save_path=save_path, mode=mode)


# TODO: create a function to check the label format in yolo images directory
def check_directory_label():
    pass


def load_yolo_label(
    image_path: str | Path, load_empty: bool = True
) -> tuple[dict, str]:
    label_path = str(Path(image_path).with_suffix(".txt")).replace("images", "labels")

    cols_yolo_obb = ["category_id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    cols_yolo = ["category_id", "x", "y", "w", "h"]

    def load_features(path: str):
        with open(path, "r", encoding="utf-8") as file:
            lines = [line.strip().split(" ") for line in file.readlines()]

        num_features = len(lines[0])
        if num_features == len(cols_yolo_obb):
            cols = cols_yolo_obb
            _format = "yolo-obb"
        else:
            cols = cols_yolo
            _format = "yolo"

        features = {
            col: np.array([float(line[i]) for line in lines])
            for i, col in enumerate(cols)
        }

        return features, _format, len(lines)

    # image is negative sample?

    if os.path.exists(label_path):
        df, _format, num_lines = load_features(label_path)

        if _format == "yolo":
            df["x1"] = df["x"] - df["w"] / 2.0
            df["y1"] = df["y"] - df["h"] / 2.0

            df["x2"] = df["x1"] + df["w"]
            df["y2"] = df["y1"]

            df["x3"] = df["x2"]
            df["y3"] = df["y2"] + df["h"]

            df["x4"] = df["x1"]
            df["y4"] = df["y3"]
        else:
            raise ValueError("Supported formats are 'yolo' and 'yolo-obb'.")

    # emtpy images
    elif load_empty:
        _format = "empty"
        num_lines = 1
        df = {
            "category_id": np.nan,
            "x1": np.nan,
            "y1": np.nan,
            "x2": np.nan,
            "y2": np.nan,
            "x3": np.nan,
            "y3": np.nan,
            "x4": np.nan,
            "y4": np.nan,
        }
    else:
        return None, "empty"

    # add features
    with Image.open(image_path) as img:
        width, height = img.size
        df["width"] = [width] * num_lines
        df["height"] = [height] * num_lines
        df["file_name"] = [str(image_path)] * num_lines

    # unnormalize values
    for i in range(1, 5):
        df[f"x{i}"] = df[f"x{i}"] * df["width"][0]
        df[f"y{i}"] = df[f"y{i}"] * df["height"][0]

    df["x_min"] = df["x1"]
    df["y_min"] = df["y1"]
    df["x_max"] = df["x2"]
    df["y_max"] = df["y3"]

    # convert values to list
    for col in df.keys():
        if isinstance(df[col], float):
            df[col] = [df[col]]
        elif isinstance(df[col], np.ndarray):
            df[col] = df[col].tolist()

    return df, _format


class DataHandler:
    def __init__(self, config: DataConfig):
        self.config = config

    @staticmethod
    def load_yolo_groundtruth(
        images_dir: str = None,
        images_paths: list[str] | Sequence = None,
        load_empty: bool = True,
        max_workers: int = 1,
    ) -> tuple[pd.DataFrame, str]:
        results = dict()
        labels_format = set()
        num_empty = 0

        if images_dir:
            assert images_paths is None, "It should not be provided."
            images_paths = list(Path(images_dir).glob("*"))

        func = partial(load_yolo_label, load_empty=load_empty)

        counter = 1
        with ThreadPool(max_workers) as executor:
            for df, _format in executor.map(func, images_paths):
                if _format == "empty":
                    num_empty += 1
                else:
                    labels_format.add(_format)
                assert len(labels_format) <= 1, (
                    f"Only one label format is supported. Labels format are {labels_format}"
                )
                if df is not None:
                    if counter == 1:
                        results.update(df)
                    else:
                        for col in results.keys():
                            results[col] = results[col] + df[col]
                    counter += 1

        if len(labels_format) == 0:
            labels_format.add("empty")

        return pd.DataFrame.from_dict(results), labels_format.pop()

    @staticmethod
    def load_json_predictions(path_result: str) -> pd.DataFrame:
        return pd.read_json(path_result, orient="records")

    @staticmethod
    def load_data_herdnet_from_dir(
        yolo_images_dir: str,
    ) -> Tuple[pd.DataFrame, List[str]]:
        if not Path(yolo_images_dir).exists():
            raise FileNotFoundError(f"Directory {yolo_images_dir} not found")

        records, empties = [], []
        images_paths = list(Path(yolo_images_dir).glob("*"))
        for image_path in tqdm(images_paths, desc=f"loading data at {yolo_images_dir}"):
            df, _ = load_yolo_label(image_path=image_path, load_empty=False)
            if df is None:
                empties.append(str(image_path))
                continue
            records.append(df)

        df_results = pd.DataFrame.from_records(
            chain.from_iterable(records)
        ).convert_dtypes()
        df_results = df_results.rename(
            columns={"file_name": "images", "category_id": "labels"}
        )
        df_results["labels"] += 1  # shift to reserve 0 for background

        return df_results, empties

    @staticmethod
    def load_data_herdnet_from_yaml(
        yaml_path: str,
        split: str,
        transforms: Dict[str, Tuple[List[Any], List[Any]]],
        empty_ratio: Optional[float] = None,
        empty_frac: Optional[float] = None,
    ) -> Tuple[ConcatDataset, pd.DataFrame, int]:
        cfg = load_yaml(yaml_path)

        assert split in cfg, f"Unknown split {split}"

        datasets, dfs, total_empty = [], [], 0

        for subset in cfg[split]:
            img_dir = os.path.join(cfg["path"], subset)
            df, empties = DataHandler.load_data_herdnet_from_dir(img_dir)
            # sample empties
            sampled = []

            if empty_ratio:
                n = min(int(empty_ratio * len(df)), len(empties))
                sampled = pd.Series(empties).sample(n).tolist()

            elif empty_frac:
                sampled = pd.Series(empties).sample(frac=empty_frac).tolist()

            paths = df["images"].unique().tolist() + list(set(sampled))

            ds = FolderDataset(
                csv_file=df,
                root_dir="",
                albu_transforms=transforms[split][0],
                end_transforms=transforms[split][1],
                images_paths=paths,
            )

            datasets.append(ds)
            dfs.append(df)
            total_empty += len(sampled)

        return ConcatDataset(datasets), pd.concat(dfs, ignore_index=True), total_empty

    @staticmethod
    def load_data_herdnet_for_prediction(
        images_path: str,
        albu_transforms=None,
        end_transforms=None,
    ) -> CSVDataset:
        images_path = list(map(str, images_path))

        # create dummy df_labels
        num_images = len(images_path)
        df_labels = {
            "x": [0.0] * num_images,
            "y": [0.0] * num_images,
            "labels": [0] * num_images,
            "images": images_path,
        }
        df_labels = pd.DataFrame.from_dict(df_labels)

        return CSVDataset(
            csv_file=df_labels,
            root_dir="",
            albu_transforms=albu_transforms,
            end_transforms=end_transforms,
        )


class ClassifierFeaturesData(Dataset):
    def __init__(
        self,
        split_data_dir: str,
        transform=None,
        tn_ratio: float = 1.0,
        tn_label: str = "true_negatives",
    ):
        super().__init__()

        self.data_dir = Path(split_data_dir)

        self.tn_label = tn_label

        self.tn_ratio = tn_ratio

        self.get_data()

        # self.transform=transform # not used

    def get_data(
        self,
    ):
        class_names = sorted(os.listdir(self.data_dir))
        self.classes = list(range(len(class_names)))
        self.class_to_idx = dict(zip(class_names, self.classes))

        samples = []

        # get true positive
        num_positive = 0
        for c in class_names:
            if c == self.tn_label:
                continue
            tp_data = list((self.data_dir / c).glob("*"))
            num_positive += len(tp_data)
            samples.append(list(tp_data))
        logger.info(f"Sampling {num_positive} True-Positives from {self.data_dir}")

        # sample true negatives
        if self.tn_label in class_names:
            assert self.tn_ratio > 0.0
            tn_data = list((self.data_dir / self.tn_label).glob("*"))
            num_tn = int(num_positive * self.tn_ratio)
            num_tn = min(num_tn, len(tn_data))
            random.seed(41)
            random.shuffle(tn_data)  # shuffle
            samples.append(tn_data[:num_tn])

            logger.info(f"Sampling {num_tn} True-Negatives from {self.data_dir}")

        self.samples = list(chain.from_iterable(samples))

    def __len__(
        self,
    ):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        features = np.load(path)
        label = self.class_to_idx[path.parent.name]

        return torch.Tensor(features), torch.Tensor(
            [
                label,
            ]
        ).long()


class ClassifierDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 1,
        img_size: int = 96,
        is_features: bool = False,
        tn_ratio: float = 1.0,
        # train_tfms=None,
        # val_tfms=None,
    ):
        super().__init__()

        self.train_dir = Path(data_dir) / "train"
        self.val_dir = Path(data_dir) / "val"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.tn_ratio = tn_ratio

        self.loader = ImageFolder
        if is_features:
            self.loader = lambda x, transform: ClassifierFeaturesData(
                x, transform=transform, tn_ratio=self.tn_ratio
            )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.train_tfms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.AutoAugment(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        self.val_tfms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        # self.train_tfms = A.Compose(
        #     A.PadIfNeeded(min_height=self.img_size*2,
        #                   min_width=self.img_size*2
        #                   ),
        #     A.Cr

        #     )
        # self.val_tfms = ...

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = self.loader(self.train_dir, transform=self.train_tfms)
            self.val_dataset = self.loader(self.val_dir, transform=self.val_tfms)
            self.num_classes = len(self.train_dataset.classes)

        if stage == "validate":
            self.val_dataset = self.loader(self.val_dir, transform=self.val_tfms)
            self.num_classes = len(self.train_dataset.classes)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=self.num_workers,
            # persistent_workers=True,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=self.num_workers,
            # persistent_workers=True,
            pin_memory=torch.cuda.is_available(),
        )


class HerdnetData(L.LightningDataModule):
    """Lightning datamodule. This class handles all the data preparation tasks. It facilitates reproducibility."""

    def __init__(
        self,
        data_config_yaml: str,
        patch_size: int,
        down_ratio: int = 2,
        tr_batch_size: int = 32,
        val_batch_size: int = 1,
        transforms: dict[str, tuple] = None,
        train_empty_ratio: float = 0.0,
        val_empty_frac: float = 1.0,
        normalization: str = "standard",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.batch_size = tr_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.down_ratio = down_ratio
        self.data_config_yaml = data_config_yaml
        self.transforms = transforms
        self.train_empty_ratio = train_empty_ratio
        self.val_empty_frac = val_empty_frac
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.predict_dataset = None
        self.predict_batchsize = 8
        self.df_train_labels_freq = None
        self.df_val_labels_freq = None
        self.num_empty_images_val = None
        self.num_empty_images_train = None
        self.num_empty_images_test = None
        self.mean = mean
        self.std = std
        self.normalization = normalization

        self.num_workers = 8
        self.pin_memory = torch.cuda.is_available()

        # Get number of classes
        with open(data_config_yaml, "r") as file:
            data_config = yaml.load(file, Loader=yaml.FullLoader)
            # accounting for background class
            self.num_classes = data_config["nc"] + 1

        if self.transforms is None:
            self._set_transforms()

    def _set_transforms(
        self,
    ):
        self.transforms = {}
        self.transforms["train"] = (
            [
                A.Resize(width=self.patch_size, height=self.patch_size, p=1.0),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.2
                ),
                A.Blur(blur_limit=15, p=0.2),
                A.Normalize(
                    normalization=self.normalization,
                    p=1.0,
                    mean=self.mean,
                    std=self.std,
                ),
            ],
            [
                MultiTransformsWrapper(
                    [
                        FIDT(num_classes=self.num_classes, down_ratio=self.down_ratio),
                        PointsToMask(
                            radius=2,
                            num_classes=self.num_classes,
                            squeeze=True,
                            down_ratio=int(
                                self.patch_size // (16 * self.patch_size / 512)
                            ),
                        ),
                    ]
                )
            ],
        )
        self.transforms["val"] = (
            [
                A.Resize(width=self.patch_size, height=self.patch_size, p=1.0),
                A.Normalize(
                    normalization=self.normalization,
                    p=1.0,
                    mean=self.mean,
                    std=self.std,
                ),
            ],
            [
                DownSample(down_ratio=self.down_ratio, anno_type="point"),
            ],
        )
        self.transforms["test"] = self.transforms["val"]

    @property
    def get_labels_weights(
        self,
    ) -> torch.Tensor:
        """Computes importance weights for cross entropy loss

        Returns:
            torch.Tensor: weights for cross entropy loss
        """
        weights = 1 / (self.df_train_labels_freq + 1e-6)
        weights = [1.0] + weights.to_list()
        assert len(weights) == self.num_classes, "Check for inconsistencies."
        return torch.Tensor(weights)

    def setup(self, stage: str):
        if stage == "fit":
            # train
            self.train_dataset, df_train_labels, self.num_empty_images_train = (
                DataHandler.load_data_herdnet_from_yaml(
                    yaml_path=self.data_config_yaml,
                    split="train",
                    transforms=self.transforms,
                    empty_ratio=self.train_empty_ratio,
                    empty_frac=None,
                )
            )
            self.df_train_labels_freq = df_train_labels[
                "labels"
            ].value_counts().sort_index() / (
                len(df_train_labels) + self.num_empty_images_train
            )
            logger.info(
                f"Train dataset as {len(self.train_dataset)} samples"
                f" including {self.num_empty_images_train} negative samples."
            )
            # val
            self.val_dataset, df_val_labels, self.num_empty_images_val = (
                DataHandler.load_data_herdnet_from_yaml(
                    yaml_path=self.data_config_yaml,
                    split="val",
                    transforms=self.transforms,
                    empty_ratio=None,
                    empty_frac=self.val_empty_frac,
                )
            )
            self.df_val_labels_freq = df_val_labels[
                "labels"
            ].value_counts().sort_index() / (
                len(df_val_labels) + self.num_empty_images_val
            )
            logger.info(
                f"Val dataset as {len(self.val_dataset)} samples"
                f" including {self.num_empty_images_val} negative samples."
            )

        elif stage == "test":
            self.test_dataset, _, self.num_empty_images_test = (
                DataHandler.load_data_herdnet_from_yaml(
                    yaml_path=self.data_config_yaml,
                    split="test",
                    transforms=self.transforms,
                    empty_frac=self.val_empty_frac,
                    empty_ratio=None,
                )
            )
            logger.info(
                f"Test dataset as {len(self.test_dataset)} samples"
                f" including {self.num_empty_images_test} negative samples."
            )

        elif stage == "validate":
            # val
            self.val_dataset, df_val_labels, self.num_empty_images_val = (
                DataHandler.load_data_herdnet_from_yaml(
                    yaml_path=self.data_config_yaml,
                    split="val",
                    transforms=self.transforms,
                    empty_ratio=None,
                    empty_frac=self.val_empty_frac,
                )
            )
            self.df_val_labels_freq = df_val_labels[
                "labels"
            ].value_counts().sort_index() / (
                len(df_val_labels) + self.num_empty_images_val
            )
            logger.info(
                f"Val dataset as {len(self.val_dataset)} samples"
                f" including {self.num_empty_images_val} negative samples."
            )

    def val_collate_fn(
        self, batch: tuple[torch.Tensor, dict]
    ) -> tuple[torch.Tensor, dict]:
        """collate_fn used to create the validation dataloader

        Args:
            batch (tuple): (img:torch.Tensor, targets:dict)

        Returns:
            tuple: (image, target)
        """

        batched = dict(points=[], labels=[])
        batch_img = torch.stack([p[0] for p in batch])
        targets = [p[1] for p in batch]
        keys = targets[0].keys()

        # get non_empty samples indidces -> set difference
        non_empty_idx = [i for i, a in enumerate(targets) if len(a["labels"]) > 0]
        targets_empty = [
            targets[i] for i in list(set(range(len(batch))) - set(non_empty_idx))
        ]
        targets = [targets[i] for i in non_empty_idx]

        # Creating batch
        for k in keys:
            batched[k] = []  # initialize to be empty list
            if k == "points" or k == "labels":
                batched[k] = [a[k].cpu().tolist() for a in targets]
                if len(targets_empty) > 0:
                    batched[k] = batched[k] + [[]] * len(targets_empty)

        return batch_img, batched

    def set_predict_dataset(self, images_path: list[str], batchsize: int = 16) -> None:
        self.predict_dataset = DataHandler.load_data_herdnet_for_prediction(
            images_path=images_path,
            albu_transforms=self.transforms["val"][0],
            end_transforms=self.transforms["val"][1],
        )
        self.predict_batchsize = batchsize

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # persistent_workers=True,
        )

    def val_dataloader(self):
        """Creates validation dataloader."""

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            sampler=torch.utils.data.SequentialSampler(self.val_dataset),
            # num_workers=self.num_workers,
            collate_fn=self.val_collate_fn,
            # persistent_workers=True,
        )

    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            sampler=torch.utils.data.SequentialSampler(self.test_dataset),
            shuffle=False,
            # num_workers=self.num_workers,
            collate_fn=self.val_collate_fn,
            # persistent_workers=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self.predict_batchsize, shuffle=False
        )
