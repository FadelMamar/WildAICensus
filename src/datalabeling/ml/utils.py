import logging
import os
import traceback
from pathlib import Path

import pandas as pd
import yaml, json

from ultralytics.utils.loss import v8DetectionLoss, E2EDetectLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

import torch


from ..common.config import EvaluationConfig, TrainingConfig
from ..common.evaluation import PerformanceEvaluator, HardSampleSelector
from ..common.io import load_yaml, save_yolo_yaml_cfg
from .models import Detector

logger = logging.getLogger(__name__)

__all__ = [
    "remove_label_cache",
    "sample_pos_neg",
    "get_data_cfg_paths_for_cl",
    "get_data_cfg_paths_for_HN",
]


def remove_label_cache(data_config_yaml: str):
    # Remove labels.cache
    with open(data_config_yaml, "r") as file:
        yolo_config = yaml.load(file, Loader=yaml.FullLoader)
    root = yolo_config["path"]
    for split in ["train", "val", "test"]:
        try:
            for p in yolo_config[split]:
                path = os.path.join(root, p, "../labels.cache")
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Removing: {os.path.join(root, p, '../labels.cache')}")
                # else:
                #     logger.info(path, "does not exist.")
        except Exception:
            # logger.info(e)
            traceback.print_exc()


def sample_pos_neg(images_paths: list, ratio: float, seed: int = 41):
    """_summary_

    Args:
        images_paths (list): images paths
        ratio (float): ratio defined as num_empty/num_non_empty
        seed (int, optional): random seed. Defaults to 41.

    Returns:
        list: selected paths to images
    """

    # build dataframe
    is_empty = [
        1 - Path(str(p).replace("images", "labels")).with_suffix(".txt").exists()
        for p in images_paths
    ]
    data = pd.DataFrame.from_dict(
        {"image_paths": images_paths, "is_empty": is_empty}, orient="columns"
    )
    # get empty and non empty
    num_empty = (data["is_empty"] == 1).sum()
    num_non_empty = len(data) - num_empty
    if num_empty == 0:
        logger.info("contains only positive samples")
    num_sampled_empty = min(int(num_non_empty * ratio), num_empty)
    sampled_empty = data.loc[data["is_empty"] == 1].sample(
        n=num_sampled_empty, random_state=seed
    )
    # concatenate
    sampled_data = pd.concat([sampled_empty, data.loc[data["is_empty"] == 0]])

    logger.info(f"Sampling: pos={num_non_empty} & neg={num_sampled_empty}")

    return sampled_data["image_paths"].to_list()


def get_data_cfg_paths_for_cl(
    ratio: float,
    data_config_yaml: str,
    cl_save_dir: str,
    seed: int = 41,
    split: str = "train",
    pattern_glob: str = "*",
):
    """_summary_

    Args:
        ratio (float): _description_
        data_config_yaml (str): _description_
        cl_save_dir (str): _description_
        seed (int, optional): _description_. Defaults to 41.
        split (str, optional): _description_. Defaults to 'train'.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    yolo_config = load_yaml(data_config_yaml)

    root = yolo_config["path"]
    dirs_images = [os.path.join(root, p) for p in yolo_config[split]]

    # sample positive and negative images
    sampled_imgs_paths = []
    for dir_images in dirs_images:
        logger.info(f"Sampling positive and negative samples from {dir_images}")
        paths = sample_pos_neg(
            images_paths=list(Path(dir_images).glob(pattern_glob)),
            ratio=ratio,
            seed=seed,
        )
        sampled_imgs_paths = sampled_imgs_paths + paths

    # save selected images in txt file
    save_path_samples = os.path.join(
        cl_save_dir, f"{split}_ratio_{ratio}-seed_{seed}.txt"
    )
    pd.Series(sampled_imgs_paths).to_csv(save_path_samples, index=False, header=False)
    logger.info(f"Saving {len(sampled_imgs_paths)} sampled images.")

    # save config
    save_path_cfg = Path(save_path_samples).with_suffix(".yaml")
    cfg = dict(root_dir=root, save_path=save_path_cfg, labels_map=yolo_config["names"])
    if split == "train":
        cfg["yolo_val"] = yolo_config["val"]
        cfg["yolo_train"] = os.path.relpath(save_path_samples, start=root)

    elif split == "val":
        cfg["yolo_val"] = os.path.relpath(save_path_samples, start=root)
        cfg["yolo_train"] = yolo_config["train"]

    else:
        raise NotImplementedError

    # save yolo data cfg
    save_yolo_yaml_cfg(mode="w", **cfg)

    logger.info(
        f"Saving samples at: {save_path_samples} and data_cfg at {save_path_cfg}",
    )

    return str(save_path_cfg)


def get_data_cfg_paths_for_HN(
    args: TrainingConfig,
    data_config_yaml: str,
    eval_config: EvaluationConfig,
    split: str = "train",
    data_config_root: str = "D:\\",
):
    """_summary_

    Args:
        args (Arguments): _description_
        data_config_yaml (str): _description_

    Returns:
        _type_: _description_
    """

    pred_results_dir = args.hn_save_dir
    save_path_samples = os.path.join(args.hn_save_dir, "hard_samples.txt")
    save_path = os.path.join(args.hn_save_dir, "hard_samples.yaml")

    eval_config.uncertainty_threshold = args.hn_uncertainty_thrs
    eval_config.score_threshold = args.hn_score_thrs
    eval_config.score_col = "max_scores"

    # Define detector
    detector = Detector(
        path_to_weights=args.path_weights,
        confidence_threshold=args.hn_confidence_threshold,
        overlap_ratio=args.hn_overlap_ratio,
        tilesize=args.hn_tilesize,
        imgsz=args.hn_imgsz,
        use_sliding_window=args.hn_use_sliding_window,
        device=args.device,
        is_yolo_obb=args.hn_is_yolo_obb,
    )

    perf_eval = PerformanceEvaluator(config=eval_config)
    hard_sampler = HardSampleSelector(config=eval_config)

    # data config yaml
    yolo_config = load_yaml(path=data_config_yaml)

    # get images_paths
    images_paths = os.path.join(yolo_config["path"], yolo_config[split])
    images_paths = pd.read_csv(images_paths, header=None, names=["paths"])[
        "paths"
    ].to_list()

    # compute performance & uncertainty of model
    df_results_per_img = perf_eval.evaluate(
        images_dirs=None,
        images_paths=images_paths,
        pred_results_dir=pred_results_dir,
        detector=detector,
        load_results=args.hn_load_results,
        save_tag="hn-sampling",
    )
    df_hard_negatives = hard_sampler.select_hard_samples(df_results_per_img)

    # save image paths in data_config yaml
    hard_sampler.save_selection_references(
        df_hard_negatives=df_hard_negatives, save_path=save_path_samples
    )

    # save data.yaml file in yolo format
    yolo_val_yaml = [
        os.path.relpath(os.path.join(yolo_config["path"], p), start=data_config_root)
        for p in yolo_config["val"]
    ]

    save_yolo_yaml_cfg(
        root_dir=data_config_root,
        labels_map=yolo_config["names"],
        yolo_train=os.path.relpath(save_path_samples, start=data_config_root),
        yolo_val=yolo_val_yaml,
        save_path=save_path,
    )

    return str(save_path)


RANK = int(os.getenv("RANK", -1))


class CustomLoss(v8DetectionLoss):
    """Custom YOLO loss that inherits from Ultralytics default loss"""

    def __init__(self, pos_weight: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pos_weight = (
            torch.Tensor(
                [
                    pos_weight,
                ]
            )
            .reshape(1, 1)
            .to(self.device)
        )
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        
        logger.info(f'Instantiating BCE loss in custom V8Detection loss iwht pos_weight={pos_weight}')


class CustomDetModel(DetectionModel):
    def __init__(self, pos_weight: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def init_criterion(self):
        return (
            E2EDetectLoss(self)
            if getattr(self, "end2end", False)
            else CustomLoss(self.pos_weight, self)
        )


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights, verbose=True):
        """Returns a customized detection model instance configured with specified config and weights."""

        pos_weight = json.loads(os.environ.get("pos_weight", "1.0"))

        model = CustomDetModel(
            pos_weight, cfg, nc=self.data["nc"], ch=3, verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)

        return model
