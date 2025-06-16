import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import yaml, json
from itertools import product
from random import shuffle

from ultralytics.utils.loss import v8DetectionLoss, E2EDetectLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.ops import xywh2xyxy
from ultralytics import YOLO
from ultralytics.data.utils import polygon2mask

from transformers import AutoModel
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import torchvision.models
from torchmetrics.functional.detection import complete_intersection_over_union

from ..common.config import EvaluationConfig, TrainingConfig
from ..common.evaluation import PerformanceEvaluator, HardSampleSelector
from ..common.io import load_yaml, save_yolo_yaml_cfg


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
        # try:
        if split in yolo_config.keys():
            for p in yolo_config[split]:
                path = os.path.join(root, p, "../labels.cache")
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Removing: {os.path.join(root, p, '../labels.cache')}")
        else:
            logger.info(f"split={split} does not exist.")
            # else:
            #     logger.info(path, "does not exist.")
        # except Exception:
        #     # logger.info(e)
        #     traceback.print_exc()


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


# TODO test, debug
def get_data_cfg_paths_for_HN(
    args: TrainingConfig,
    data_config_yaml: str,
    model: YOLO,
    split: str = "train",
):
    """_summary_

    Args:
        args (Arguments): _description_
        data_config_yaml (str): _description_

    Returns:
        _type_: _description_
    """

    from .interface import InferenceEngine, PredictionConfig
    from ..common.dataset_loader import LabelingDataset

    save_path_samples = os.path.join(args.hn_save_dir, "hard_samples.txt")
    save_path = os.path.join(args.hn_save_dir, "hard_samples.yaml")

    eval_config = EvaluationConfig()
    eval_config.uncertainty_threshold = args.hn_uncertainty_thrs
    eval_config.score_threshold = args.hn_score_thrs

    # Define detector
    config = PredictionConfig(
        imgsz=args.hn_imgsz,
        tilesize=args.hn_tilesize,
        batch_size=args.hn_batch_size,
        overlap_ratio=args.hn_overlap_ratio,
        confidence_threshold=args.hn_confidence_threshold,
        inference_service_url=None,
        flight_height=180,
        sensor_height=24,
        gsd=None,
        nms_iou=0.5,
        verbose=True,
        # min_area=100,
        # max_area=None,
        cls_imgsz=98,
        device=args.device,
    )

    model.eval()
    engine, _ = InferenceEngine.load_engine(
        pred_config=config,
        roi_classifier_path=None,
        detection_label_map=model.names,
        detection_model=model,
        mlflow_model_alias=None,
        mlflow_model_name="labeler",
    )

    perf_eval = PerformanceEvaluator()
    hard_sampler = HardSampleSelector(config=eval_config)

    # data config yaml
    yolo_config = load_yaml(data_config_yaml)

    # build dataset and add predictions
    images_dirs = [os.path.join(yolo_config["path"], p) for p in yolo_config[split]]
    dataset = LabelingDataset.from_dirs(images_dirs)
    dataset.add_predictions(engine=engine, build=True)

    # compute performance & uncertainty of model
    df_results_per_img = perf_eval.run(
        dataset=dataset,
        pred_results_dir=args.hn_save_dir,
        load_results=args.hn_load_results,
        tp_iou_threshold=eval_config.tp_iou_threshold,
        save_tag="hn-sampling",
    )
    df_hard_negatives = hard_sampler.run(df_results_per_img)

    # save image paths in data_config yaml
    hard_sampler.save_selection_references(
        df_hard_negatives=df_hard_negatives, save_path=save_path_samples
    )

    data_config_root = list(Path(yolo_config["path"]).parents)[
        -1
    ]  # D:\ or C:\ or /home
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

    model.train()

    return str(save_path)


RANK = int(os.getenv("RANK", -1))


class CustomLoss(v8DetectionLoss):
    """Custom YOLO loss that inherits from Ultralytics default loss"""

    def __init__(
        self,
        model,
        pos_weight: float = 1.0,
        fp_tp_loss_weight: float = 0.0,
        count_loss_weight: float = 0.0,
        area_loss_weight: float = 0.0,
        mask_loss_weight: float = 0.0,
    ):
        super().__init__(model=model)

        self.fp_tp_loss_weight = fp_tp_loss_weight
        self.count_loss_weight = count_loss_weight
        self.area_loss_weight = area_loss_weight
        self.mask_loss_weight = mask_loss_weight

        self.model = model
        self.count_loss = nn.SmoothL1Loss(reduction="sum")
        self.area_loss = nn.SmoothL1Loss(reduction="sum")

        assert isinstance(pos_weight, float) or pos_weight is None
        if pos_weight:
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

        logger.debug(
            f"Instantiating BCE loss in custom V8Detection loss iwht pos_weight={pos_weight}"
        )

    def compute_count_area_loss(self, target_bboxes: torch.Tensor, scale_tensor):
        self.model._forward_aux()  # collect area and count logits

        loss = torch.zeros(2, device=self.device)
        pred_count = self.model.pred_aux.get("pred_count", None)
        pred_area = self.model.pred_aux.get("pred_area", None)

        target_bboxes[..., 1:5] = target_bboxes[..., 1:5].div_(scale_tensor)

        if pred_count is not None:
            target_count = torch.zeros_like(pred_count)
            if target_bboxes.shape[1] != 0:
                target_count = (
                    (target_bboxes[..., 0] > 0).sum(dim=1).unsqueeze(1)
                )  # count number of bboxes with positive class_id i.e. label
            loss[0] = self.count_loss(pred_count, target_count) * self.count_loss_weight

        if pred_area is not None:
            target_area = torch.zeros_like(pred_area)
            if target_bboxes.shape[1] != 0:
                w = target_bboxes[..., 3] - target_bboxes[..., 1]  # x2-x1
                h = target_bboxes[..., 4] - target_bboxes[..., 2]  # y2-y1
                target_area = (w * h).sum(dim=1).unsqueeze(1)
            loss[1] = self.area_loss(pred_area, target_area) * self.area_loss_weight

        return loss.sum()

    def compute_mask_loss(
        self, gt_boxes: torch.Tensor, fg_mask: torch.Tensor, batch_images: torch.Tensor
    ):
        p3_layer_idx = self.model.mask_p3_layer_indx
        x = dict(
            p3=self.model.activations[p3_layer_idx],
            img=batch_images,
            gt_boxes=gt_boxes,
            fg_mask=fg_mask,
        )

        return self.model.mask_head(**x)

    def _generate_synthetic_boxes(
        self, img_height: int, img_width: int, area_thresh: float, num: int = 10
    ) -> torch.Tensor:
        w = int(np.sqrt(area_thresh))
        h = w

        xs = torch.randint(
            low=0,
            high=img_width - w,
            size=(num,),
        )

        ys = torch.randint(
            low=0,
            high=img_height - h,
            size=(num,),
        )

        boxes = []
        pairs = list(product(xs, ys))
        shuffle(pairs)  # shuffle
        for x, y in pairs[:num]:
            box = torch.Tensor([x, y, x + w, y + h])
            boxes.append(box)

        return torch.vstack(boxes).to(self.device)

    def _sample_pred(
        self,
        bboxes: torch.Tensor,
        scores: torch.Tensor,
        max_num: int,
        area_thresh: Optional[float] = None,
        scores_range: Optional[Tuple[float, float]] = (0.35, 0.6),
    ) -> torch.Tensor:
        """
        Sample prediction indices based on bounding box area and score criteria.

        Args:
            bboxes: Bounding boxes [n_preds, 4] in xyxy format
            scores: Prediction scores [n_preds]
            max_num: Maximum number of indices to sample
            area_thresh: Minimum area threshold for valid boxes
            scores_range: Score range (min_score, max_score) for filtering

        Returns:
            Selected indices [n_selected] where n_selected <= max_num
        """
        n_preds = bboxes.shape[0]
        device = bboxes.device

        if n_preds == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        # Compute bbox areas
        widths = torch.clamp(bboxes[:, 2] - bboxes[:, 0], min=0)  # Ensure non-negative
        heights = torch.clamp(bboxes[:, 3] - bboxes[:, 1], min=0)
        areas = widths * heights

        indices = torch.arange(n_preds, device=device)

        # Create selection masks
        area_mask = None
        if area_thresh is not None and area_thresh > 0:
            area_mask = areas >= area_thresh
            if not area_mask.any():
                area_mask = None

        score_mask = None
        if scores_range is not None and len(scores_range) == 2:
            min_score, max_score = scores_range
            score_mask = (scores >= min_score) & (scores <= max_score)
            if not score_mask.any():
                score_mask = None

        # Combine masks
        if area_mask is not None and score_mask is not None:
            valid_mask = area_mask & score_mask
        elif area_mask is not None:
            valid_mask = area_mask
        elif score_mask is not None:
            valid_mask = score_mask
        else:
            valid_mask = None

        # Fallback to non-degenerate boxes if no valid mask
        if valid_mask is None or not valid_mask.any():
            # Select boxes that are not degenerate (have positive area)
            valid_mask = areas > 0
            if not valid_mask.any():
                # If all boxes are degenerate, return empty tensor
                return torch.empty(0, dtype=torch.long, device=device)

        valid_indices = indices[valid_mask]

        # Randomly sample up to max_num indices
        n_valid = valid_indices.shape[0]
        if n_valid == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        sample_size = min(max_num, n_valid)
        if sample_size == n_valid:
            return valid_indices

        # Random sampling without replacement
        random_idx = torch.randperm(n_valid, device=device)[:sample_size]
        return valid_indices[random_idx]

    def _sample_pred_by_score(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        target_labels: torch.Tensor,
        img_height: int,
        img_width: int,
        fg_mask: torch.Tensor,
        max_num_tn: int = 5,
        area_thresh: float = 400,
        scores_range: Tuple[float, float] = (0.4, 0.6),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample bounding boxes based on a combination of bbox area and prediction scores.

        Args:
            pred_bboxes: Predicted bounding boxes [batch_size, n_predictions, 4] in xyxy format
            pred_scores: Class prediction scores [batch_size, n_predictions, n_classes]
            gt_bboxes: Ground truth bounding boxes [batch_size, n_gt, 4]
            target_labels: Ground truth labels [batch_size, n_gt, n_classes]
            img_height: Image height in pixels
            img_width: Image width in pixels
            fg_mask: Foreground mask [batch_size, n_predictions] indicating valid detections
            max_num_tn: Maximum number of boxes to sample per negative image
            area_thresh: Minimum area threshold for valid bounding boxes
            scores_range: Score range for sampling (min_score, max_score)

        Returns:
            Tuple of (selected_pred_bboxes, selected_gt_bboxes, selected_labels, image_indices)
        """
        batch_size = pred_scores.shape[0]

        # Initialize output lists
        selected_pred_bboxes: List[torch.Tensor] = []
        image_indices: List[torch.Tensor] = []
        selected_gt_bboxes: List[torch.Tensor] = []
        selected_labels: List[torch.Tensor] = []

        for batch_idx in range(batch_size):
            # Extract foreground predictions for current image
            fg_indices = fg_mask[batch_idx]

            if not fg_indices.any():
                # No foreground detections - generate synthetic boxes
                synthetic_boxes = self._generate_synthetic_boxes(
                    img_height, img_width, area_thresh, max_num_tn
                )
                selected_pred_bboxes.append(synthetic_boxes)
                image_indices.append(
                    torch.full(
                        (synthetic_boxes.shape[0],), batch_idx, device=self.device
                    )
                )

                # Add empty tensors for GT data
                selected_gt_bboxes.append(torch.zeros_like(synthetic_boxes))
                selected_labels.append(
                    torch.full((synthetic_boxes.shape[0],), -1, device=self.device)
                )
                continue

            # Get foreground predictions
            fg_bboxes = pred_bboxes[batch_idx, fg_indices]  # [n_fg, 4]
            fg_scores = pred_scores[batch_idx, fg_indices]  # [n_fg, n_classes]
            fg_gt_bboxes = gt_bboxes[batch_idx, fg_indices]
            fg_gt_labels = target_labels[batch_idx, fg_indices]

            # Get max scores across classes
            max_scores, _ = fg_scores.max(dim=1)  # [n_fg]

            # Sample valid indices
            valid_indices = self._sample_pred(
                bboxes=fg_bboxes.detach(),
                scores=max_scores.detach(),
                area_thresh=area_thresh,
                scores_range=scores_range,
                max_num=fg_gt_bboxes.shape[0],
            )

            if valid_indices.numel() > 0:
                # Select valid predictions
                selected_pred_bboxes.append(fg_bboxes[valid_indices])
                image_indices.append(
                    torch.full((valid_indices.shape[0],), batch_idx, device=self.device)
                )
                # Select corresponding GT data if available
                selected_gt_bboxes.append(fg_gt_bboxes[valid_indices])
                selected_labels.append(fg_gt_labels[valid_indices])

            else:
                # No valid prediction samples found - generate synthetic boxes
                synthetic_boxes = self._generate_synthetic_boxes(
                    img_height, img_width, area_thresh, fg_gt_bboxes.shape[0]
                )
                selected_pred_bboxes.append(synthetic_boxes)
                image_indices.append(
                    torch.full(
                        (synthetic_boxes.shape[0],), batch_idx, device=self.device
                    )
                )
                selected_gt_bboxes.append(fg_gt_bboxes)
                selected_labels.append(fg_gt_labels)

        # Concatenate all results
        final_pred_bboxes = (
            torch.cat(selected_pred_bboxes, dim=0)
            if selected_pred_bboxes
            else torch.empty(0, 4, device=self.device)
        )
        final_image_indices = (
            torch.cat(image_indices, dim=0)
            if image_indices
            else torch.empty(0, dtype=torch.long, device=self.device)
        )
        final_gt_bboxes = (
            torch.cat(selected_gt_bboxes, dim=0)
            if selected_gt_bboxes and selected_gt_bboxes[0].numel() > 0
            else torch.empty(0, 4, device=self.device)
        )
        final_labels = (
            torch.cat(selected_labels, dim=0)
            if selected_labels and selected_labels[0].numel() > 0
            else torch.empty(0, device=self.device)
        )

        return final_pred_bboxes, final_gt_bboxes, final_labels, final_image_indices

    def compute_loss_from_fptp(
        self,
        target_bboxes: torch.Tensor,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch_images: torch.Tensor,
        target_labels: torch.Tensor,
        # target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        image_idx: torch.Tensor,
        max_num_tn: int = 5,
        box_area_thresh: int = 100,
        scores_range: Tuple = (0.1, 0.9),
    ) -> Tuple[torch.Tensor]:
        # batch has only negative samples

        fg_mask = fg_mask > 0.0

        # max_num = int(fg_mask.sum()) if fg_mask.any() else num_max_boxes
        # max_num = min(max_num,num_max_boxes)
        # max_num = 5

        _, _, img_height, img_width = batch_images.shape
        pred_bboxes, gt_bboxes, target_labels, image_idx = self._sample_pred_by_score(
            pred_bboxes=pred_bboxes.detach(),
            gt_bboxes=target_bboxes,
            target_labels=target_labels,
            pred_scores=pred_scores.detach(),
            img_height=img_height,
            img_width=img_width,
            max_num_tn=max_num_tn,
            area_thresh=box_area_thresh,
            scores_range=scores_range,
            fg_mask=fg_mask,
        )

        p3_layer_idx = self.model.roi_classifier_layers["p3"]
        p4_layer_idx = self.model.roi_classifier_layers["p4"]
        x = dict(
            p3=self.model.activations[p3_layer_idx][image_idx],
            p4=self.model.activations[p4_layer_idx][image_idx],
            img=batch_images[image_idx],
        )

        x.update(
            dict(
                gt_bboxes=gt_bboxes,
                pred_bboxes=pred_bboxes,
                target_labels=target_labels,
            )
        )

        return self.model.roi_classifier(x)

    def __call__(
        self, preds: Any, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""

        if (
            self.count_loss_weight > 0.0
            or self.area_loss_weight > 0.0
            or self.fp_tp_loss_weight > 0.0
            or self.mask_loss_weight > 0.0
        ):
            loss = torch.zeros(4, device=self.device)  # box, cls, dfl, auxilary
        else:
            loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = (
            self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
        )

        target_scores_sum = max(target_scores.sum(), 1)

        ## compute auxilary losses
        if (
            self.count_loss_weight > 0.0 or self.area_loss_weight > 0.0
        ) and self.model.training:
            area_count_loss = self.compute_count_area_loss(
                targets, scale_tensor=imgsz[[1, 0, 1, 0]]
            )
            loss[3] += area_count_loss / target_scores_sum

        if (self.fp_tp_loss_weight > 0.0) and self.model.training:
            if fg_mask.sum() < 1.0:  # batch has only negative samples
                bbox_idx = target_gt_idx[fg_mask > 0.0].cpu()  # tensor([])
                image_idx = bbox_idx.clone()  # tensor([])
            else:
                bbox_idx = target_gt_idx[fg_mask].cpu()  # valid bbox indices
                image_idx = (
                    batch["batch_idx"][bbox_idx].long().cpu()
                )  # mapping img -> bbox

            fp_tp_loss = self.compute_loss_from_fptp(
                target_bboxes=target_bboxes / stride_tensor,
                pred_bboxes=pred_bboxes,  # .detach(),  # disable detach to allow gradient flowing through detection head as well
                pred_scores=pred_scores.detach(),
                batch_images=batch["img"],
                target_labels=target_labels,
                # target_scores=target_scores,
                image_idx=image_idx,
                fg_mask=fg_mask,
                max_num_tn=5,
                box_area_thresh=2500,
                scores_range=(0.1, 0.9),
            )
            loss[3] = loss[3] + fp_tp_loss * self.fp_tp_loss_weight / target_scores_sum

        if self.mask_loss_weight > 0.0 and self.model.training:
            mask_loss = self.compute_mask_loss(
                target_bboxes / stride_tensor,
                fg_mask=fg_mask,
                batch_images=batch["img"],
            )
            loss[3] = loss[3] + mask_loss * self.mask_loss_weight / target_scores_sum

        # Cls loss -> Objectness
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class CustomDetModel(DetectionModel):
    def __init__(self, *args, pos_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def init_criterion(self):
        return (
            E2EDetectLoss(self)
            if getattr(self, "end2end", False)
            else CustomLoss(
                model=self,
                pos_weight=self.pos_weight,
                fp_tp_loss_weight=0.0,
                count_loss_weight=0.0,
                area_loss_weight=0.0,
            )
        )


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights, verbose=True):
        """Returns a customized detection model instance configured with specified config and weights."""

        pos_weight = json.loads(os.environ.get("pos_weight", "1.0"))

        model = CustomDetModel(
            cfg,
            nc=self.data["nc"],
            ch=3,
            pos_weight=pos_weight,
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model


class RegressorHead(torch.nn.Module):
    def __init__(self, out_channels: int = 64):
        super().__init__()

        self.reducer = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.AdaptiveAvgPool2d((1, 1)),  # gives (B,out_channels,1,1)
        )

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.reducer(x)
        x = x.flatten(1)
        return self.mlp(x)


class MaskHead(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.upsampler = nn.Sequential(
            # nn.LazyConv2d(64, kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # *2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # *4
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # *8
            nn.BatchNorm2d(1),
        )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _bbox_to_mask(self, bboxes: torch.Tensor, imgsz: tuple = (800, 800)):
        """
        Convert multiple bounding boxes to polygons in batch.

        Args:
            bboxes: Tensor/array of shape (N, 4) with N bounding boxes
            format: Output format - 'numpy' or 'torch'

        Returns:
            mask
        """
        if isinstance(bboxes, list):
            bboxes = torch.Tensor(bboxes)
        elif isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).float()
        else:
            bboxes = bboxes.cpu()

        # Extract coordinates
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        polygons = (
            torch.stack(
                [
                    torch.stack([x1, y1], dim=1),  # top-left
                    torch.stack([x2, y1], dim=1),  # top-right
                    torch.stack([x2, y2], dim=1),  # bottom-right
                    torch.stack([x1, y2], dim=1),  # bottom-left
                ],
                dim=1,
            )
            .cpu()
            .tolist()
        )
        polygons = [np.array(p) for p in polygons]

        mask = polygon2mask(
            imgsz,  # tuple
            polygons,  # input as list
            color=1,  # 8-bit binary
            downsample_ratio=1,
        )
        mask = torch.from_numpy(mask).int()

        return mask

    def forward(
        self,
        p3: torch.Tensor,
        gt_boxes: torch.Tensor,
        fg_mask: torch.Tensor,
        img: torch.Tensor,
    ):
        if p3 is None:
            raise ValueError("p3 is not available")
        else:
            assert all([p3.shape[i] == (img.shape[i] / 8) for i in [2, 3]]), (
                f"p3:{p3.shape[2:]} != img:{img.shape[2:]}/8"
            )

        self.device = p3.device
        self.upsampler = self.upsampler.to(self.device)

        B, C, H, W = img.shape

        if fg_mask.sum() < 1:
            target_mask = torch.zeros(B, 1, H, W).to(self.device)

        else:
            target_mask = []
            for batch_idx in range(img.shape[0]):
                # Extract foreground predictions for current image
                fg_indices = fg_mask[batch_idx]
                if not fg_indices.any():
                    # No foreground detections in img
                    target_mask.append(torch.zeros(1, H, W).to(self.device))
                    continue
                # Get foreground masks
                fg_gt_bboxes = gt_boxes[batch_idx, fg_indices]
                mask = self._bbox_to_mask(fg_gt_bboxes, imgsz=img.shape[2:])
                mask = mask.unsqueeze(0).to(self.device)
                target_mask.append(mask)
            target_mask = torch.stack(target_mask, dim=0)

        # pos_weight = fg_mask.sum().to(self.device).clamp(0.,10.)
        # pos_weight = pos_weight if pos_weight > 0 else None
        pos_weight = None

        logits = self.upsampler(p3)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, target_mask, reduction="sum", pos_weight=pos_weight
        )
        return loss


class RoiClassifierHead(torch.nn.Module):
    def __init__(
        self,
        scale_factor: list = [2.0, 3.0],
        box_size: int = 98,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(384),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # self.loss = nn.SmoothL1Loss(
        #     reduction="sum"
        # )
        self.loss = nn.BCEWithLogitsLoss(reduction="sum")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.image_encoder = AutoModel.from_pretrained(
            "facebook/dinov2-with-registers-small",
            torch_dtype="auto",
            device_map="auto",
        )
        # self.tain_image_encoder = False
        # self.image_encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        # )

        # self._non_persistent_buffers_set.add('image_encoder')

        self.box_size = torch.Tensor([box_size]).float()

    def forward(self, x: dict):
        p3 = x.get("p3", None)
        p4 = x.get("p4", None)
        gt_boxes = x.get("gt_bboxes")
        pred_boxes = x.get("pred_bboxes")
        target_labels = x.get("target_labels")
        img = x.get("img")

        if p3 is None or p4 is None:
            raise ValueError("p3 or p4 is not available")

        self.device = p3.device

        self.image_encoder = self.image_encoder.to(self.device)
        self.mlp = self.mlp.to(self.device)
        gt_boxes = gt_boxes.to(self.device)
        pred_boxes = pred_boxes.to(self.device)
        self.box_size = self.box_size.to(self.device)

        positive_mask = target_labels > -1.0
        fp_tp_target_label = torch.zeros_like(target_labels).float()

        # extract ROI features for pred
        pred_features = self._extract_roi_features(
            boxes=pred_boxes,
            p3=p3,
            p4=p4,
            roi_align_shape=(self.box_size, self.box_size),
            img=img,
        )
        features = torch.zeros_like(pred_features)

        if positive_mask.any():
            # get ious
            box_ious = complete_intersection_over_union(
                preds=pred_boxes.detach()[positive_mask],
                target=gt_boxes[positive_mask],
                aggregate=False,
            )
            best_iou, best_gt_idx = box_ious.max(dim=1)

            # update target
            fp_tp_target_label[positive_mask] = best_iou.clamp(
                0.0
            )  # (best_iou > self.tp_iou_threshold)*1

            # extract positive roi features
            gt_features_pos = self._extract_roi_features(
                boxes=gt_boxes[positive_mask],
                p3=p3[positive_mask],
                p4=p4[positive_mask],
                roi_align_shape=(self.box_size, self.box_size),
                img=img[positive_mask],
            )
            features[positive_mask] = gt_features_pos

        else:
            features = pred_features

        logits = self.mlp(features)  # (M, nc)
        loss = self.loss(logits, fp_tp_target_label.unsqueeze(1))

        return loss

    def _extract_patches(self, images: torch.Tensor, boxes: torch.Tensor):
        """
        Extract patches from batched images given bounding boxes.

        Args:
            images (Tensor): (B, C, H, W) input images.
            boxes (Tensor): List of length B, each Tensor is (B, 4) containing [x1, y1, x2, y2]

        Returns:
            Tensor: cropped patches for each box.
        """
        B, C, H, W = images.shape
        image_patches = []
        images = images.cpu()
        boxes = boxes.detach().cpu().int()

        assert boxes.shape[0] == B, "Every bbox should correspond to an image"

        to_numpy = lambda x: x.permute(1, 2, 0).detach().cpu().numpy()
        pad_and_crop = A.Compose(
            [
                A.PadIfNeeded(min_height=self.box_size, min_width=self.box_size),
                A.CenterCrop(
                    height=self.box_size, width=self.box_size, pad_if_needed=True
                ),
                A.ToTensorV2(),
            ]
        )
        pad_and_crop = lambda patch: pad_and_crop(image=to_numpy(patch))["image"]

        for i in range(B):
            image = images[i]
            box = boxes[i]
            x1, y1, x2, y2 = box
            # Clamp coordinates to avoid indexing errors
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, W)
            y2 = min(y2, H)
            patch = image[:, y1:y2, x1:x2]
            if any([p != self.box_size for p in patch.shape[1:]]):
                patch = pad_and_crop(patch)
            image_patches.append(patch.unsqueeze(0))

        image_patches = torch.cat(image_patches, dim=0).to(self.device)
        images = images.to(self.device)
        boxes = boxes.to(self.device)
        return image_patches

    def _extract_roi_features(self, boxes, p3, p4, img, roi_align_shape):
        """
        Extract multi-scale RoI features and compute confidence multipliers

        Args:
            boxes: tensor (M, 4) with [x1, y1, x2, y2,]
            p3: feature map from YOLO P3 level (B, C, H, W)
            p4: feature map from YOLO P4 level (B, C, H, W)
            original_image: tensor (B, 3, H, W) original input image
            image_size: tuple of (height, width)

        Returns:
            confidence_multipliers: tensor (M,) with confidence adjustment factors
        """

        image_size = img.shape[2:]

        # Expand boxes by scale factor around center
        scaled_boxes = self._expand_boxes(boxes, image_size)
        batch_indices = torch.arange(boxes.shape[0], device=self.device).view(-1, 1)
        roi_box = torch.cat([batch_indices, scaled_boxes], dim=1)

        # RoI align on P3 (higher resolution)
        roi_features_p3 = roi_align(
            p3,
            roi_box,
            output_size=roi_align_shape,
            spatial_scale=1.0 / 8,  # P3 stride
            aligned=True,
        )  # (M, C, *roi_align_shape)

        # RoI align on P4 (lower resolution, larger receptive field)
        roi_features_p4 = roi_align(
            p4,
            roi_box,
            output_size=roi_align_shape,
            spatial_scale=1.0 / 16,  # P4 stride
            aligned=True,
        )  # (M, C, *roi_align_shape)

        # RoI original images
        original_crops = roi_align(
            img,
            roi_box,
            output_size=roi_align_shape,  # Standard input size for image encoder
            spatial_scale=1.0,  # Original image scale
            aligned=True,
        )  # (M, 3, *roi_align_shape)

        # original_crops = self._extract_patches(images=img, boxes=roi_box[:, 1:])
        # grid_crops = torchvision.utils.make_grid(img).permute(1,2,0)
        # import matplotlib.pyplot as plt
        # plt.imsave('test.jpg',grid_crops)
        # img_bbox = torchvision.utils.draw_bounding_boxes(img[0],roi_box[:,1:].int()).permute(1,2,0)
        # plt.imsave('test_img.jpg',img_bbox)

        # Encode original image crops
        # if self.tain_image_encoder:
        #     img_features = self.image_encoder(original_crops)
        # else:
        with torch.no_grad():
            img_features = self.image_encoder(original_crops)

        try:
            img_features = img_features.pooler_output
        except:
            pass

        # Global average pooling to get feature vectors
        roi_feat_p3_pooled = F.adaptive_avg_pool2d(roi_features_p3, (1, 1)).flatten(1)
        roi_feat_p4_pooled = F.adaptive_avg_pool2d(roi_features_p4, (1, 1)).flatten(1)

        # Concatenate P3 and P4 features for this scale
        features = torch.cat(
            [
                roi_feat_p4_pooled,
                roi_feat_p3_pooled,
                img_features,
            ],
            dim=1,
        )

        return features

    def _expand_boxes(self, boxes: torch.Tensor, image_size: tuple):
        """
        Expand bounding boxes by scale_factor around their centers

        Args:
            boxes: tensor (N, 4) in xyxy format
            scale_factor: float, expansion factor
            image_size: tuple (height, width)

        Returns:
            expanded_boxes: tensor (N, 4) in xyxy format, clamped to image bounds
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # Calculate centers and dimensions
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # w = x2 - x1
        # h = y2 - y1

        # Expand dimensions
        new_w = self.box_size  # w * scale_factor
        new_h = self.box_size  #  h * scale_factor

        # Calculate new coordinates
        new_x1 = cx - new_w / 2
        new_y1 = cy - new_h / 2
        new_x2 = cx + new_w / 2
        new_y2 = cy + new_h / 2

        # Clamp to image boundaries
        new_x1 = torch.clamp(new_x1, 0, image_size[1])
        new_y1 = torch.clamp(new_y1, 0, image_size[0])
        new_x2 = torch.clamp(new_x2, 0, image_size[1])
        new_y2 = torch.clamp(new_y2, 0, image_size[0])

        return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)


class DetectionSystem(DetectionModel):
    def __init__(
        self,
        *args,
        roi_classifier_layers: dict = {},
        count_regressor_layers: int = None,
        area_regressor_layers: int = None,
        mask_p3_layer_indx: int = None,
        roi_scale_factor: list = [
            2.0,
        ],
        pos_weight: float | None = 1.0,
        fp_tp_loss_weight: float = 0.0,
        mask_loss_weight: float = 0.0,
        count_loss_weight: float = 0.0,
        area_loss_weight: float = 0.0,
        **kwargs,
    ):
        self._is_operational = False

        # auxilary tasks
        self.activations = dict()
        self.pred_aux = dict()
        self.hooks_handles = []

        super().__init__(*args, **kwargs)

        self.pos_weight = pos_weight
        self.fp_tp_loss_weight = fp_tp_loss_weight
        self.count_loss_weight = count_loss_weight
        self.area_loss_weight = area_loss_weight
        self.mask_loss_weight = mask_loss_weight

        if pos_weight is not None:
            assert pos_weight > 0.0, (
                f"Expected positive weight > 0.0 but received {pos_weight}"
            )
        assert count_loss_weight >= 0.0
        assert area_loss_weight >= 0.0
        assert fp_tp_loss_weight >= 0.0

        assert isinstance(roi_classifier_layers, dict)
        assert (
            isinstance(count_regressor_layers, int) or count_regressor_layers is None
        ), f"Found type:'{type(count_regressor_layers)}'"
        assert (
            isinstance(area_regressor_layers, int) or area_regressor_layers is None
        ), f"Found type:'{type(area_regressor_layers)}'"

        self.roi_classifier_layers = roi_classifier_layers
        if self.roi_classifier_layers:
            self.roi_classifier = RoiClassifierHead(
                # nc=self.yaml["nc"],
                scale_factor=roi_scale_factor
            )

        self.count_regressor_layers = count_regressor_layers
        if count_regressor_layers and count_loss_weight > 0.0:
            self.count_regressor = RegressorHead(out_channels=64)

        self.area_regressor_layers = area_regressor_layers
        if self.area_regressor_layers and area_loss_weight > 0.0:
            self.area_regressor = RegressorHead(out_channels=64)

        self.mask_p3_layer_indx = mask_p3_layer_indx
        if self.mask_p3_layer_indx and self.mask_loss_weight > 0.0:
            self.mask_head = MaskHead()

        self._is_operational = True

        self.initialize_lazy_modules()

    def initialize_lazy_modules(self):
        # initialize Lazy modules
        with torch.no_grad():
            self.add_hooks()
            self._predict_once(torch.rand(1, 3, 256, 256))
            self._forward_aux()
            self.remove_hooks()
            self.activations = dict()
            self.pred_aux = dict()

    # get intermediate features p3, p4 etc.
    def add_hooks(
        self,
    ):
        logger.debug("adding hooks")

        def hook_get_activation(name):
            def hook(module, args, output):
                self.activations[name] = output
                # logger.debug(f"saving activation {name}")
                return None

            return hook

        # registering hooks to intermediate layers
        layers = [
            self.count_regressor_layers,
            self.area_regressor_layers,
            self.mask_p3_layer_indx,
        ] + list(self.roi_classifier_layers.values())
        layers_to_register = [a for a in layers if a is not None]

        for l in layers_to_register:
            handle = self.model[l].register_forward_hook(hook_get_activation(l))
            self.hooks_handles.append(handle)

    def remove_hooks(
        self,
    ):
        try:
            for a in self.hooks_handles:
                a.remove()
                logger.debug("removing hook")
            self.hooks_handles = []
        except Exception as e:
            print(e)

    def _forward_aux(
        self,
    ) -> None:
        if not self._is_operational:
            return None

        # count regressor
        if self.count_regressor_layers and self.count_loss_weight > 0.0:
            pred_count = self.count_regressor(
                self.activations[self.count_regressor_layers]
            )
            self.pred_aux["pred_count"] = pred_count

        # area regressor
        if self.area_regressor_layers and self.area_loss_weight > 0.0:
            pred_area = self.area_regressor(
                self.activations[self.area_regressor_layers]
            )
            self.pred_aux["pred_area"] = pred_area

        return None

    def forward(self, x, *args, **kwargs):
        if self.training == False:
            self.remove_hooks()

        if isinstance(x, dict):  # for cases of training and validating while training.
            if self.training and len(self.hooks_handles) < 1:
                self.add_hooks()
            return self.loss(x, *args, **kwargs)

        return self.predict(x, *args, **kwargs)

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return CustomLoss(
            model=self,
            pos_weight=self.pos_weight,
            fp_tp_loss_weight=self.fp_tp_loss_weight,
            count_loss_weight=self.count_loss_weight,
            area_loss_weight=self.area_loss_weight,
        )


class DetectionSystemTrainer(DetectionTrainer):
    def get_model(self, cfg, weights, verbose=True):
        """Returns a customized detection model instance configured with specified config and weights."""

        args = json.loads(os.environ.get("args_det_system", json.dumps(dict())))

        model = DetectionSystem(
            **args,
            cfg=cfg,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io
        # from copy import deepcopy

        self.model.remove_hooks()

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": self.model,  # resume and final checkpoints derive from EMA
                # "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                # "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(
                serialized_ckpt
            )  # save epoch, i.e. 'epoch3.pt'


class CustomYOLO(YOLO):
    def __init__(
        self,
        count_regressor_layers: int = None,
        area_regressor_layers: int = None,
        roi_classifier_layers: dict = None,
        mask_p3_layer_indx: int = None,
        roi_scale_factor: list[float] = [2.0, 3.0, 4.0],
        pos_weight: float | None = None,
        fp_tp_loss_weight: float = 0.0,
        count_loss_weight: float = 0.0,
        area_loss_weight: float = 0.0,
        mask_loss_weight: float = 0.0,
        model="yolo11n.pt",
        task="detect",
        verbose=False,
    ):
        super().__init__(model=model, task=task, verbose=verbose)

        # if count_regressor_layers is not None:
        #     assert isinstance(count_regressor_layers, int), (
        #         f"Expected type 'int' but received {type(count_regressor_layers)}"
        #     )
        #     # os.environ["count_regressor_layers"] = str(count_regressor_layers)

        # if area_regressor_layers is not None:
        #     assert isinstance(area_regressor_layers, int), (
        #         f"Expected type 'int' but received {type(area_regressor_layers)}"
        #     )
        #     # os.environ["area_regressor_layers"] = str(area_regressor_layers)
        #     args_det_system['area_regressor_layers'] = area_regressor_layers

        # if roi_classifier_layers is not None:
        #     assert isinstance(roi_classifier_layers, dict), (
        #         f"Expected type 'dict' but received {type(roi_classifier_layers)}"
        #     )
        # os.environ["roi_classifier_layers"] = json.dumps(roi_classifier_layers)

        args_det_system = dict()
        args_det_system["count_regressor_layers"] = count_regressor_layers
        args_det_system["area_regressor_layers"] = area_regressor_layers
        args_det_system["roi_classifier_layers"] = roi_classifier_layers or dict()
        args_det_system["pos_weight"] = pos_weight
        args_det_system["roi_scale_factor"] = roi_scale_factor
        args_det_system["fp_tp_loss_weight"] = fp_tp_loss_weight
        args_det_system["count_loss_weight"] = count_loss_weight
        args_det_system["area_loss_weight"] = area_loss_weight
        args_det_system["mask_p3_layer_indx"] = mask_p3_layer_indx
        args_det_system["mask_loss_weight"] = mask_loss_weight

        # add to environment variables
        os.environ["args_det_system"] = json.dumps(args_det_system)

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        from ultralytics.models import yolo

        return {
            "detect": {
                "model": DetectionSystem,
                "trainer": DetectionSystemTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
        }
