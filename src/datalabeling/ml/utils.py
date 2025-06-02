import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np
import yaml, json
from itertools import product

from ultralytics.utils.loss import v8DetectionLoss, E2EDetectLoss
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.ops import xywh2xyxy
from ultralytics import YOLO


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


# TODO test
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

    from .models import Detector

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

    def __init__(
        self,
        model,
        pos_weight: float = 1.0,
        fp_tp_loss_weight: float = 0.0,
        is_fp_tp_multiplier: bool = True,
        count_loss_weight=0.0,
        area_loss_weight=0.0,
    ):
        super().__init__(model=model)

        self.fp_tp_loss_weight = fp_tp_loss_weight
        self.count_loss_weight = count_loss_weight
        self.area_loss_weight = area_loss_weight
        self.is_fp_tp_multiplier = is_fp_tp_multiplier

        self.model = model
        self.count_loss = nn.SmoothL1Loss(reduction="sum")
        self.area_loss = nn.SmoothL1Loss(reduction="sum")

        assert isinstance(pos_weight, float)
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
        
        self.model._forward_aux() # collect area and count logits
        
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

    def sample_tn(
        self, img_height: int, img_width: int, w: int = 50, h: int = 50, num: int = 10
    ) -> torch.Tensor:
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

        count = 0
        boxes = []
        for x, y in product(xs, ys):
            if count == num:
                break
            box = torch.Tensor([x, y, x + w, y + h])
            boxes.append(box)
            count += 1

        return torch.vstack(boxes)

    def sample_pred_by_area_score(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        img_height: int,
        img_width: int,
        max_num=10,
        area_thresh=50**2,  #
        scores_range: tuple = (0.3, 0.65),
    ):
        """
        Sample bounding boxes based on a combination of bbox area and prediction scores.

        Args:
            pred_bboxes: torch.Size([b, 34000, 4]) in xyxy format [x1, y1, x2, y2]
            pred_scores: torch.Size([b, 34000, nc]) class prediction scores
            top_k: Number of samples to select per image
            area_weight: Weight for area component in sampling (0-1)
            score_weight: Weight for score component in sampling (0-1)

        Returns:
        """
        batch_size, n_preds, n_classes = pred_scores.shape

        selected_bboxes = []
        image_idx = []
        for i in range(batch_size):
            box_size = int(np.sqrt(area_thresh))
            boxes = self.sample_tn(
                img_height=img_height,
                img_width=img_width,
                w=box_size,
                h=box_size,
                num=max_num,
            )

            selected_bboxes.append(boxes)
            image_idx.append([i] * max_num)

        #     bboxes = pred_bboxes[i]  # [34000, 4]
        #     scores, _ = pred_scores[i].max(dim=1)  # [34000,]

        #     # Compute bbox areas
        #     widths = bboxes[:, 2] - bboxes[:, 0]  # x2 - x1
        #     heights = bboxes[:, 3] - bboxes[:, 1]  # y2 - y1
        #     areas = widths * heights  # Area = (x2 - x1) * (y2 - y1)

        #     # select based on area
        #     selector_areas = areas >= area_thresh

        #     # select based on scores
        #     selector_scores = (scores > min(scores_range)) & (scores < max(scores_range))

        #     # selected
        #     valid_mask = selector_areas & selector_scores
        #     valid_indices = torch.arange(0, n_preds, device=pred_scores.device)
        #     valid_indices = valid_indices[valid_mask]
        #     if valid_indices.shape[0] == 0:
        #         random_idx = torch.randint(low=0, high=valid_indices.shape[0], size=max_num)
        #         pass
        #     else:
        #         # size = min(max_num, valid_indices.shape[0])
        #         random_idx = torch.randint(low=0, high=valid_indices.shape[0], size=max_num)
        #         valid_indices = valid_indices[random_idx]
        #         # if size < max_num:
        #         #     pass
        # selected_bboxes.append(bboxes[valid_indices])
        # image_idx.append([i] * max_num)

        selected_bboxes = torch.vstack(selected_bboxes).to(self.device)
        image_idx = (
            torch.Tensor(image_idx).flatten().long().to(self.device)
        )

        return selected_bboxes, image_idx

    def get_cnf_scores_multiplier_from_fptp(
        self,
        target_bboxes: torch.Tensor,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch_images: torch.Tensor,
        target_labels: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        image_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        # batch has only negative samples

        if fg_mask.sum() < 1.0:
            _, _, img_height, img_width = batch_images.shape
            pred_bboxes, image_idx = self.sample_pred_by_area_score(
                pred_bboxes=pred_bboxes,
                pred_scores=pred_scores,
                img_height=img_height,
                img_width=img_width,
                max_num=5,
                area_thresh=50**2,
            )

            gt_bboxes = target_bboxes[fg_mask > 0.0]
            target_labels = target_labels[fg_mask > 0.0]

        else:
            gt_bboxes = target_bboxes[fg_mask]
            pred_bboxes = pred_bboxes[fg_mask]
            target_labels = target_labels[fg_mask]

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
        if (self.count_loss_weight > 0.0 or self.area_loss_weight > 0.0) and self.model.training:
            loss[3] += self.compute_count_area_loss(
                targets, scale_tensor=imgsz[[1, 0, 1, 0]]
            )
            loss[3] /= target_scores_sum

        # TODO:  debug fp_tp
        if (self.fp_tp_loss_weight > 0.0 or self.is_fp_tp_multiplier) and self.model.training:
            if fg_mask.sum() < 1.0:  # batch has only negative samples
                bbox_idx = target_gt_idx[fg_mask>0.]  # tensor([])
                image_idx = bbox_idx.clone()  # tensor([])
            else:
                bbox_idx = target_gt_idx[fg_mask].cpu()  # valid bbox indices
                image_idx = batch["batch_idx"][bbox_idx].long().cpu()  # mapping img -> bbox

            scores_multiplier, fp_tp_loss = self.get_cnf_scores_multiplier_from_fptp(
                target_bboxes=target_bboxes / stride_tensor,
                pred_bboxes=pred_bboxes.detach(),  # disable detach to allow gradient flowing through detection head as well
                pred_scores=pred_scores.detach(),
                batch_images=batch["img"],
                target_labels=target_labels,
                target_scores=target_scores,
                image_idx=image_idx,
                fg_mask=fg_mask,
            )

            # if self.is_fp_tp_multiplier:
            #     pred_scores[fg_mask] *= scores_multiplier
            # else:
            loss[3] += (fp_tp_loss * self.fp_tp_loss_weight / target_scores_sum)

        # Cls loss
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
            pos_weight, cfg, nc=self.data["nc"], ch=3, verbose=verbose and RANK == -1
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
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.reducer(x)
        x = x.flatten(1)
        return self.mlp(x)


class RoiClassifierHead(torch.nn.Module):
    def __init__(
        self,
        # nc: int,
        scale_factor: list = [2.0, 3.0],
        # label_smoothing: float = 0.0,
        # tp_iou_threshold:float=0.3
    ):
        super().__init__()

        # assert isinstance(nc, int), f"Expected 'int' but received {type(nc)}"
        assert isinstance(scale_factor, list), (
            f"Expected 'list' but received {type(scale_factor)}"
        )
        assert all([a > 0.99 for a in scale_factor]), (
            "All scaling factors must be greater than 1.0"
        )

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(384),
            nn.Linear(384, 128),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

        # self.tp_iou_threshold = tp_iou_threshold

        self.loss = nn.SmoothL1Loss(
            reduction="sum"
        )  # nn.BCEWithLogitsLoss(reduction="sum")

        self.image_encoder = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14_reg"
        ).half()

        self.register_buffer(
            "scale_factor", torch.Tensor(scale_factor), persistent=True
        )

    def forward(self, x: dict):
        p3 = x.get("p3", None)
        p4 = x.get("p4", None)
        gt_boxes = x.get("gt_bboxes")
        pred_boxes = x.get("pred_bboxes")
        target_labels = x.get("target_labels")
        img = x.get("img")

        
        device = p3.device        
        
        self.image_encoder = self.image_encoder.to(device)
        self.mlp = self.mlp.to(device)
        self.scale_factor = self.scale_factor.to(device)
        gt_boxes = gt_boxes.to(device)
        pred_boxes = pred_boxes.to(device)

        if gt_boxes.numel() > 0:
            box_ious = complete_intersection_over_union(
                preds=pred_boxes.detach(), target=gt_boxes, aggregate=False
            )
            best_iou, best_gt_idx = box_ious.max(dim=1)

            fp_tp_target_label = best_iou.unsqueeze(1).clamp(
                0.0
            )  # (best_iou > self.tp_iou_threshold)*1

        else:  # only negative samples
            num_boxes = pred_boxes.shape[0]
            fp_tp_target_label = torch.zeros((num_boxes,1)).to(device)

        if p3 is None or p4 is None:
            raise ValueError("p3 or p4 are not available")            
        

        # Predict confidence multipliers
        multiscale_features = self._extract_multiscale_roi_features(
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            p3=p3,
            p4=p4,
            roi_align_shape=(70, 70),
            img=img,
        )

        logits = self.mlp(multiscale_features)  # (M, nc)
        loss = self.loss(logits, fp_tp_target_label)

        # confidence_multipliers = confidence_multipliers.sigmoid().squeeze()  # (M,)

        return logits.sigmoid(), loss

    def _extract_multiscale_roi_features(
        self, gt_boxes, pred_boxes, p3, p4, img, roi_align_shape
    ):
        """
        Extract multi-scale RoI features and compute confidence multipliers

        Args:
            target: tensor (M, 4) with [x1, y1, x2, y2,]
            pred: tensor (M, 4) with [x1, y1, x2, y2,]
            p3: feature map from YOLO P3 level (B, C, H, W)
            p4: feature map from YOLO P4 level (B, C, H, W)
            original_image: tensor (B, 3, H, W) original input image
            image_size: tuple of (height, width)

        Returns:
            confidence_multipliers: tensor (M,) with confidence adjustment factors
        """

        device = p3.device
        image_size = img.shape[2:]
        batch_indices = torch.arange(pred_boxes.shape[0], device=device).view(-1, 1)

        # Prepare for RoI extraction
        all_roi_features = []

        for scale_factor in self.scale_factor:  # Local, Context, Environment
            # Expand boxes by scale factor around center
            scaled_pred_boxes = self._expand_boxes(pred_boxes, scale_factor, image_size)
            pred_roi_boxes = torch.cat([batch_indices, scaled_pred_boxes], dim=1)
            multipliers = [-1]
            roi_boxes = [
                pred_roi_boxes,
            ]
            
            if gt_boxes.numel() > 0:
                scaled_gt_boxes = self._expand_boxes(gt_boxes, scale_factor, image_size)
                gt_roi_boxes = torch.cat([batch_indices, scaled_gt_boxes], dim=1)
                multipliers.append(1)
                roi_boxes.append(gt_roi_boxes)

            # compute the difference between gt_roi_boxes and pred_roi_boxes
            roi_feat_p3_pooled = torch.zeros(1, device=device)
            roi_feat_p4_pooled = torch.zeros(1, device=device)
            img_features = torch.zeros(1, device=device)

            for m, roi_boxes in zip(multipliers, roi_boxes):
                # RoI align on P3 (higher resolution)
                roi_features_p3 = roi_align(
                    p3,
                    roi_boxes,
                    output_size=roi_align_shape,
                    spatial_scale=1.0 / 8,  # P3 stride
                    aligned=True,
                )  # (M, C, *roi_align_shape)

                # RoI align on P4 (lower resolution, larger receptive field)
                roi_features_p4 = roi_align(
                    p4,
                    roi_boxes,
                    output_size=roi_align_shape,
                    spatial_scale=1.0 / 16,  # P4 stride
                    aligned=True,
                )  # (M, C, *roi_align_shape)

                # RoI original images
                original_crops = roi_align(
                    img,
                    roi_boxes,
                    output_size=(196, 196),  # Standard input size for image encoder
                    spatial_scale=1.0,  # Original image scale
                    aligned=True,
                )  # (M, 3, 196, 196)

                # Encode original image crops
                with torch.no_grad():
                    img_features = m * self.image_encoder(original_crops) + img_features

                # Global average pooling to get feature vectors
                roi_feat_p3_pooled = (
                    m * F.adaptive_avg_pool2d(roi_features_p3, (1, 1)).flatten(1)
                    + roi_feat_p3_pooled
                )
                roi_feat_p4_pooled = (
                    m * F.adaptive_avg_pool2d(roi_features_p4, (1, 1)).flatten(1)
                    + roi_feat_p4_pooled
                )

            # Concatenate P3 and P4 features for this scale
            scale_features = torch.cat(
                [roi_feat_p3_pooled, roi_feat_p4_pooled, img_features], dim=1
            )
            all_roi_features.append(scale_features)

        # Concatenate all scales
        multiscale_features = torch.cat(all_roi_features, dim=1)

        return multiscale_features

    def _expand_boxes(self, boxes, scale_factor, image_size):
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
        w = x2 - x1
        h = y2 - y1
        
        

        # Expand dimensions
        new_w = w * scale_factor
        new_h = h * scale_factor

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
        roi_scale_factor: list = [1.0, 2.0, 4.0],
        pos_weight: float = 1.0,
        fp_tp_loss_weight: float = 0.0,
        is_fp_tp_multiplier: bool = False,
        count_loss_weight: float = 0.0,
        area_loss_weight: float = 0.0,
        **kwargs,
    ):
        self._is_operational = False
        super().__init__(*args, **kwargs)

        self.pos_weight = pos_weight
        self.fp_tp_loss_weight = fp_tp_loss_weight
        self.is_fp_tp_multiplier = is_fp_tp_multiplier
        self.count_loss_weight = count_loss_weight
        self.area_loss_weight = area_loss_weight

        assert pos_weight > 0.0, (
            f"Expected positive weight > 0.0 but received {pos_weight}"
        )
        assert count_loss_weight >= 0.0
        assert area_loss_weight >= 0.0
        assert fp_tp_loss_weight >= 0.0

        if is_fp_tp_multiplier:
            assert fp_tp_loss_weight == 0, "should be 0 when is_fp_tp_multiplier=True"

        assert isinstance(roi_classifier_layers, dict)
        assert (
            isinstance(count_regressor_layers, int) or count_regressor_layers is None
        ), f"Found type:'{type(count_regressor_layers)}'"
        assert (
            isinstance(area_regressor_layers, int) or area_regressor_layers is None
        ), f"Found type:'{type(area_regressor_layers)}'"

        # auxilary tasks
        self.activations = dict()
        self.pred_aux = dict()
        self.hooks_handles = []

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
        
        self.add_hooks() # adding hooks
        self._is_operational = True

        # initialize Lazy modules
        with torch.no_grad():
            self._predict_once(torch.rand(1, 3, 256, 256))
            self._forward_aux()
            self.activations = dict()
    
    # get intermediate features p3, p4 etc.
    def add_hooks(self,):
        
        logger.info("adding hooks")
        
        def hook_get_activation(name):
            def hook(module, args, output):
                self.activations[name] = output
                return None
            return hook
        
        # registering hooks to intermediate layers
        layers = [self.count_regressor_layers, self.area_regressor_layers] + list(
            self.roi_classifier_layers.values()
        )
        layers_to_register = [a for a in layers if a is not None]
        
        for l in layers_to_register:
            handle = self.model[l].register_forward_hook(hook_get_activation(l))
            self.hooks_handles.append(handle)
        
    def remove_hooks(self,):
        for a in self.hooks_handles:
            a.remove()
            logger.info("removing hook")
            
        self.hooks_handles = []
    

    def _forward_aux(
        self,
    ) -> None:
        
        if not self._is_operational:
            return None
        
        # adding hooks
        if len(self.hooks_handles) == 0:
            self.add_hooks()
            
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
    
        if isinstance(x, dict):  # for cases of training and validating while training.
            
            if self.training and len(self.hooks_handles)<1:
                self.add_hooks()
            else:
                self.remove_hooks()
            
            return self.loss(x, *args, **kwargs)
        
        return self.predict(x, *args, **kwargs)
    
    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return CustomLoss(
            model=self,
            pos_weight=self.pos_weight,
            fp_tp_loss_weight=self.fp_tp_loss_weight,
            is_fp_tp_multiplier=self.is_fp_tp_multiplier,
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
        from copy import deepcopy

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": self.model,  # resume and final checkpoints derive from EMA
                # "ema": deepcopy(self.ema.ema).half(),
                # "updates": self.ema.updates,
                # "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                # "date": datetime.now().isoformat(),
                # "version": __version__,
                # "license": "AGPL-3.0 (https://ultralytics.com/license)",
                # "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # save mosaic checkpoint


class CustomYOLO(YOLO):
    def __init__(
        self,
        count_regressor_layers: int = None,
        area_regressor_layers: int = None,
        roi_classifier_layers: dict = None,
        roi_scale_factor: list[float] = [2.0, 3.0, 4.0],
        pos_weight: float = 1.0,
        fp_tp_loss_weight: float = 0.0,
        is_fp_tp_multiplier: bool = False,
        count_loss_weight: float = 0.0,
        area_loss_weight: float = 0.0,
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
        args_det_system["is_fp_tp_multiplier"] = is_fp_tp_multiplier

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
