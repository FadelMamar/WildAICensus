import logging
import os

import numpy as np
import pandas as pd
import json
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.functional.detection import complete_intersection_over_union
from tqdm import tqdm

from ..ml.models import Detector
from .config import DataConfig, EvaluationConfig
from .io import DataHandler

logger = logging.getLogger(__name__)


# =====================
# Performance Evaluation
# =====================
class PerformanceEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.label_format = None
        self.predictions, self.ground_truth = None, None

    # TODO:  debug for negative samples
    def evaluate(
        self,
        images_dirs: list[str],
        pred_results_dir: str,
        detector: Detector,
        images_paths: list[str] = None,
        load_results: bool = False,
        save_tag: str = "",
    ) -> pd.DataFrame:
        """Calculate performance metrics"""

        self.predictions, self.ground_truth = self.get_preds_targets(
            images_dirs=images_dirs,
            pred_results_dir=pred_results_dir,
            detector=detector,
            images_paths=images_paths,
            load_results=load_results,
            save_tag=save_tag,
        )

        results_per_img, df_eval = self._calculate_base_metrics(
            self.predictions.copy(), self.ground_truth.copy()
        )
        metrics = results_per_img.merge(df_eval, on="file_name", how="left")
        return metrics

    def _calculate_base_metrics(
        self, df_pred: pd.DataFrame, df_gt: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute precision, recall, mAP etc."""

        logger.info("Computing TP, FP and mAP.")

        m_ap = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            max_detection_thresholds=[1, 10, 100],
            iou_thresholds=[0.15, 0.25, 0.35, 0.5, 0.75, 0.85, 0.95],
        )

        ciou = complete_intersection_over_union

        map_50s = list()
        maps_75s = list()
        max_scores = list()
        all_scores = list()
        pred_flags = []
        gt_flags = []

        image_paths = df_pred["file_name"].unique()

        for image_path in tqdm(image_paths, desc="Computing metrics"):
            # get gt
            mask_gt = df_gt["file_name"] == image_path
            df_gt_i = df_gt.loc[mask_gt, :].iloc[:, 1:]
            gt = torch.from_numpy(self._get_bbox(gt=df_gt_i))
            labels = df_gt.loc[mask_gt, "category_id"].to_numpy().astype(int)

            # get preds
            mask_pred = df_pred["file_name"] == image_path
            df_pred_i = df_pred.loc[mask_pred, ["x_min", "y_min", "x_max", "y_max"]]
            pred = np.clip(
                df_pred_i.to_numpy(), a_min=0, a_max=df_pred_i.to_numpy().max()
            )
            pred = torch.from_numpy(pred)
            pred_score = df_pred.loc[mask_pred, "score"].to_numpy()
            classes = df_pred.loc[mask_pred, "category_id"].to_numpy().astype(int)
            max_scores.append(pred_score.max())
            all_scores.append(pred_score)

            # compute mAPs
            pred_list = [
                {
                    "boxes": pred,
                    "scores": torch.from_numpy(pred_score),
                    "labels": torch.from_numpy(classes),
                }
            ]
            target_list = [
                {
                    "boxes": gt,
                    "labels": torch.from_numpy(labels),
                }
            ]

            metric = m_ap(preds=pred_list, target=target_list)
            map_50s.append(metric["map_50"].item())
            maps_75s.append(metric["map_75"].item())

            df_pred_i = df_pred.loc[mask_pred, :].copy().reset_index(drop=True)
            df_gt_i = df_gt.loc[mask_gt, :].copy().reset_index(drop=True)
            if df_gt_i.empty:
                df_pred_i["TP"] = 0
                df_pred_i["FP"] = len(df_pred_i)
                pred_flags.append(df_pred_i)
                continue

            df_gt_i[["x_min", "y_min", "x_max", "y_max"]] = self._get_bbox(
                gt=df_gt_i.iloc[:, 1:]
            )

            # TODO: make it work for multiclass?
            # compute ious
            box_ious = ciou(preds=pred, target=gt, aggregate=False)
            # For each prediction: find best-matching GT
            best_iou, best_gt_idx = box_ious.max(dim=1)
            df_pred_i["matching_gt"] = "None"
            df_pred_i["matching_gt"] = df_pred_i["matching_gt"].astype("object")
            df_pred_i["pred_label"] = "None"
            df_pred_i["file_name"] = image_path
            for i in range(len(df_pred_i)):
                df_pred_i.loc[i, "TP"] = (
                    best_iou[i].item() >= self.config.tp_iou_threshold
                )
                df_pred_i.loc[i, "FP"] = (
                    best_iou[i].item() < self.config.tp_iou_threshold
                )
                df_pred_i.loc[i, "best_ciou"] = best_iou[i].item()
                df_pred_i.loc[i, "matching_gt"] = (
                    json.dumps(gt[best_gt_idx[i]].numpy().tolist())
                    if df_pred_i.loc[i, "TP"]
                    else "None"
                )
                # df_pred_i["pred_label"] = (
                #     json.dumps(df_gt_i.loc[best_gt_idx[i],'category_id'].numpy().tolist())
                #     if df_pred_i.loc[i, "TP"]
                #     else "None"
                # )
            pred_flags.append(df_pred_i)

            # For each ground-truth: mark FN if never matched
            worst_pred_iou, _ = box_ious.max(dim=0)
            df_gt_i["file_name"] = image_path
            for i in range(len(df_gt_i)):
                df_gt_i.loc[i, "FN"] = (
                    worst_pred_iou[i].item() < self.config.tp_iou_threshold
                )
            gt_flags.append(df_gt_i)

        results_per_img = {
            "map50": map_50s,
            "map75": maps_75s,
            "max_scores": max_scores,
            "all_scores": all_scores,
            "file_name": image_paths,
        }
        results_per_img = pd.DataFrame.from_dict(results_per_img, orient="columns")

        df_pred_flagged = pd.concat(pred_flags, ignore_index=True)
        df_pred_flagged.rename(
            columns={
                col: f"pred_{col}"
                for col in df_pred_flagged.columns
                if col != "file_name"
            },
            inplace=True,
        )

        df_gt_flagged = pd.concat(gt_flags, ignore_index=True)
        df_gt_flagged.rename(
            columns={
                col: f"gt_{col}" for col in df_gt_flagged.columns if col != "file_name"
            },
            inplace=True,
        )

        df_eval = pd.concat(
            [df_pred_flagged, df_gt_flagged], ignore_index=True, sort=False
        )  # df_gt_flagged.merge(df_pred_flagged,on='file_name',how='left')

        return results_per_img, df_eval

    def _get_bbox(self, gt: pd.DataFrame):
        return gt[["x_min", "y_min", "x_max", "y_max"]].to_numpy()

    def get_preds_targets(
        self,
        images_dirs: list[str],
        pred_results_dir: str,
        detector: Detector,
        images_paths: list[str] = None,
        load_results: bool = False,
        save_tag: str = "",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # when providing a list of images
        if images_paths is not None:
            assert images_dirs is None, "images_dirs should be None!"
            sfx = save_tag
            save_path = os.path.join(pred_results_dir, f"predictions-{sfx}.json")

            # get prediction results
            if load_results and os.path.exists(save_path):
                df_results = DataHandler.load_json_predictions(save_path)
            else:
                df_results = detector.predict_directory(
                    path_to_dir=None,
                    images_paths=images_paths,
                    as_dataframe=True,
                    save_path=save_path,
                )
            df_labels, label_format = DataHandler.load_yolo_groundtruth(
                images_dir=None, images_paths=images_paths, load_empty=True
            )
            self.label_format = label_format

            return df_results, df_labels

        # when providing directories of images
        df_results = list()
        df_labels = list()
        labels_format = set()
        for image_dir in images_dirs:
            sfx = str(image_dir).split(":\\")[-1].replace("\\", "_").replace("/", "_")
            sfx = sfx + save_tag
            save_path = os.path.join(pred_results_dir, f"predictions-{sfx}.json")

            # get prediction results
            if load_results and os.path.exists(save_path):
                results = DataHandler.load_json_predictions(save_path)
            else:
                results = detector.predict_directory(
                    path_to_dir=image_dir,
                    images_paths=None,
                    as_dataframe=True,
                    return_gps=True,
                    save_path=save_path,
                )
            df_results.append(results)

            # get targets
            labels, _format = DataHandler.load_yolo_groundtruth(
                images_dir=image_dir, images_paths=None, load_empty=True
            )
            df_labels.append(labels)

            # update and check for changes
            labels_format.add(_format)
            (
                len(labels_format) == 1,
                f"There are inconcistencies in the labels formats.Found {labels_format}",
            )

        # Labels format
        self.label_format = labels_format.pop()

        df_results = pd.concat(df_results, axis=0).reset_index(drop=True)
        df_labels = pd.concat(df_labels, axis=0).reset_index(drop=True)

        return df_results, df_labels


# =====================
# Reporting Interface
# =====================
class ReportGenerator:
    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler

    def generate_performance_report(self, metrics: pd.DataFrame) -> None:
        """Generate comprehensive performance report"""
        pass

    def generate_hard_samples_report(self, hard_samples: pd.DataFrame) -> None:
        """Generate report on challenging samples"""
        pass


# =====================
# Uncertainty Analysis
# =====================
class UncertaintyAnalyzer:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def calculate_uncertainty(self, df_results_per_img: pd.DataFrame) -> pd.DataFrame:
        """Calculate uncertainty metrics"""

        df_results_per_img = self._get_uncertainty(
            df_results_per_img,
            reoder_ascending=False,
        )

        return df_results_per_img

    def _get_uncertainty(
        self,
        df_results_per_img: pd.DataFrame,
        reoder_ascending: bool = False,
    ) -> pd.DataFrame:
        if self.config.uncertainty_method == "entropy":
            entropy_func = lambda x: -1 * (np.log(x) * x).sum()
            df_results_per_img["uncertainty"] = df_results_per_img["all_scores"].apply(
                entropy_func
            )

        elif self.config.uncertainty_method == "1-p":
            df_results_per_img["uncertainty"] = df_results_per_img["all_scores"].apply(
                lambda x: 1.0 - np.mean(x)
            )

        else:
            raise NotImplementedError(
                "uncertainty computing method is not implemented yet. entropy or 1-p"
            )

        df_results_per_img.sort_values(
            "uncertainty", axis=0, ascending=reoder_ascending, inplace=True
        )

        return df_results_per_img


# =====================
# Hard Sample Analysis
# =====================
class HardSampleSelector:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.df_hard_negatives = None
        self.uncertainty = UncertaintyAnalyzer(config=config)

    def select_hard_samples(self, df_results_per_img: pd.DataFrame) -> pd.DataFrame:
        """Identify challenging samples based on multiple criteria"""
        self.df_hard_negatives = self._filter(df_results_per_img)
        return self.df_hard_negatives

    def save_selection_references(
        self, df_hard_negatives: pd.DataFrame, save_path: str
    ) -> None:
        df_hard_negatives["file_name"].to_csv(save_path, index=False, header=False)

    def _filter(self, df_results_per_img: pd.DataFrame) -> pd.DataFrame:
        """Apply filtering"""

        # select images based on FPs and FNs
        tps_fps_fns = (
            df_results_per_img.groupby(by=["file_name"])[
                ["pred_TP", "pred_FP", "gt_FN"]
            ]
            .sum()
            .sort_values("pred_TP", ascending=False)
            * 1
        )
        tps_fps_fns = tps_fps_fns.reset_index()
        tps_fps_fns["fp_tp_ratio"] = tps_fps_fns["pred_FP"] / (
            tps_fps_fns["pred_TP"] + 1e-8
        )
        tps_fps_fns["fn_tp_ratio"] = tps_fps_fns["gt_FN"] / (
            tps_fps_fns["pred_TP"] + 1e-8
        )

        mask = tps_fps_fns["fn_tp_ratio"] > self.config.fn_tp_ratio_threshold
        mask = mask + (tps_fps_fns["fp_tp_ratio"] > self.config.fp_tp_ratio_threshold)

        selected_images = tps_fps_fns.loc[mask, "file_name"].tolist()

        df_hard_negatives = df_results_per_img.merge(
            tps_fps_fns, on="file_name", how="left"
        )
        df_hard_negatives = [
            df_hard_negatives.loc[
                df_hard_negatives["file_name"].isin(selected_images), :
            ],
        ]

        # select images based on mAP and uncertainty
        df_results_per_img = self.uncertainty.calculate_uncertainty(df_results_per_img)
        mask_low_map = (df_results_per_img["map50"] < self.config.map_threshold) * (
            df_results_per_img["map75"] < self.config.map_threshold
        )
        mask_high_scores = (
            df_results_per_img[self.config.score_col] > self.config.score_threshold
        )
        mask_low_scores = df_results_per_img[self.config.score_col] < (
            1 - self.config.score_threshold
        )
        mask_selected = (
            mask_low_map * mask_high_scores
            + mask_low_map * mask_low_scores
            + (df_results_per_img["uncertainty"] > self.config.uncertainty_threshold)
        )

        df_hard_negatives.append(df_results_per_img.loc[mask_selected])
        df_hard_negatives = (
            pd.concat(df_hard_negatives)
            .reset_index(drop=True)
            .drop_duplicates("file_name")
        )

        # select interesting columns and dropping duplicates
        cols = [
            "file_name",
            "map50",
            "map75",
            "all_scores",
            "uncertainty",
            "fn_tp_ratio",
            "fp_tp_ratio",
        ]
        df_hard_negatives = (
            df_hard_negatives[cols].drop_duplicates("file_name").reset_index(drop=True)
        )

        return df_hard_negatives


# =====================
# Main Controller
# =====================
class CVModelEvaluator:
    def __init__(self, data_config: DataConfig, eval_config: EvaluationConfig):
        self.data_handler = DataHandler(data_config)
        self.evaluator = PerformanceEvaluator(eval_config)
        self.uncertainty = UncertaintyAnalyzer(eval_config)
        self.sample_selector = HardSampleSelector(eval_config)
        self.reporter = ReportGenerator(self.data_handler)

    def run_full_evaluation(
        self,
        detector: Detector,
        images_dirs: list[str],
        images_paths: list[str],
        pred_results_dir: str = None,
        save_tag: str = "",
        load_results: bool = False,
    ) -> None:
        """Complete evaluation workflow"""
        # Load data
        predictions, ground_truth = self.evaluator.get_preds_targets(
            detector=detector,
            images_dirs=images_dirs,
            images_paths=images_paths,
            pred_results_dir=pred_results_dir,
            load_results=load_results,
            save_tag=save_tag,
        )

        # Calculate metrics
        metrics, detailed_metrics = self.evaluator.evaluate(predictions, ground_truth)
        predictions = self.uncertainty.calculate_uncertainty(predictions)

        # Analyze results
        hard_samples = self.sample_selector.select_hard_samples(predictions)

        # Generate reports
        self.reporter.generate_performance_report(metrics)
        self.reporter.generate_hard_samples_report(hard_samples)

        return metrics, hard_samples
