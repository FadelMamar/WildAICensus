import logging
import os
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import json
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.functional.detection import complete_intersection_over_union
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.data import converter

from ..ml.interface import InferenceEngine
from ..ml.utils import remove_label_cache
from .config import EvaluationConfig
from .dataset_loader import LabelingDataset

logger = logging.getLogger(__name__)


class Metrics:
    def __init__(self, config: EvaluationConfig):
        self.mean_ap = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            max_detection_thresholds=[1, 10, 100],
            iou_thresholds=[0.15, 0.25, 0.35, 0.5, 0.75, 0.85, 0.95],
        )

        self.config = config

        self.bbox_cols = ["x_min", "y_min", "x_max", "y_max"]

    # TODO: debug
    def run(self, dataset: LabelingDataset, max_workers: int = 1) -> pd.DataFrame:
        logger.info("Computing evaluation metrics per image...")

        def iterator(df_pred, df_gt):
            image_paths = df_pred["file_name"].unique()

            for path in image_paths:
                pred = (
                    df_pred.loc[df_pred["file_name"] == path, :]
                    .reset_index(drop=True)
                    .copy()
                )
                gt = (
                    df_gt.loc[df_gt["file_name"] == path, :]
                    .reset_index(drop=True)
                    .copy()
                )
                yield gt, pred

        data = dataset.data
        data.sort_values("file_name", inplace=True)

        assert not data.empty, "The dataset is empty. Please check"

        # partition rows
        df_pred = (
            data.loc[data["is_annot"] == False, :]
            .dropna(subset="is_annot", axis=0)
            .drop(columns="is_annot")
        )
        df_gt = (
            data.loc[data["is_annot"] == True, :]
            .dropna(subset="is_annot", axis=0)
            .drop(columns="is_annot")
        )

        if df_pred.empty:
            raise ValueError("There are no predictions.")

        # drop NaNs
        df_gt = df_gt.dropna(
            axis=0,
            subset=["label", "x_min", "y_min", "x_max", "y_max"],
            how="any",
        )

        df_results = []

        def func(*x):
            return self._run_per_image(df_gt_i=x[0], df_pred_i=x[1])

        loader = iterator(df_pred=df_pred, df_gt=df_gt)

        with ThreadPool(max_workers) as executor:
            for df_eval in tqdm(executor.map(func, loader), desc="computing..."):
                # df_eval = self._run_per_image(df_gt_i=df_gt_i, df_pred_i=df_pred_i)
                if not df_eval.empty:
                    df_results.append(df_eval)

        if len(df_results) > 0:
            df_results = pd.concat(
                df_results, ignore_index=True, sort=False
            ).reset_index(drop=True)
        else:
            df_results = pd.DataFrame()
            logger.warning("There are 0 positives samples and 0 Detections were found.")

        return df_results

    def _get_bbox(self, gt: pd.DataFrame):
        return gt[self.bbox_cols].to_numpy().astype(float)

    def _run_per_image(
        self, df_gt_i: pd.DataFrame, df_pred_i: pd.DataFrame
    ) -> pd.DataFrame:
        # check validity
        unique_images_gt = set(df_gt_i["file_name"])
        unique_images_pred = set(df_pred_i["file_name"])
        assert len(unique_images_gt) <= 1 and len(unique_images_pred) <= 1, (
            "df_gt_i or df_pred_i has data for more than one image. Not Allowed!"
        )
        assert unique_images_gt.issubset(unique_images_pred), (
            "groundtruth image are does not match prediction image"
        )

        # gt
        gt = torch.from_numpy(self._get_bbox(gt=df_gt_i).clip(min=0))
        labels = df_gt_i.loc[:, "label"].to_numpy().astype(int)

        # pred
        pred = self._get_bbox(gt=df_pred_i).clip(
            min=0,
        )

        pred = torch.from_numpy(pred)
        pred_score = df_pred_i.loc[:, "score"].to_numpy()
        classes = df_pred_i.loc[:, "label"].to_numpy().astype(int)

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

        # compute iou
        box_ious = complete_intersection_over_union(
            preds=pred, target=gt, aggregate=False
        )

        # compute mAP
        metric = self.mean_ap(preds=pred_list, target=target_list)

        stats = dict(
            map50=metric["map_50"].cpu().item(),
            map75=metric["map_75"].cpu().item(),
            all_scores=pred_score.tolist(),
            max_scores=pred_score.max(),
        )

        ## get FPs
        if df_gt_i.empty:
            df_pred_i["TP"] = 0
            df_pred_i["FP"] = len(df_pred_i)

            # rename columns
            df_eval = df_pred_i.rename(
                columns={
                    col: f"pred_{col}"
                    for col in df_pred_i.columns
                    if col != "file_name"
                },
            )
            for k, v in stats.items():
                df_eval[k] = v

            return df_eval

        ## get FNs
        # TODO: make it work for multiclass?
        # For each prediction: find best-matching GT
        best_iou, best_gt_idx = box_ious.max(dim=1)
        df_pred_i["matching_gt"] = "None"
        df_pred_i["matching_gt"] = df_pred_i["matching_gt"].astype("object")
        df_pred_i["pred_label"] = "None"

        for i in range(len(df_pred_i)):
            df_pred_i.loc[i, "TP"] = best_iou[i].item() >= self.config.tp_iou_threshold
            df_pred_i.loc[i, "FP"] = best_iou[i].item() < self.config.tp_iou_threshold
            df_pred_i.loc[i, "best_ciou"] = best_iou[i].item()
            df_pred_i.loc[i, "matching_gt"] = (
                json.dumps(gt[best_gt_idx[i]].numpy().tolist())
                if df_pred_i.loc[i, "TP"]
                else "None"
            )

        # For each ground-truth: mark FN if never matched
        worst_pred_iou, _ = box_ious.max(dim=0)
        for i in range(len(df_gt_i)):
            df_gt_i.loc[i, "FN"] = (
                worst_pred_iou[i].item() < self.config.tp_iou_threshold
            )

        # rename columns
        df_pred_i.rename(
            columns={
                col: f"pred_{col}" for col in df_pred_i.columns if col != "file_name"
            },
            inplace=True,
        )

        df_gt_i.rename(
            columns={col: f"gt_{col}" for col in df_gt_i.columns if col != "file_name"},
            inplace=True,
        )

        # merge dfs
        df_eval = []
        if not df_gt_i.empty:
            df_eval.append(df_gt_i)

        if not df_pred_i.empty:
            df_eval.append(df_pred_i)

        if len(df_eval) > 0:
            df_eval = pd.concat(df_eval, ignore_index=True, sort=False).reset_index(
                drop=True
            )
            for k, v in stats.items():
                df_eval[k] = v

        else:
            df_eval = pd.DataFrame()

        return df_eval


# =====================
# Performance Evaluation
# =====================
class PerformanceEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.label_format = None
        self.metrics: Metrics = Metrics(config=config)

    def run(
        self,
        dataset: LabelingDataset,
        pred_results_dir: str,
        load_results: bool = False,
        save_tag: str = "",
    ) -> pd.DataFrame | None:
        """Calculate performance metrics"""

        # when providing a list of images
        stem = (
            f"predictions-{save_tag}" + save_tag if len(save_tag) > 0 else "predictions"
        )
        save_path = os.path.join(pred_results_dir, stem + ".csv")

        # get prediction results
        if load_results:
            try:
                dataset.import_data(save_path)
            except FileNotFoundError:
                traceback.print_exc()
                raise FileNotFoundError()
            except:
                traceback.print_exc()
                raise ValueError()
        else:
            dataset.save_data_csv(save_path=save_path)

        # compute metrics per image
        df_metrics_per_image = self.metrics.run(dataset)

        return df_metrics_per_image


# TODO: debug
class CustomValidator(DetectionValidator):
    """From https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py
    Adapted to compute confusion matrix for a given iou threshold
    """

    def init_metrics(self, model):
        """
        Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (
                val.endswith(f"{os.sep}val2017.txt")
                or val.endswith(f"{os.sep}test-dev2017.txt")
            )
        )  # is COCO
        self.is_lvis = (
            isinstance(val, str) and "lvis" in val and not self.is_coco
        )  # is LVIS
        self.class_map = (
            converter.coco80_to_coco91_class()
            if self.is_coco
            else list(range(1, len(model.names) + 1))
        )
        self.args.save_json |= (
            self.args.val and (self.is_coco or self.is_lvis) and not self.training
        )  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(
            nc=self.nc, conf=self.args.conf, iou_thres=self.args.iou, task="detect"
        )
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])


# TODO: debug
def ultralytics_val(args: EvaluationConfig):
    # remove label.cache files
    remove_label_cache(data_config_yaml=args.data_config)

    for split in args.splits:
        print("-" * 20, split, "-" * 20)

        val_args = dict(model=args.weights, data=args.data_config)

        run_name = (
            args.name
            + "#"
            + split
            + f"#{round(args.conf_threshold * 100)}#{round(args.iou_threshold * 100)}#{args.augment}#{args.max_det}-"
        )

        validator = CustomValidator(
            args=val_args, save_dir=Path(args.project_name) / run_name
        )

        # set args
        validator.args.conf = args.conf_threshold
        validator.args.iou = args.iou_threshold
        validator.args.mode = "val"
        validator.args.imgsz = args.imgsz
        validator.args.batch = args.batch_size
        validator.args.device = args.device
        validator.args.augment = args.augment
        validator.args.split = split
        validator.args.name = run_name
        validator.args.project = args.project_name
        validator.args.max_det = args.max_det
        validator.args.save_crop = False
        validator.args.save_json = False
        validator.args.plots = args.plots
        validator.args.save_hybridd = args.save_hybrid
        validator.args.save_txt = args.save_txt
        validator.args.save_conf = args.save_txt

        # run evaluation
        results = validator()

        cf_matrix = validator.confusion_matrix.matrix
        labels = list(validator.names.values()) + ["background"]

        for i, label in enumerate(labels + ["background"]):
            if label == "background":
                break

            tp = cf_matrix[i, i]
            actual_positive = cf_matrix[:, i].sum()
            predicted_positive = cf_matrix[i, :].sum()
            # fp = predicted_positive - tp
            # fn = actual_positive - tp

            precision = tp / (predicted_positive + 1e-8)
            recall = tp / (actual_positive + 1e-8)
            f1score = 2 * precision * recall / (precision + recall + 1e-8)

            results = dict(
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1score=round(f1score, 4),
            )

            print(f"results for {label} : ", results, end="\n")

        return results


# =====================
# Reporting Interface
# =====================
class ReportGenerator:
    def __init__(
        self,
    ):
        pass

    def generate_performance_report(self, df_results_per_img: pd.DataFrame) -> None:
        """Generate comprehensive performance report"""
        pass

    def generate_hard_samples_report(self, df_hard_negatives: pd.DataFrame) -> None:
        """Generate report on challenging samples"""
        pass


# =====================
# Uncertainty Analysis
# =====================
class UncertaintyAnalyzer:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def run(self, df_results_per_img: pd.DataFrame) -> pd.DataFrame:
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

    def run(self, df_metrics_per_image: pd.DataFrame) -> pd.DataFrame:
        """Identify challenging samples based on multiple criteria"""

        selected_images = self._select(df_metrics_per_image)

        mask = df_metrics_per_image["file_name"].isin(selected_images)
        self.df_hard_negatives = df_metrics_per_image.loc[mask, :]

        return self.df_hard_negatives

    def save_selection_references(
        self, df_hard_negatives: pd.DataFrame, save_path: str
    ) -> None:
        df_hard_negatives["file_name"].to_csv(save_path, index=False, header=False)

    def _filter_ratio(
        self,
        df_results_per_img: pd.DataFrame,
        col: str = "pred_FP",
        ratio_thres: float = 0.2,
    ) -> list:
        ratio_name = {"pred_FP": "fp_tp_ratio", "gt_FN": "fn_tp_ratio"}

        ratio_name = ratio_name[col]
        # select images based on FPs
        tps_col = (
            df_results_per_img.groupby(by=["file_name"])[["pred_TP", col]]
            .sum()
            .sort_values("pred_TP", ascending=False)
            * 1
        ).reset_index()
        tps_col[ratio_name] = tps_col[col] / (tps_col["pred_TP"] + 1e-8)
        mask = tps_col[ratio_name] > ratio_thres
        selected_images = tps_col.loc[mask, "file_name"].tolist()

        df_hard_negatives = df_results_per_img.merge(
            tps_col, on="file_name", how="left"
        )

        out = df_hard_negatives.loc[
            df_hard_negatives["file_name"].isin(selected_images),
        ]

        return out["file_name"].to_list()

    # TODO: include score range [0.4,0.6] for uncertain samples?
    def _filter_uncertainty(self, df_results_per_img) -> list:
        # select images based on mAP and uncertainty
        df_results_per_img = self.uncertainty.run(df_results_per_img)

        mask_uncertainty = (
            df_results_per_img["uncertainty"] > self.config.uncertainty_threshold
        )

        mask_low_map = df_results_per_img["map50"] < self.config.map_threshold  # * (
        # df_results_per_img["map75"] < self.config.map_threshold
        # )
        mask_high_scores = (
            df_results_per_img[self.config.score_col] > self.config.score_threshold
        )
        mask_low_scores = df_results_per_img[self.config.score_col] < (
            1 - self.config.score_threshold
        )
        mask_selected = (
            mask_low_map * mask_high_scores
            + mask_low_map * mask_low_scores
            + mask_uncertainty
        )

        out = df_results_per_img.loc[mask_selected]

        return out["file_name"].to_list()

    def _select(self, df_results_per_img: pd.DataFrame) -> list[str]:
        """Apply filtering"""

        # select images based on FPs
        selected_images_fp = self._filter_ratio(
            df_results_per_img,
            col="pred_FP",
            ratio_thres=self.config.fp_tp_ratio_threshold,
        )
        selected_images = set(selected_images_fp)

        # select images based on FNs
        if "gt_FN" in df_results_per_img.columns:
            selected_images_fn = self._filter_ratio(
                df_results_per_img,
                col="gt_FN",
                ratio_thres=self.config.fn_tp_ratio_threshold,
            )
            selected_images = selected_images.union(selected_images_fn)
        else:
            logger.info("No False Negatives were found")

        # select images based on uncertainty
        selected_images = selected_images.union(
            self._filter_uncertainty(df_results_per_img)
        )

        # select interesting columns and drop duplicates
        # cols = [
        #     "file_name",
        #     "map50",
        #     "map75",
        #     "all_scores",
        #     "uncertainty",
        #     "fp_tp_ratio",
        # ]

        # if "gt_FN" in df_results_per_img.columns:
        #     cols.append("fn_tp_ratio")

        # df_hard_negatives = (
        #     df_hard_negatives[cols].drop_duplicates("file_name").reset_index(drop=True)
        # )

        return selected_images


# =====================
# Main Controller
# =====================
class ModelEvaluator:
    def __init__(self, eval_config: EvaluationConfig):
        self.evaluator = PerformanceEvaluator(eval_config)
        self.uncertainty = UncertaintyAnalyzer(eval_config)
        self.sample_selector = HardSampleSelector(eval_config)
        self.reporter = ReportGenerator()

    def run(
        self,
        engine: InferenceEngine,
        dataset: LabelingDataset,
        pred_results_dir: str = None,
        save_tag: str = "",
        load_results: bool = False,
    ) -> None:
        """Complete evaluation workflow"""

        # Load data
        df_metrics_per_image = self.evaluator.run(
            engine=engine,
            dataset=dataset,
            pred_results_dir=pred_results_dir,
            load_results=load_results,
            save_tag=save_tag,
        )

        # Analyze results
        df_hard_negatives = self.sample_selector.run(df_metrics_per_image)

        # Generate reports
        self.reporter.generate_performance_report(df_metrics_per_image)
        self.reporter.generate_hard_samples_report(df_hard_negatives)

        return df_metrics_per_image, df_hard_negatives
