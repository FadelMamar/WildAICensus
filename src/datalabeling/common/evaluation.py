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
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.data import converter
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from functools import lru_cache


from ..ml.interface import InferenceEngine
from ..ml.models import Detector
from .config import EvaluationConfig, PredictionConfig
from .dataset_loader import LabelingDataset

logger = logging.getLogger("Evaluation")


class Metrics:
    def __init__(
        self,
        score_threshold: float = 0.25,
        tp_method: str = "iou",
        tp_iou_threshold: float = 0.5,
        tp_distance_threshold: float = 50.0,
    ):
        self.mean_ap = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            max_detection_thresholds=[1, 10, 100],
            # iou_thresholds=[0.15, 0.25, 0.35, 0.5, 0.75, 0.85, 0.95],
        )
        self.score_threshold = score_threshold
        self.bbox_cols = ["x_min", "y_min", "x_max", "y_max"]
        self.tp_iou_threshold = tp_iou_threshold
        self.tp_distance_threshold = tp_distance_threshold
        self.tp_method = tp_method

        assert tp_method in ["iou", "distance"], f"Received: {tp_method}"

    def run(self, dataset: LabelingDataset, max_workers: int = 1) -> pd.DataFrame:
        logger.info("Computing evaluation metrics per image...")

        def iterator(df_pred: pd.DataFrame, df_gt: pd.DataFrame):
            image_paths = set(df_gt["file_name"]).union(set(df_pred["file_name"]))
            image_paths = list(image_paths)

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

        data = dataset.data.copy()
        data.sort_values("file_name", inplace=True)

        assert not data.empty, "The dataset is empty. Please check"

        data["x_center"] = (data["x_min"] + data["x_max"]) / 2
        data["y_center"] = (data["y_min"] + data["y_max"]) / 2

        # partition rows
        mask_pred = data["is_annot"] == False  # * (data['score']>=self.score_threshold)
        df_pred = (
            data.loc[mask_pred, :]
            .dropna(
                axis=0,
                subset=["is_annot", "label", "x_min", "y_min", "x_max", "y_max"],
                how="any",
            )
            .drop(columns="is_annot")
        )

        # positive samples
        df_gt = (
            data.loc[data["is_annot"] == True, :]
            .dropna(
                axis=0,
                subset=["is_annot", "label", "x_min", "y_min", "x_max", "y_max"],
                how="any",
            )
            .drop(columns="is_annot")
        )

        df_results = []

        def func(x):
            return self._run_per_image(df_gt_i=x[0], df_pred_i=x[1])

        loader = iterator(df_pred=df_pred, df_gt=df_gt)

        # with ThreadPool(max_workers) as executor:
        #     for df_eval in tqdm(executor.map(func, loader), desc="computing..."):
        #         if not df_eval.empty:
        #             df_results.append(df_eval)

        for df_eval in tqdm(map(func, loader), desc="computing..."):
            if not df_eval.empty:
                df_results.append(df_eval)

        if len(df_results) > 0:
            df_results = pd.concat(
                df_results, ignore_index=True, sort=False
            ).reset_index(drop=True)
        else:
            df_results = pd.DataFrame(columns=data.columns)
            logger.info("There are 0 positives samples and 0 Detections were found.")

        # adding True Negatives
        missing = set(data.file_name.to_list()) - set(df_results.file_name.to_list())
        missing = list(missing)
        df_results["pred_TN"] = 0
        for path in missing:
            i = len(df_results)  # add a row
            df_results.at[i, "file_name"] = path
            df_results.at[i, "pred_TN"] = 1
            df_results.at[i, "pred_TP"] = 0
            df_results.at[i, "pred_FP"] = 0
            df_results.at[i, "map50"] = np.nan
            df_results.at[i, "map75"] = np.nan

        df_results[["pred_TP", "gt_FN", "pred_FP", "pred_TN"]] = df_results[
            ["pred_TP", "gt_FN", "pred_FP", "pred_TN"]
        ].fillna(0)

        # every row is a pred or gt
        check = (
            df_results[["pred_TP", "pred_FP", "pred_TN", "gt_FN"]].sum(axis=1) != 1
        ).sum()
        assert check == 0, (
            "Error happened in the confusion matrix compution. Please check code."
        )

        # TP + FN = Total Ground Truth
        assert df_results[["gt_FN", "pred_TP"]].values.sum() == len(df_gt)

        # TP + FP = Total Predictions
        assert df_results[["pred_FP", "pred_TP"]].values.sum() == len(df_pred)

        return df_results

    def add_distance_to_closest(
        self, df_pred: pd.DataFrame, df_gt: pd.DataFrame
    ) -> pd.DataFrame:
        if df_gt.empty:
            df_pred["dist_closest_gt"] = np.inf
            df_pred["matched_gt_idx"] = -1
            return df_pred

        if df_pred.empty:
            return df_pred

        if "x_center" not in df_pred.columns or "x_center" not in df_gt.columns:
            df_pred["x_center"] = (df_pred["x_min"] + df_pred["x_max"]) / 2
            df_pred["y_center"] = (df_pred["y_min"] + df_pred["y_max"]) / 2
            df_gt["x_center"] = (df_gt["x_min"] + df_gt["x_max"]) / 2
            df_gt["y_center"] = (df_gt["y_min"] + df_gt["y_max"]) / 2

        distances = distance_matrix(
            df_pred[["x_center", "y_center"]].values,
            df_gt[["x_center", "y_center"]].values,
        )

        # Initialize arrays
        df_pred["dist_closest_gt"] = np.inf  # distances.min(1)
        df_pred["matched_gt_idx"] = -1

        # Perform Hungarian algorithm for optimal one-to-one assignment
        # Handle case where number of predictions != number of ground truths
        if len(df_pred) <= len(df_gt):
            # More or equal ground truths than predictions
            pred_indices, gt_indices = linear_sum_assignment(distances)

            # Assign matched distances and indices
            for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                df_pred.iloc[pred_idx, df_pred.columns.get_loc("dist_closest_gt")] = (
                    distances[pred_idx, gt_idx]
                )
                df_pred.iloc[pred_idx, df_pred.columns.get_loc("matched_gt_idx")] = (
                    gt_idx
                )

        else:
            # More predictions than ground truths
            # Transpose the distance matrix and solve
            gt_indices, pred_indices = linear_sum_assignment(distances.T)

            # Assign matched distances and indices for matched predictions
            for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                df_pred.iloc[pred_idx, df_pred.columns.get_loc("dist_closest_gt")] = (
                    distances[pred_idx, gt_idx]
                )
                df_pred.iloc[pred_idx, df_pred.columns.get_loc("matched_gt_idx")] = (
                    gt_idx
                )

        return df_pred

    def _get_bbox(self, gt: pd.DataFrame):
        return gt[self.bbox_cols].to_numpy().astype(float)

    def compute_map_ciou(self, df_pred: pd.DataFrame, df_gt: pd.DataFrame):
        # gt
        gt = torch.from_numpy(self._get_bbox(gt=df_gt).clip(min=0))
        labels = df_gt.loc[:, "label"].to_numpy().astype(int)

        # pred
        pred = self._get_bbox(gt=df_pred).clip(
            min=0,
        )
        pred = torch.from_numpy(pred)
        pred_score = df_pred.loc[:, "score"].to_numpy()
        classes = df_pred.loc[:, "label"].to_numpy().astype(int)

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
            max_scores=np.nan,
        )
        if len(pred_score) > 0:
            stats["stats"] = max(stats["all_scores"])

        return box_ious, stats

    def compute_tp_fp_iou(
        self, df_pred: pd.DataFrame, df_gt: pd.DataFrame, box_ious: torch.Tensor
    ):
        ## get FPs
        if df_gt.empty:
            df_pred["TP"] = 0
            df_pred["FP"] = len(df_pred)

            # rename columns
            df_eval = df_pred.rename(
                columns={
                    col: f"pred_{col}" for col in df_pred.columns if col != "file_name"
                },
            )
            df_eval["gt_FN"] = 0

            return df_eval

        # For each prediction: find best-matching GT
        best_iou, best_gt_idx = box_ious.max(dim=1)
        # df_pred["matching_gt"] = "None"
        # df_pred["matching_gt"] = df_pred["matching_gt"].astype("object")
        # df_pred["pred_label"] = "None"

        for i in range(len(df_pred)):
            df_pred.loc[i, "TP"] = (best_iou[i].item() >= self.tp_iou_threshold) * 1
            df_pred.loc[i, "FP"] = (best_iou[i].item() <= 0.0) * 1  # never matched
            df_pred.loc[i, "best_ciou"] = best_iou[i].item()
            # df_pred.loc[i, "matching_gt"] = (
            #     json.dumps(gt[best_gt_idx[i]].numpy().tolist())
            #     if df_pred.loc[i, "TP"]
            #     else "None"
            # )

        # get FN
        worst_pred_iou, _ = box_ious.max(dim=0)
        for i in range(len(df_gt)):
            df_gt.loc[i, "FN"] = (worst_pred_iou[i].item() < self.tp_iou_threshold) * 1
        df_gt.rename(
            columns={col: f"gt_{col}" for col in df_gt.columns if col != "file_name"},
            inplace=True,
        )
        gt_FN = df_gt.loc[df_gt["gt_FN"] > 0, :]

        # rename columns
        df_eval = df_pred.rename(
            columns={
                col: f"pred_{col}" for col in df_pred.columns if col != "file_name"
            },
        )

        df_eval = pd.concat([df_eval, gt_FN]).reset_index(drop=True)

        return df_eval

    def compute_tp_fp_distance(self, df_pred: pd.DataFrame, df_gt: pd.DataFrame):
        df_pred = self.add_distance_to_closest(df_pred=df_pred, df_gt=df_gt)

        if df_pred.empty:
            # df_pred.loc[0,:] = np.nan
            df_eval = df_pred.rename(
                columns={
                    col: f"pred_{col}" for col in df_pred.columns if col != "file_name"
                },
            )
            df_gt["gt_FN"] = 1
            df_eval = pd.concat([df_eval, df_gt]).reset_index(drop=True)

        else:
            df_pred["TP"] = (
                df_pred["dist_closest_gt"] <= self.tp_distance_threshold
            ) * 1

            if df_gt.empty:
                df_pred["FP"] = 1
            else:
                df_pred["FP"] = (
                    df_pred["dist_closest_gt"] > self.tp_distance_threshold
                ) * 1

            # get FN
            mask_fn = (df_pred["dist_closest_gt"] > self.tp_distance_threshold) * (
                np.isinf(df_pred["dist_closest_gt"]) == False
            )
            unmatched_gt_indx = df_pred.loc[mask_fn, "matched_gt_idx"]
            unmatched_gt_indx = list(unmatched_gt_indx)
            df_gt["gt_FN"] = 0
            df_gt.iloc[unmatched_gt_indx, df_gt.columns.get_loc("gt_FN")] = 1
            df_gt_unmatched = df_gt.iloc[unmatched_gt_indx, :].copy()

            df_eval = df_pred.rename(
                columns={
                    col: f"pred_{col}" for col in df_pred.columns if col != "file_name"
                },
            )

            if not df_gt_unmatched.empty:
                df_eval = pd.concat([df_eval, df_gt_unmatched]).reset_index(drop=True)
            else:
                df_eval["gt_FN"] = 0

        return df_eval

    def compute_tp_fp_maps(
        self, df_pred: pd.DataFrame, df_gt: pd.DataFrame, method: str = "iou"
    ):
        box_ious, stats = self.compute_map_ciou(df_pred=df_pred, df_gt=df_gt)

        if method == "iou":
            df_eval = self.compute_tp_fp_iou(
                df_pred=df_pred, df_gt=df_gt, box_ious=box_ious
            )

        elif method == "distance":
            df_eval = self.compute_tp_fp_distance(
                df_pred=df_pred,
                df_gt=df_gt,
            )

        else:
            raise NotImplementedError

        if df_eval.empty:
            return df_eval

        mask_pred = ~df_eval["pred_x_min"].isna()
        for k, v in stats.items():
            df_eval.loc[mask_pred, k] = v

        return df_eval

    def _run_per_image(
        self, df_gt_i: pd.DataFrame, df_pred_i: pd.DataFrame
    ) -> pd.DataFrame:
        # check validity
        unique_images_gt = set(df_gt_i["file_name"])
        unique_images_pred = set(df_pred_i["file_name"])
        assert len(unique_images_gt) <= 1 and len(unique_images_pred) <= 1, (
            "df_gt_i or df_pred_i has data for more than one image. Not Allowed!"
        )
        if not df_pred_i.empty:
            assert unique_images_gt.issubset(unique_images_pred), (
                "groundtruth image are does not match prediction image"
            )

        df_eval = self.compute_tp_fp_maps(
            df_pred=df_pred_i, df_gt=df_gt_i, method=self.tp_method
        )
        return df_eval


# =====================
# Performance Evaluation
# =====================
class PerformanceEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def run(
        self,
        dataset: LabelingDataset,
        pred_results_dir: str,
        load_results: bool = False,
        save_tag: str = "",
    ) -> pd.DataFrame | None:
        """Calculate performance metrics"""

        metrics = Metrics(
            tp_method=self.config.tp_method,
            score_threshold=self.config.score_threshold,
            tp_iou_threshold=self.config.tp_iou_threshold,
            tp_distance_threshold=self.config.tp_distance_threshold,
        )

        # when providing a list of images
        stem = (
            f"predictions-{save_tag}" + save_tag if len(save_tag) > 0 else "predictions"
        )
        save_path = os.path.join(pred_results_dir, stem + ".csv")

        # get prediction results
        if load_results:
            try:
                dataset.import_data(save_path)
            except:
                traceback.print_exc()
                raise ValueError()
        else:
            dataset.save_data_csv(save_path=save_path)

        # compute metrics per image
        df_results = metrics.run(dataset)

        # if results_per_image:
        return df_results, self._get_per_image_metrics(df_results)

    def _get_per_image_metrics(self, df_results: pd.DataFrame):
        results = dict()

        conf_matrix = ["pred_TP", "pred_FP", "pred_TN", "gt_FN"]
        maps = ["map75", "map50"]

        for filename, df in df_results.groupby("file_name"):
            metrics = dict()
            metrics["max_scores"] = df["pred_score"].dropna().max()
            if "pred_best_ciou" in df.columns:
                metrics["mean_ciou"] = df["pred_best_ciou"].dropna().mean()

            if "distance_closest_gt" in df.columns:
                metrics["mean_dist_closest_gt"] = (
                    df["distance_closest_gt"].dropna().mean()
                )

            metrics["mean_pred_area"] = df["pred_area"].dropna().mean()
            for col in conf_matrix:
                metrics[col] = df[col].dropna().sum()
            for col in maps:
                metrics[col] = df[col].dropna().mean()

            results[filename] = metrics

        results = pd.DataFrame.from_dict(results, orient="index")

        return results


class Calibrator:
    def __init__(
        self,
        pred_results_dir: str,
        batch_size: int = 8,
        inference_service_url: str | None = None,  # "http://localhost:4141/predict",
        feature_extractor_path: str | None = "facebook/dinov2-with-registers-small",
        roi_weights: str = r"..\base_models_weights\roi_classifier.ckpt",
        detection_label_map: dict = {0: "wildlife"},
        roi_cls_label_map: dict = {0: "gt", 1: "tn"},
        roi_keep_classes: list = ["gt"],
        roi_cls_is_features: bool = True,
        detection_model: Detector = None,
        mlflow_model_alias: str = "demo",
        mlflow_model_name: str = "labeler",
    ):
        self.dataset: LabelingDataset = None

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.pred_results_dir = pred_results_dir
        self.inference_service_url = inference_service_url
        self.detection_label_map = detection_label_map
        self.roi_cls_label_map = roi_cls_label_map
        self.roi_keep_classes = roi_keep_classes
        self.roi_cls_is_features = roi_cls_is_features
        self.detection_model = detection_model
        self.mlflow_model_alias = mlflow_model_alias
        self.mlflow_model_name = mlflow_model_name
        self.roi_weights = roi_weights
        self.feature_extractor_path = feature_extractor_path
        self.batch_size = batch_size

        self.count = 0

        # assert not (inference_service_url is None and detection_model is None)

    # @lru_cache(maxsize=None)
    def _add_predictions(
        self,
        imgsz: int,
        tilesize: int,
        overlap_ratio: float,
        nms_iou: float,
        cls_imgsz: int,
        flight_height: int = 180,
        sensor_height: int = 24,
    ):
        config = PredictionConfig(
            imgsz=imgsz,
            tilesize=tilesize,
            batch_size=self.batch_size,
            overlap_ratio=overlap_ratio,
            confidence_threshold=0.1,
            inference_service_url=self.inference_service_url,
            flight_height=flight_height,
            sensor_height=sensor_height,
            gsd=None,
            nms_iou=nms_iou,
            verbose=False,
            min_area=None,
            max_area=None,
            cls_imgsz=cls_imgsz,
            device=self.device,
        )

        engine, _ = InferenceEngine.load_engine(
            pred_config=config,
            roi_classifier_path=self.roi_weights,
            roi_cls_is_features=self.roi_cls_is_features,
            roi_cls_label_map=self.roi_cls_label_map,
            roi_keep_classes=self.roi_keep_classes,
            detection_label_map=self.detection_label_map,
            feature_extractor_path=self.feature_extractor_path,
            detection_model=self.detection_model,
            mlflow_model_alias=self.mlflow_model_alias,
            mlflow_model_name=self.mlflow_model_name,
        )
        self.dataset.add_predictions(engine=engine, build=True)
        return None

    def _run_once(self, params: dict, save_tag: str = ""):
        eval_config = EvaluationConfig(**params)

        perf_eval = PerformanceEvaluator(eval_config)

        df_results, df_metrics_per_img = perf_eval.run(
            dataset=self.dataset,
            pred_results_dir=self.pred_results_dir,
            load_results=False,
            save_tag=save_tag,
        )

        # report generation
        reporter = ReportGenerator()
        stats = reporter.run(df_results, plot=False, save_plot=None)
        # if np.isnan(stats["map50"]).all():
        #     map50 = 1.0
        # else:
        #     map50 = np.nansum(stats["map50"])

        # if np.isnan(stats["map75"]).all():
        #     map75 = 1.0
        # else:
        #     map75 = np.nansum(stats["map75"])

        # results = [stats["FP"]] + [stats[k] for k in ["TP", "TN"]] + [map50, map75]

        return stats["F1"]  # results

    def __call__(self, trial):
        hyperparameters = dict(
            score_threshold=np.linspace(0.1, 0.7, 5).round(3).tolist(),
            # min_area=(np.arange(5,25,5)**2).tolist(),
            # max_area=(np.arange(25,100,5)**2).tolist(),
        )

        tp_method = trial.suggest_categorical("tp_method", ["iou", "distance"])
        if tp_method == "iou":
            hyperparameters["tp_iou_threshold"] = (
                np.linspace(0.2, 0.7, 5).round(3).tolist()
            )
        else:
            hyperparameters["tp_distance_threshold"] = (np.arange(1, 10) * 20).tolist()

        params = {
            k: trial.suggest_categorical(f"{k}", v) for k, v in hyperparameters.items()
        }

        scores = self._run_once(params, save_tag=f"trial-{self.count}")

        self.count += 1

        return scores

    def run(
        self,
        dataset: LabelingDataset,
        tilesize=800,
        imgsz=800,
        cls_imgsz=98,
        overlap_ratio=0.2,
        n_trials=20,
        study_name="demo-muti",
        mlflow_exp_name="calibrating",
        storage="sqlite:///hypsearch.sql",
    ):
        import optuna
        from optuna.samplers import TPESampler
        from optuna.integration.mlflow import MLflowCallback
        import mlflow

        self.dataset = dataset

        mlflow.set_tracking_uri(uri="http://localhost:5000")

        mlflow_metric_name = "fitness"

        try:
            exp_id = mlflow.get_experiment_by_name(mlflow_exp_name).experiment_id
        except:
            exp_id = mlflow.create_experiment(name=mlflow_exp_name)

        mlflow.set_experiment(experiment_id=exp_id)
        mlflc = MLflowCallback(
            metric_name=mlflow_metric_name,
            create_experiment=False,
        )

        opt_direction = dict()
        opt_direction["direction"] = "maximize"

        # compute detections
        config = dict(
            overlap_ratio=overlap_ratio,
            tilesize=tilesize,
            imgsz=imgsz,
            cls_imgsz=cls_imgsz,
        )

        for nms_iou in np.linspace(0.1, 0.7, 5):
            config["nms_iou"] = nms_iou
            msg = "Running with :\n" + json.dumps(config, indent=2)
            logger.info(msg)
            self._add_predictions(**config)

            study = optuna.create_study(
                sampler=TPESampler(multivariate=True, group=True),
                study_name=study_name,
                pruner=optuna.pruners.HyperbandPruner(),
                load_if_exists=True,
                storage=storage,
                **opt_direction,
            )

            study.optimize(
                self,
                n_trials=n_trials,
                n_jobs=1,
                show_progress_bar=True,
                timeout=60 * 60 * 3,
                callbacks=[mlflc],
            )

        return


# =====================
# Reporting Interface
# =====================
class ReportGenerator:
    def __init__(
        self,
    ):
        pass

    def get_stats(self, df_results: pd.DataFrame):
        tp_fp_tn = (
            df_results
            # .dropna()
            .groupby("file_name")[
                [
                    "pred_TP",
                    "pred_FP",
                    # "pred_TN",
                    "gt_FN",
                ]
            ]
            .sum()
            .rename(
                columns={
                    "pred_TP": "TP",
                    "pred_FP": "FP",
                    # "pred_TN": "TN",
                    "gt_FN": "FN",
                }
            )
        )
        tp_fp_tn_sum = tp_fp_tn.sum().to_dict()
        map50 = df_results["map50"].to_numpy()
        map75 = df_results["map75"].to_numpy()

        stats = dict(
            map50=map50,
            map75=map75,
        )
        stats.update(tp_fp_tn_sum)

        # precision
        p = 0
        if (stats["TP"] + stats["FP"]) > 0:
            p = stats["TP"] / (stats["TP"] + stats["FP"])

        # recall
        r = 0
        if (stats["TP"] + stats["FN"]) > 0:
            r = stats["TP"] / (stats["TP"] + stats["FN"])

        stats["precision"] = p
        stats["recall"] = r

        stats["F1"] = 0
        if (p + r) > 0:
            stats["F1"] = 2 * p * r / (p + r)

        return stats, tp_fp_tn

    def plot(self, df_results, tp_fp_tn, save_plot: str = None):
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))

        axs = axs.ravel()

        sns.barplot(data=tp_fp_tn, ax=axs[0], estimator=np.sum, errorbar=None)

        sns.histplot(
            data=df_results.loc[df_results["map50"] > -1, "map50"],
            ax=axs[1],
            stat="count",
        )

        sns.boxplot(data=df_results["pred_area"], orient="h", ax=axs[2])

        sns.histplot(data=df_results["pred_score"], ax=axs[3], stat="count")

        if save_plot:
            fig.savefig(
                save_plot,
                dpi=300,  # Resolution
                facecolor="white",  # Background color
                edgecolor="none",  # Border color
                format="png",
            )
        pass

    def run(
        self,
        df_results: pd.DataFrame,
        plot: bool = False,
        save_plot: str = None,
    ) -> None:
        """Generate comprehensive performance report"""

        stats, tp_fp_tn = self.get_stats(df_results)

        if plot:
            self.plot(df_results=df_results, tp_fp_tn=tp_fp_tn, save_plot=save_plot)

        return stats


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

        self.eval_config = eval_config

    # TODO
    def run(
        self,
        engine: InferenceEngine,
        dataset: LabelingDataset,
        pred_results_dir: str = None,
        save_tag: str = "",
        load_results: bool = False,
    ) -> None:
        """Complete evaluation workflow"""

        if not load_results:
            dataset.add_predictions(engine=engine, build=True)

        # Load data
        df_results, df_metrics_per_img = self.evaluator.run(
            dataset=dataset,
            tp_method="distance",
            tp_distance_threshold=50,
            tp_iou_threshold=self.eval_config.tp_iou_threshold,
            pred_results_dir=pred_results_dir,
            load_results=load_results,
        )

        # mining hard sampels
        df_hard_negatives = self.sample_selector.run(df_results)

        # report generation
        reporter = ReportGenerator()
        stats, fig = reporter.run(df_results, plot=False)

        return df_metrics_per_image, df_hard_negatives
