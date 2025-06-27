from datalabeling.common.evaluation import (
    PerformanceEvaluator,
    HardSampleSelector,
    ReportGenerator,
    Calibrator,
    Metrics,
)
from datalabeling.common.config import EvaluationConfig, PredictionConfig
from datalabeling.ml.interface import InferenceEngine
from datalabeling.ml.models import UltralyticsDetector, GroundingDinoDetector
from datalabeling.common.dataset_loader import LabelingDataset

from pathlib import Path
import pandas as pd


def run_metrics_computation():
    metric = Metrics(tp_iou_threshold=0.5)

    df_pred = pd.DataFrame(
        {
            "x_min": [10, 50],
            "x_max": [20, 60],
            "y_min": [10, 50],
            "y_max": [20, 60],
        }
    )

    df_gt = pd.DataFrame(
        {
            "x_min": [12, 100],
            "x_max": [22, 110],
            "y_min": [12, 100],
            "y_max": [22, 110],
        }
    )

    df_pred = metric.add_distance_to_closest(df_pred=df_pred, df_gt=df_gt)
    # print(df_match)

    return df_pred


def run_perf_evaluator(load_results=False):
    eval_config = EvaluationConfig()
    eval_config.score_threshold = 0.2
    eval_config.map_threshold = 0.3
    eval_config.uncertainty_method = "entropy"
    eval_config.uncertainty_threshold = 4
    eval_config.score_col = "max_scores"
    eval_config.tp_iou_threshold = 0.5
    eval_config.tp_method = "distance"
    eval_config.tp_distance_threshold = 100

    config = PredictionConfig(
        imgsz=800,
        tilesize=800,
        overlap_ratio=0.2,
        confidence_threshold=0.2,
        inference_service_url=None,
        # min_area=100,
        # max_area=None,
        cls_imgsz=128,
        # device="cuda:0",
    )

    label_map = {0: "wildlife"}

    detection_model = UltralyticsDetector(
        model_path="D:/datalabeling/base_models_weights/best.pt", config=config
    )

    engine, _ = InferenceEngine.load_engine(
        pred_config=config,
        roi_classifier_path=None,  # r"..\base_models_weights\roi_classifier.ckpt",
        roi_cls_is_features=True,
        roi_cls_label_map={0: "gt", 1: "tn"},
        roi_keep_classes=["gt"],
        detection_label_map=label_map,
        feature_extractor_path="facebook/dinov2-with-registers-small",
        detection_model=detection_model,
        mlflow_model_alias="demo",
        mlflow_model_name="labeler",
    )

    perf_eval = PerformanceEvaluator(eval_config)

    images_dirs = [
        r"D:\workspace\data\savmap_dataset_v2\annotated_py_paul\yolo_format\images",
    ]

    # creating dataset and adding predictions
    dataset = LabelingDataset.from_yolo(images_dirs, label_map=label_map)

    print(dataset.get_stats())

    if not load_results:
        dataset.add_predictions(engine=engine, build=True)

    # run evaluator
    df_results, df_metrics_per_img = perf_eval.run(
        dataset=dataset,
        pred_results_dir=r"D:\workspace\data\savmap_dataset_v2\images_tmp",
        load_results=load_results,
    )

    # print("results: ", df_metrics_per_img)

    # mining hard sampels
    sample_selector = HardSampleSelector(config=eval_config)
    df_hard_negatives = sample_selector.run(df_results)

    # report generation
    reporter = ReportGenerator()
    stats = reporter.run(df_results, plot=True, save_plot="report.png")

    return df_results, df_metrics_per_img, stats, df_hard_negatives


def calibration():
    # import optuna
    # from optuna.samplers import TPESampler
    # from optuna.integration.mlflow import MLflowCallback
    # import mlflow
    from ultralytics import YOLO

    images_dirs = [
        r"D:\workspace\data\savmap_dataset_v2\images_tmp",
    ]

    save_dir = ".tmp/calibration"

    Path(save_dir).mkdir(exist_ok=True, parents=True)

    dataset = LabelingDataset.from_dirs(images_dirs)

    calibrator = Calibrator(
        pred_results_dir=save_dir,
        inference_service_url="http://localhost:4141/predict",
        # detection_model=YOLO("../base_models_weights/best.pt"),
        feature_extractor_path="facebook/dinov2-with-registers-small",
        roi_weights=r"..\base_models_weights\roi_classifier.ckpt",
        detection_label_map={0: "wildlife"},
        roi_cls_label_map={0: "gt", 1: "tn"},
        roi_keep_classes=["gt"],
        roi_cls_is_features=True,
        mlflow_model_alias="demo",
        mlflow_model_name="labeler",
    )

    calibrator.run(dataset=dataset)

    return


if __name__ == "__main__":
    df_results, df_metrics_per_img, stats, df_hard_negatives = run_perf_evaluator(True)

    # calibration()

    # df_pred = run_metrics_computation()

    pass
