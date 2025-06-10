from datalabeling.common.evaluation import (
    PerformanceEvaluator,
    HardSampleSelector,
    ReportGenerator,
    Calibrator,
)
from datalabeling.common.config import EvaluationConfig, PredictionConfig
from datalabeling.ml.interface import InferenceEngine
from datalabeling.common.dataset_loader import LabelingDataset
from ultralytics import YOLO
from pathlib import Path


def run_perf_evaluator():
    eval_config = EvaluationConfig()
    eval_config.score_threshold = 0.25
    eval_config.map_threshold = 0.3
    eval_config.uncertainty_method = "entropy"
    eval_config.uncertainty_threshold = 4
    eval_config.score_col = "max_scores"
    eval_config.tp_iou_threshold = 0.5

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

    engine, _ = InferenceEngine.load_engine(
        pred_config=config,
        roi_classifier_path=None,  # r"..\base_models_weights\roi_classifier.ckpt",
        roi_cls_is_features=True,
        roi_cls_label_map={0: "gt", 1: "tn"},
        roi_keep_classes=["gt"],
        detection_label_map={0: "wildlife"},
        feature_extractor_path="facebook/dinov2-with-registers-small",
        detection_model=YOLO(r"..\base_models_weights\best.pt"),
        mlflow_model_alias="demo",
        mlflow_model_name="labeler",
    )

    perf_eval = PerformanceEvaluator()

    images_dirs = [
        r"D:\workspace\data\savmap_dataset_v2\images_tmp",
    ]

    load_results = False

    # creating dataset and adding predictions
    dataset = LabelingDataset.from_dirs(images_dirs)
    if not load_results:
        dataset.add_predictions(engine=engine, build=True)

    # run evaluator
    df_metrics_per_img = perf_eval.run(
        dataset=dataset,
        tp_iou_threshold=eval_config.tp_iou_threshold,
        pred_results_dir=r"D:\workspace\data\savmap_dataset_v2\images_tmp",
        load_results=load_results,
    )

    # print("results: ", df_metrics_per_img)

    # mining hard sampels
    # sample_selector = HardSampleSelector(config=eval_config)
    # df_hard_negatives = sample_selector.run(df_metrics_per_img)

    # report generation
    reporter = ReportGenerator()
    stats, fig = reporter.run(df_metrics_per_img, plot=True)

    return df_metrics_per_img  # , df_hard_negatives


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

    # mlflow.set_tracking_uri(uri="http://localhost:5000")

    # n_trials = 20
    # mlflow_exp_name = "calibrating"
    # mlflow_metric_name = "fitness"
    # study_name = "demo"

    # try:
    #     exp_id = mlflow.get_experiment_by_name(mlflow_exp_name).experiment_id
    # except:
    #     exp_id = mlflow.create_experiment(name=mlflow_exp_name)

    # mlflow.set_experiment(experiment_id=exp_id)
    # mlflc = MLflowCallback(metric_name=mlflow_metric_name,
    #                        create_experiment=False,
    #                        )

    # opt_direction = dict()
    # # opt_direction['directions'] = ["maximize"]*3 # multi-objective
    # opt_direction['direction'] = "maximize"

    # study = optuna.create_study(
    #     # sampler=TPESampler(multivariate=True, group=True),
    #     study_name=study_name,
    #     pruner=optuna.pruners.HyperbandPruner(),
    #     load_if_exists=True,
    #     storage="sqlite:///hypsearch.sql",
    #     **opt_direction
    # )

    # study.optimize(
    #     objective,
    #     n_trials=n_trials,
    #     n_jobs=1,
    #     show_progress_bar=True,
    #     timeout=60 * 60 * 3,
    #     callbacks=[mlflc],
    # )

    return


if __name__ == "__main__":
    # df_metrics_per_img = run_perf_evaluator()

    calibration()

    pass
