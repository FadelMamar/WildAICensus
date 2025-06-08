import fire
from datalabeling.common.evaluation import PerformanceEvaluator, HardSampleSelector
from datalabeling.common.config import EvaluationConfig, PredictionConfig
from datalabeling.ml.interface import load_engine
from datalabeling.common.dataset_loader import LabelingDataset


def run_perf_evaluator(
    images_dirs: list[str],
    load_existing_predictions: bool = False,
    save_path: str = None,
):
    eval_config = EvaluationConfig()
    eval_config.score_threshold = 0.25
    eval_config.map_threshold = 0.3
    eval_config.uncertainty_method = "entropy"
    eval_config.uncertainty_threshold = 4
    eval_config.score_col = "max_scores"

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

    engine, _ = load_engine(
        pred_config=config,
        roi_classifier_path=r"..\base_models_weights\roi_classifier.ckpt",
        roi_cls_is_features=True,
        roi_cls_label_map={0: "gt", 1: "tn"},
        roi_keep_classes=["gt"],
        detection_label_map=None,  # {0: "wildlife"},
        feature_extractor_path="facebook/dinov2-with-registers-small",
        detection_model=None,
        mlflow_model_alias="demo",
        mlflow_model_name="labeler",
    )

    perf_eval = PerformanceEvaluator(config=eval_config)

    # creating dataset and adding predictions
    dataset = LabelingDataset.from_dirs(images_dirs)
    if not load_existing_predictions:
        dataset.add_predictions(engine=engine)
        dataset.build(force_rebuild=True)

    # run evaluator
    df_metrics_per_img = perf_eval.run(
        dataset=dataset,
        pred_results_dir=r"D:\savmap_dataset_v2\images_tmp",
        load_results=load_existing_predictions,
    )

    # mining hard sampels
    sample_selector = HardSampleSelector(config=eval_config)

    df_hard_negatives = sample_selector.run(df_metrics_per_img)

    if save_path:
        df_hard_negatives.to_csv(save_path, index=False)

    return None


if __name__ == "__main__":
    fire.Fire()
