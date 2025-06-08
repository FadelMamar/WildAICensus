from datalabeling.common.evaluation import PerformanceEvaluator, HardSampleSelector
from datalabeling.common.config import EvaluationConfig, PredictionConfig
from datalabeling.ml.interface import load_engine
from datalabeling.common.dataset_loader import LabelingDataset


def run_perf_evaluator():
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

    engine = load_engine(config=config)

    perf_eval = PerformanceEvaluator(config=eval_config)

    images_dirs = [
        r"D:\savmap_dataset_v2\images_tmp",
    ]

    load_results = True

    # creating dataset and adding predictions
    dataset = LabelingDataset.from_dirs(images_dirs)
    if not load_results:
        dataset.add_predictions(engine=engine)
        dataset.build(force_rebuild=True)

    # run evaluator
    df_metrics_per_img = perf_eval.run(
        dataset=dataset,
        pred_results_dir=r"D:\savmap_dataset_v2\images_tmp",
        load_results=load_results,
    )

    # print("results: ", df_metrics_per_img)

    # mining hard sampels
    sample_selector = HardSampleSelector(config=eval_config)

    df_hard_negatives = sample_selector.run(df_metrics_per_img)

    return df_metrics_per_img, df_hard_negatives


if __name__ == "__main__":
    df_metrics_per_img, df_hard_negatives = run_perf_evaluator()

    pass
