from datalabeling.common.evaluation import PerformanceEvaluator, HardSampleSelector
from datalabeling.common.config import EvaluationConfig
from datalabeling.ml.models import Detector


def run_perf_evaluator():
    eval_config = EvaluationConfig()
    eval_config.score_threshold = 0.25
    eval_config.map_threshold = 0.3
    eval_config.uncertainty_method = "entropy"
    eval_config.uncertainty_threshold = 4
    eval_config.score_col = "max_scores"

    detector = Detector(
        path_to_weights=r"D:\datalabeling\base_models_weights\best.pt",
        confidence_threshold=0.25,
        overlap_ratio=0.2,
        tilesize=800,
        imgsz=800,
        use_sliding_window=True,
        device="cpu",
    )

    perf_eval = PerformanceEvaluator(config=eval_config)

    df_metrics_per_img = perf_eval.evaluate(
        images_dirs=[
            r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images",
        ],
        pred_results_dir=r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format",
        detector=detector,
        load_results=False,
    )

    # mining hard sampels
    sample_selector = HardSampleSelector(config=eval_config)

    df_hard_negatives = sample_selector.select_hard_samples(df_metrics_per_img)

    return df_metrics_per_img, df_hard_negatives


if __name__ == "__main__":
    pass

    # df_metrics_per_img, df_hard_negatives = run_perf_evaluator()
