from datalabeling.common.evaluation import PerformanceEvaluator, HardSampleSelector
from datalabeling.common.config import EvaluationConfig, PredictionConfig
from datalabeling.ml.models import Detector, ImageClassifier
from datalabeling.ml.interface import InferenceEngine
from datalabeling.common.dataset_loader import LabelingDataset
from datalabeling.common.processor import get_processor, DetectionsPostprocessor


def load_engine(config: PredictionConfig):
    # get image classifier
    path = r"..\base_models_weights\roi_classifier.ckpt"
    model = ImageClassifier.load_from_checkpoint(
        path, cls_is_features=True, map_location=config.device
    )
    handler = get_processor("classifier")(
        model,
        label_map={0: "gt", 1: "tn"},
        device=config.device,
        feature_extractor=get_processor("feature_extractor")(),
        imgsz=config.cls_imgsz,
    )

    # build postprocessor
    processor = DetectionsPostprocessor(
        keep_classes=["gt"],
    )
    processor.set_handler(handler)

    # load detector
    detector = Detector(config=config, detection_model=None)
    detector.set_detection_model(
        detection_model=None,
        path_to_weights=r"D:\datalabeling\base_models_weights\best.pt",
        yolo_model=None,
    )

    # load engine
    engine = InferenceEngine(config=config)
    engine.set_detector(detector, model_tag="demo")
    engine.set_processor(image_processor=None, detection_processor=processor)

    return engine


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
