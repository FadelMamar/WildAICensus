import fire
import os

import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv
from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio
from ultralytics import YOLO
from datalabeling.common.config import DataConfig, LabelConfig
from datalabeling.common.pipeline import (
    LabelstudioToYolo,
    ObbToDotaStep,
    Pipeline,
    YoloToObbStep,
    YoloToSegStep,
    ObbToYoloStep,
    YoloToCocoStep,
)
from datalabeling.common.io import load_datasets
from datalabeling.ml.interface import InferenceEngine
from datalabeling.common.config import TrainingConfig
from datalabeling.ml.train import TrainingManager
from datalabeling.common.config import EvaluationConfig, PredictionConfig
from datalabeling.common.io import load_yaml
from datalabeling.common.dataset_loader import (
    ClassificationDatasetBuilder,
)


logger = logging.getLogger(__name__)


# TODO debug
def create_classification_data(
    yaml_path: str,
    save_dir: str,
    strategies: list[str] = ["gt", "hn", "fp"],
    alias="demo",
    detection_model_path: str = None,
    roi_classifier_path: str = None,
    roi_cls_label_map: dict = {0: "gt", 1: "tn"},
):
    eval_config = EvaluationConfig()
    eval_config.score_threshold = 0.25
    eval_config.map_threshold = 0.3
    eval_config.uncertainty_method = "entropy"
    eval_config.uncertainty_threshold = 4
    eval_config.score_col = "max_scores"
    eval_config.tp_iou_threshold = 0.2
    eval_config.load_results = (
        False  # Set to True to load existing predictions if applicable
    )

    pred_config = PredictionConfig(
        imgsz=800,
        tilesize=800,
        overlap_ratio=0.2,
        confidence_threshold=0.2,
        batch_size=8,
        # min_area=100,
        # max_area=None,
        cls_imgsz=98,
        # device="cpu",
    )

    handler = ClassificationDatasetBuilder(
        eval_config,
    )

    cfg = load_yaml(yaml_path)

    model = None
    if detection_model_path:
        model = YOLO(detection_model_path)

    engine, feature_extractor = InferenceEngine.load_engine(
        pred_config,
        roi_classifier_path=roi_classifier_path,
        detection_model=model,
        roi_cls_is_features=True,
        roi_cls_label_map=roi_cls_label_map,
        mlflow_model_alias=alias,
    )

    for split in ["train", "val"]:
        source_dirs = [os.path.join(cfg["path"], subset) for subset in cfg[split]]

        logger.info(f"source_dirs: {source_dirs}")

        handler.set_dirs(
            source_dirs=source_dirs, output_dir=os.path.join(save_dir, split)
        )

        handler.run(
            strategies=strategies,
            save_true_negatives=True,
            feature_extractor=feature_extractor,
            detector=engine,
            bbox_resize_factor=1,
            tn_kwargs=dict(w=pred_config.cls_imgsz, h=pred_config.cls_imgsz, number=3),
            tp_kwargs=dict(w=pred_config.cls_imgsz, h=pred_config.cls_imgsz),
            hn_kwargs=dict(w=pred_config.cls_imgsz, h=pred_config.cls_imgsz),
        )


# TODO debug
def train(args: TrainingConfig):
    TrainingManager(
        args=args,
        herdnet_loss=None,
    ).run()


# TODO debug
def convert_dataset(
    data_config_yaml: str,
    yolo_to_obb: bool,
    obb_to_dota: bool,
    obb_to_yolo: bool,
    yolo_to_coco: bool,
    yolo_to_seg: bool,
    coco_dir: str = None,
    sam_weights: str = "sam2.1_l.pt",
    clear: bool = True,
    skip: bool = True,
):
    if yolo_to_coco:
        obb_to_yolo = True

    assert (yolo_to_obb + obb_to_yolo) < 2, "Both arguments can't be True"

    directories = load_datasets(data_config_yaml)
    steps = []
    data_config = load_yaml(data_config_yaml)

    for p in directories:
        try:
            p_new = p.replace("images", "labels")

            if yolo_to_obb or obb_to_dota:
                steps.append(
                    YoloToObbStep(
                        yolo_labels_dir=p_new,
                        obb_labels_dir=p_new,
                        skip=skip,
                    )
                )

            if obb_to_yolo:
                steps.append(
                    ObbToYoloStep(
                        obb_labels_dir=p_new,
                        yolo_labels_dir=p_new,
                        skip=skip,
                    )
                )

            if obb_to_dota:
                labels_output_dir = Path(p_new).parent / "dota_labels"
                steps.append(
                    ObbToDotaStep(
                        obb_img_dir=p_new,
                        dota_dir=labels_output_dir,
                        label_map=data_config["names"],
                        skip=skip,
                        clear_old=clear,
                    ),
                )

            pipeline = Pipeline(steps)
            pipeline.run()

        except Exception:
            # logger.warning(f"Failed for {p_new}")
            traceback.print_exc()

    if yolo_to_coco:
        assert coco_dir is not None
        steps = []
        for split in ["train", "val", "test"]:
            try:
                steps.append(
                    YoloToCocoStep(
                        coco_dir=coco_dir,
                        split=split,
                        data_config=data_config,
                        clear=clear,
                    )
                )
            except:
                traceback.print_exc()
                print(f"Skipping split={split}")

        pipeline = Pipeline(steps)
        pipeline.run()

    if yolo_to_seg:
        from ultralytics import SAM
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        steps = [
            YoloToSegStep(
                data_config_yaml=data_config_yaml,
                model_sam=SAM(sam_weights),
                device=device,
                copy_images_dir=True,
                clear=clear,
            )
        ]
        pipeline = Pipeline(steps)
        pipeline.run()

    return None


# TODO debug
def create_yolo_dataset(
    image_dir: str,
    is_single_cls: bool,
    yolo_data_config_yaml: str,
    dest_path_images: str,
    dest_path_labels: str,
    clear_output: bool,
    coco_json_dir: str,
    ls_json_dir: str,
    parse_ls_config: bool,
    load_existing_coco_annotations: bool,
    tilesize: int,
    min_area_ratio: float = 0.8,
    empty_ratio: float = 1.0,
    dotenv_path: str = "../.env",
    save_all: bool = False,
    save_only_empty: bool = False,
    label_map: str = r"D:\datalabeling\exported_annotations\label_mapping.json",
    ls_xml_config: str = r"D:\datalabeling\exported_annotations\label_studio_config.xml",
    labels_to_keep=("wildlife",),
    labels_to_discard=("other",),
):
    ## ---- Creating yolo dataset from Label studio labels
    dataset_config = DataConfig()
    dataset_config.dest_path_images = dest_path_images
    dataset_config.dest_path_labels = dest_path_labels

    dataset_config.clear_output = clear_output

    dataset_config.coco_json_dir = coco_json_dir
    dataset_config.ls_json_dir = ls_json_dir
    dataset_config.parse_ls_config = parse_ls_config
    dataset_config.load_coco_annotations = load_existing_coco_annotations

    dataset_config.tilesize = tilesize
    dataset_config.min_area_ratio = min_area_ratio
    dataset_config.empty_ratio = empty_ratio

    dataset_config.is_single_cls = is_single_cls
    dataset_config.yolo_data_config_yaml = yolo_data_config_yaml

    dataset_config.dotenv_path = dotenv_path

    dataset_config.save_all = save_all
    dataset_config.save_only_empty = save_only_empty

    label_config = LabelConfig()
    label_config.label_map = label_map
    label_config.keep = labels_to_keep
    label_config.discard = labels_to_discard

    load_dotenv(dotenv_path=dotenv_path)

    # # label studio client
    LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
    API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

    if LABEL_STUDIO_URL is None:
        raise ValueError("env variable LABEL_STUDIO_URL is not set.")
    if API_KEY is None:
        raise ValueError("env variable API_KEY is not set.")

    ls_client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

    steps = [
        LabelstudioToYolo(
            dataset_config=dataset_config,
            label_config=label_config,
            ls_xml_config=ls_xml_config,
            ls_client=ls_client,
            image_dir=image_dir,
        ),
    ]

    pipeline = Pipeline(steps)
    pipeline.run()


def calibrate_model(
    yolo_images_dirs: list[str],
    label_map={0: "wildlife"},
    roi_cls_label_map={0: "gt", 1: "tn"},
    tilesize=640,
    imgsz=640,
    cls_imgsz=98,
    overlap_ratio=0.2,
    roi_keep_classes=["gt"],
    roi_cls_is_features=True,
    feature_extractor_path="facebook/dinov2-with-registers-small",
    roi_classifier_path=r"..\base_models_weights\roi_classifier.ckpt",
    model_path=r"D:/datalabeling/base_models_weights/best.pt",
    detection_model_type="ultralytics",
    mlflow_model_alias="demo",
    mlflow_model_name="labeler",
    save_dir=".tmp/calibration",
):
    from datalabeling.common.evaluation import Calibrator
    from datalabeling.common.dataset_loader import LabelingDataset

    # creating dataset and adding predictions
    dataset = LabelingDataset.from_yolo(yolo_images_dirs, label_map=label_map)

    Path(save_dir).mkdir(exist_ok=True, parents=True)

    calibrator = Calibrator(
        pred_results_dir=save_dir,
        inference_service_url=None,  # "http://localhost:4141/predict",
        feature_extractor_path=feature_extractor_path,
        roi_classifier_path=roi_classifier_path,
        detection_label_map=label_map,
        roi_cls_label_map=roi_cls_label_map,
        roi_keep_classes=roi_keep_classes,
        roi_cls_is_features=roi_cls_is_features,
        model_path=model_path,
        detection_model_type=detection_model_type,
        mlflow_model_alias=mlflow_model_alias,
        mlflow_model_name=mlflow_model_name,
    )

    calibrator.run(
        dataset=dataset,
        tilesize=tilesize,
        imgsz=imgsz,
        cls_imgsz=cls_imgsz,
        overlap_ratio=overlap_ratio,
    )


def evaluate(
    yolo_images_dirs: list[str],
    pred_results_dir: str = ".tmp",
    save_plot: str = "report.png",
    save_hn: str = "hard_negative.csv",
    label_map={0: "wildlife"},
    roi_cls_label_map={0: "gt", 1: "tn"},
    tilesize=640,
    imgsz=640,
    cls_imgsz=98,
    overlap_ratio=0.2,
    confidence_threshold=0.2,
    roi_keep_classes=["gt"],
    roi_cls_is_features=True,
    feature_extractor_path="facebook/dinov2-with-registers-small",
    roi_classifier_path=r"..\base_models_weights\roi_classifier.ckpt",
    model_path=r"D:/datalabeling/base_models_weights/best.pt",
    detection_model_type="ultralytics",
    mlflow_model_alias="demo",
    mlflow_model_name="labeler",
    save_dir=".tmp/calibration",
    load_results=False,
):
    from datalabeling.common.evaluation import (
        PerformanceEvaluator,
        HardSampleSelector,
        ReportGenerator,
    )
    from datalabeling.common.config import EvaluationConfig, PredictionConfig
    from datalabeling.ml.interface import InferenceEngine
    from datalabeling.common.dataset_loader import LabelingDataset

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
        imgsz=imgsz,
        tilesize=tilesize,
        overlap_ratio=overlap_ratio,
        confidence_threshold=confidence_threshold,
        inference_service_url=None,
        # min_area=100,
        # max_area=None,
        cls_imgsz=cls_imgsz,
        # device="cuda:0",
    )

    engine, _ = InferenceEngine.load_engine(
        pred_config=config,
        roi_classifier_path=None,  # r"..\base_models_weights\roi_classifier.ckpt",
        roi_cls_is_features=True,
        roi_cls_label_map=roi_cls_label_map,
        roi_keep_classes=roi_keep_classes,
        detection_label_map=label_map,
        feature_extractor_path=feature_extractor_path,
        model_path=model_path,
        detection_model_type=detection_model_type,
        mlflow_model_alias=mlflow_model_alias,
        mlflow_model_name=mlflow_model_name,
    )

    perf_eval = PerformanceEvaluator(eval_config)

    # creating dataset and adding predictions
    dataset = LabelingDataset.from_yolo(yolo_images_dirs, label_map=label_map)

    print(dataset.get_stats())

    if not load_results:
        dataset.add_predictions(engine=engine, build=True)

    # run evaluator
    df_results, df_metrics_per_img = perf_eval.run(
        dataset=dataset,
        pred_results_dir=pred_results_dir,
        load_results=load_results,
    )

    # mining hard sampels
    sample_selector = HardSampleSelector(config=eval_config)
    df_hard_negatives = sample_selector.run(df_results)

    # report generation
    reporter = ReportGenerator()
    stats = reporter.run(
        df_results, plot=isinstance(save_plot, str), save_plot=save_plot
    )

    if isinstance(save_hn, str):
        df_hard_negatives.to_csv(save_hn, index=False)

    return df_results, df_metrics_per_img, stats, df_hard_negatives


def visualize_dataset(
    dataset_path: str, fo_dataset_name: str, fo_dataset_persistent: bool = True
):
    from datalabeling.common.visualizer import FiftyOneVisualizer
    import joblib

    print("Loading dataset...")

    dataset = joblib.load(dataset_path)

    print(dataset.get_stats())

    visualizer = FiftyOneVisualizer(
        dataset=dataset, dataset_name=fo_dataset_name, persistent=fo_dataset_persistent
    )
    visualizer.create_load_dataset()

    logger.info("Fiftyone dataset '{fo_dataset_name}' created.")


if __name__ == "__main__":
    fire.Fire()
