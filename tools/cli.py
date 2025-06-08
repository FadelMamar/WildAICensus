import fire
import os

import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv
from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio

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
from datalabeling.ml.interface import load_engine
from datalabeling.common.config import TrainingConfig
from datalabeling.ml.train import TrainingManager
from datalabeling.common.config import EvaluationConfig, PredictionConfig
from datalabeling.common.io import load_yaml
from datalabeling.common.dataset_loader import (
    ClassificationDatasetBuilder,
)

# import mlflow
# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
# from ultralytics import settings
# # Update a setting
# settings.update({"mlflow": True})
# mlflow.set_tracking_uri("file:///c:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow")
# mlflow.set_tracking_uri("http://localhost:5000")


logger = logging.getLogger(__name__)


# TODO debug
def create_classification_data(
    yaml_path: str,
    save_dir: str,
    strategies: list[str] = ["gt", "hn", "fp"],
    alias="demo",
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
        cls_imgsz=196,
        # device="cpu",
    )

    handler = ClassificationDatasetBuilder(
        eval_config,
    )

    cfg = load_yaml(yaml_path)

    engine, feature_extractor = load_engine(
        pred_config,
        roi_classifier_path=None,
        roi_cls_is_features=True,
        roi_cls_label_map={},
        mlflow_model_alias=alias,
    )

    for split in ["train", "val"]:
        source_dirs = [os.path.join(cfg["path"], subset) for subset in cfg[split]]

        print(f"source_dirs: {source_dirs}")

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
        herdnet_training_backend=args.herdnet_training_backend,
        classifier_training_backend=args.cls_training_backend,
        model_type=args.model_type,
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


if __name__ == "__main__":
    fire.Fire()
