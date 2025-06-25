# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 19:29:12 2025

@author: FADELCO
"""

# import fire

from tqdm import tqdm
import os
from datalabeling.common.config import (
    TilingConfig,
    EvaluationConfig,
    PredictionConfig,
    DataConfig,
    LabelConfig,
)
from datalabeling.common.dataset_loader import LabelingDataset, TileBuilder
from datalabeling.ml.workers import ObjectDetectionSystem
from datalabeling.ml.interface import InferenceEngine
from datalabeling.ml.models import Detector, ImageClassifier
from datalabeling.common.processor import get_processor, DetectionsPostprocessor
from ultralytics import YOLO
from dotenv import load_dotenv
import os
from sahi.utils.file import load_json
from label_studio_sdk.client import LabelStudio


def load_herd_net():
    from datalabeling.common.io import HerdnetData

    data_config_yaml = r"..\configs\yolo_configs\data_config.yaml"
    patch_size = 640
    batchsize = 4
    down_ratio = 2
    train_empty_ratio = 0.0

    datamodule = HerdnetData(
        data_config_yaml=data_config_yaml,
        patch_size=patch_size,
        tr_batch_size=batchsize,
        val_batch_size=1,
        down_ratio=down_ratio,
        train_empty_ratio=train_empty_ratio,
    )

    datamodule.setup("fit")
    num_classes = datamodule.num_classes

    for batch_train in tqdm(
        datamodule.train_dataloader(), desc="Iterating thru train_dataloader"
    ):
        continue

    for batch_val in tqdm(
        datamodule.val_dataloader(), desc="Iterating thru val_dataloader"
    ):
        continue


def create_classification_data(strategies: list[str] = ["gt", "hn"], alias="demo"):
    from datalabeling.common.io import load_yaml
    from datalabeling.common.mlflow_utils import load_registered_model
    from datalabeling.common.dataset_loader import (
        ClassificationDatasetBuilder,
    )
    from datalabeling.common.processor import get_processor

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
        # min_area=100,
        # max_area=None,
        cls_imgsz=128,
        # device="cpu",
    )

    handler = ClassificationDatasetBuilder(
        eval_config,
    )

    yaml_path = r"..\configs\yolo_configs\data\data_config.yaml"
    cfg = load_yaml(yaml_path)

    root_dir = r"D:\datalabeling\.tmp\cls-features"

    engine, feature_extractor = InferenceEngine.load_engine(
        pred_config,
        roi_classifier_path=r"..\base_models_weights\roi_classifier.ckpt",
        roi_cls_label_map={0: "gt", 1: "tn"},
        roi_cls_is_features=True,
        roi_keep_classes=["gt"],
        feature_extractor_path="facebook/dinov2-with-registers-small",
        detection_model=YOLO(model=r"..\base_models_weights\best.pt"),
        detection_label_map={0: "wildlife"},
    )

    for split in ["train", "val"]:
        source_dirs = [os.path.join(cfg["path"], subset) for subset in cfg[split]]

        print(f"source_dirs: {source_dirs}")

        handler.set_dirs(
            source_dirs=source_dirs, output_dir=os.path.join(root_dir, split)
        )

        handler.run(
            strategies=strategies,
            save_true_negatives=True,
            feature_extractor=feature_extractor,
            detector=engine,
            bbox_resize_factor=1,  # resizes the bbox for tn,tp,fp
            tn_kwargs=dict(
                w=pred_config.cls_imgsz, h=pred_config.cls_imgsz, number=3
            ),  # to disable use {}
            tp_kwargs=dict(
                w=pred_config.cls_imgsz, h=pred_config.cls_imgsz
            ),  # or {} to use actual bbox
            hn_kwargs=dict(w=pred_config.cls_imgsz, h=pred_config.cls_imgsz),
        )


def load_classification_features_data():
    from datalabeling.common.io import ClassifierDataModule

    data = ClassifierDataModule(
        data_dir=r"..\.tmp\cls-features",
        batch_size=64,
        is_features=True,
        img_size=96,
        tn_ratio=1.0,
    )

    data.setup("fit")

    for tr_batch in tqdm(data.train_dataloader(), desc="train loader"):
        pass

    for val_batch in tqdm(data.val_dataloader(), desc="train loader"):
        pass


def load_dataset_from_ls(
    untiled_data_dir: str,
    project_id=4,
    top_n=0,
    load_existing_metadata=True,
):
    # # Load environment variables
    load_dotenv(dotenv_path="../.env")

    # # label studio client
    LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
    API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
    labelstudio_client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

    # collect tile metadata: gps coords
    config = TilingConfig(
        root=untiled_data_dir,
        overlapfactor=0.1,
        ratiowidth=0.5,
        ratioheight=0.33,
        rmheight=0.0,
        rmwidth=0.0,
        flight_height=180,
        sensor_height=24,
        gsd=2.26,
        dest=r"..\.tmp",
        save_coords_only=True,  # set to False to save tiles i.e. patches
    )

    tile_metadata = TileBuilder(config=config).run(
        load_existing_metadata=True, max_workers=2
    )

    dataset = LabelingDataset.from_ls(
        labelstudio_client,
        project_id=project_id,
        config=config,
        top_n=top_n,
        max_workers=1,
        tile_metadata=tile_metadata,
        load_existing_metadata=load_existing_metadata,
    )

    return tile_metadata, dataset


def push_dataset_to_ls():
    # # Load environment variables
    load_dotenv(dotenv_path="../.env")

    # # label studio client
    LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
    API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
    labelstudio_client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

    images_dirs = [
        r"D:\workspace\data\savmap_dataset_v2\annotated_py_paul\yolo_format\images"
    ]

    dataset = LabelingDataset.from_yolo(
        images_dirs=images_dirs, load_empty=True, label_map=None
    )
    dataset.set_labelstudio_client(labelstudio_client)

    dataset.to_ls(project_title="savmap_yolo", reference_project_id=4)


def slice_and_save_as_yolo(dataset: LabelingDataset):
    data_config = DataConfig(
        is_single_cls=True,
        root_dir="D:\\",
        yolo_data_config_yaml="../configs/yolo_configs/data/data_config.yaml",
        dotenv_path="../.env",
        tilesize=640,
        overlap_ratio=0.2,
        save_all=False,
        save_only_empty=False,
        load_coco_annotations=False,
        parse_ls_config=False,
        empty_ratio=1.0,
    )
    label_config = LabelConfig(
        keep=["wildlife"], label_map="../exported_annotations/label_mapping.json"
    )

    dataset.slice_and_save_as_yolo(data_config, label_config)

    return None


def load_dataset_from_dirs():
    images_dirs = [
        r"D:\savmap_dataset_v2\images_tmp",
    ]

    dataset = LabelingDataset.from_dirs(images_dirs)

    return dataset


def load_dataset_from_yolo():
    images_dirs = [
        r"D:\workspace\data\savmap_dataset_v2\annotated_py_paul\yolo_format\images",
    ]

    dataset = LabelingDataset.from_yolo(
        images_dirs=images_dirs,
        paths=None,
        load_empty=True,
        max_workers=2,
        label_map={0: "wildlife"},
    )

    return dataset


if __name__ == "__main__":
    #     fire.Fire(
    #         {
    #             "create-features": create_classification_data,
    #             "load-features": load_classification_features_data,
    #             "load-herdnet": load_herd_net,
    #         }
    #     )

    # create_classification_data([
    #                             # "gt",
    #                             # "hn",
    #                             # "fp"
    #                             ])

    # load_classification_features_data()

    # tile_metadata, dataset = load_dataset_from_ls(
    #     project_id=4,
    #     top_n=0,
    #     load_existing_metadata=True,
    #     untiled_data_dir=r"D:\workspace\data\savmap_dataset_v2\raw\images",
    # )

    # slice_and_save_as_yolo(dataset=dataset)

    # push_dataset_to_ls()

    # data = dataset.data
    # gps_data = dataset.export_detections_gps()

    # dataset = load_dataset_from_dirs()

    dataset = load_dataset_from_yolo()

    pass
