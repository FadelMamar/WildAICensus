from datalabeling.common.config import PredictionConfig, FlightSpecs
from datalabeling.ml.interface import InferenceEngine
from datalabeling.ml.models import (
    UltralyticsDetector,
    GroundingDinoDetector,
    build_detector,
)
from datalabeling.ml.workers import ObjectDetectionSystem
from ultralytics import YOLO
from datalabeling.common.mlflow_utils import load_registered_model
import os
import torch
import numpy as np
from PIL import Image
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from time import perf_counter
from pathlib import Path
from datalabeling.common.base import Tile

config = PredictionConfig(
    imgsz=800,
    tilesize=800,
    batch_size=4,
    overlap_ratio=0.2,
    confidence_threshold=0.2,
    inference_service_url=None,  # "http://localhost:4141/predict",  # None "http://localhost:4141/predict"
    flight_specs=FlightSpecs(
        flight_height=180,
        sensor_height=24,
    ),
    nms_iou=0.5,
    verbose=False,
    # min_area=100,
    # max_area=None,
    cls_imgsz=98,
    # device="cuda:0",
)

ALIAS = "demo"  # -rt-batch8'
NAME = "labeler"

MODEL_PATH = "D:/datalabeling/base_models_weights/best.pt"
roi_classifier_path = None  # r"..\base_models_weights\roi_classifier.ckpt"
roi_cls_is_features = True
roi_cls_label_map = {0: "gt", 1: "tn"}
roi_keep_classes = ["gt"]
detection_label_map = {0: "wildlife"}
feature_extractor_path = "facebook/dinov2-with-registers-small"
timeout = 15
buffer_size = 64


def run_inference_engine(image_paths: list[str]):
    engine, _ = InferenceEngine.load_engine(
        pred_config=config,
        roi_classifier_path=roi_classifier_path,
        roi_cls_is_features=roi_cls_is_features,
        roi_cls_label_map=roi_cls_label_map,
        roi_keep_classes=roi_keep_classes,
        detection_label_map=detection_label_map,
        feature_extractor_path=feature_extractor_path,
        model_path=MODEL_PATH,
        mlflow_model_alias=ALIAS,
        mlflow_model_name=NAME,
        timeout=timeout,
        buffer_size=buffer_size,
    )

    detections = engine.inference(images_paths=image_paths, return_as_df=True)

    return detections


def run_inference_on_dataset(
    project_id=4,
    top_n=0,
    load_existing_metadata=True,
    untiled_data_dir=r"D:\workspace\data\savmap_dataset_v2\raw\images",
):
    from label_studio_sdk.client import LabelStudio
    from datalabeling.common.config import TilingConfig
    from datalabeling.common.dataset_loader import LabelingDataset, TileBuilder
    from dotenv import load_dotenv

    engine, _ = InferenceEngine.load_engine(
        pred_config=config,
        roi_classifier_path=roi_classifier_path,
        roi_cls_is_features=roi_cls_is_features,
        roi_cls_label_map=roi_cls_label_map,
        roi_keep_classes=roi_keep_classes,
        detection_label_map=detection_label_map,
        feature_extractor_path=feature_extractor_path,
        model_path=MODEL_PATH,
        mlflow_model_alias=ALIAS,
        mlflow_model_name=NAME,
    )

    # # Load environment variables
    load_dotenv(dotenv_path="../.env")

    # # label studio client
    LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
    API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
    labelstudio_client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

    # collect tile metadata: gps coords
    tiling_config = TilingConfig(
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

    tile_metadata = TileBuilder(config=tiling_config).run(
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

    dataset.add_predictions(engine, build=True)
    return dataset


def run_model(path: str):
    model = build_detector(
        detection_model_type="ultralytics",
        model_path=MODEL_PATH,
        model=None,
        config=config,
    )

    # model = build_detector(detection_model_type="hf-groundingdino",
    #                                  model_path="IDEA-Research/grounding-dino-tiny",
    #                                  model=None,
    #                                  config=config,
    #                                  )

    image = Image.open(path)

    image = torch.rand(1, 3, 800, 800)

    result = model.predict(
        image=image,
    )

    return result


def run_detector(
    image_paths: list,
):
    # t1_start = perf_counter()

    detector = ObjectDetectionSystem(
        config=config, buffer_size=32, timeout=15, detection_label_map={0: "wildlife"}
    )
    # detector.set_processor(roi_processor=processor)

    model = build_detector(
        detection_model_type="ultralytics",
        model_path=MODEL_PATH,
        model=None,
        config=config,
    )

    # model = build_detector(detection_model_type="hf-groundingdino",
    #                                  model_path="IDEA-Research/grounding-dino-tiny",
    #                                  model=None,
    #                                  config=config,
    #                                  )

    detector.set_model(model=model)

    results = detector.run(images_paths=image_paths)

    return results


def run_annotator(
    project_id=4,
    top_n=3,
    add_processor=True,
    dotenv_path="../.env",
):
    annotator, _ = InferenceEngine.load_engine(
        pred_config=config,
        roi_classifier_path=roi_classifier_path if add_processor else None,
        roi_cls_is_features=roi_cls_is_features,
        roi_cls_label_map=roi_cls_label_map,
        roi_keep_classes=roi_keep_classes,
        detection_label_map=detection_label_map,
        feature_extractor_path=feature_extractor_path,
        model_path=MODEL_PATH,
        mlflow_model_alias=ALIAS,
        mlflow_model_name=NAME,
        set_ls_client=True,
        dot_env_path=dotenv_path,
    )

    annotator.upload_predictions(
        project_id=project_id, top_n=top_n, tag="-" + str(add_processor)
    )

    return "success"


if __name__ == "__main__":
    # image_path = r"D:\workspace\data\savmap_dataset_v2\annotated_py_paul\yolo_format\images\00a033fefe644429a1e0fcffe88f8b39_0_4_0_512_640_1152.jpg"
    # image_path = r"D:\workspace\data\savmap_dataset_v2\raw\tmp\0a4a499660dc4e7c986779f8b6786f87.JPG"
    image_path = r"D:\workspace\data\general_dataset\original-data\train\images\0af7b1ea3a107e511353adbaba10c2e55a0bddf2.JPG"

    # image_path = r"D:\workspace\data\savmap_dataset_v2\annotated_py_paul\yolo_format\images\00a033fefe644429a1e0fcffe88f8b39_0_4_0_1024_640_1664.jpg"

    images = [image_path]

    # images = Path(
    #     r"D:\PhD\Data per camp\Dry season\Kapiri\Camp 3\Rep 2 - tiled"
    # ).glob("*.JPG")
    # images = list(images)[:20]

    # results = run_model(image_path)

    # results = run_detector(image_paths=images)

    # t1_start = perf_counter()

    # results = run_inference_engine(images)

    # t1_stop = perf_counter()
    # print("Inference time: ", t1_stop - t1_start)

    # print("detections:", detections)

    # image = imread(img_path)

    # for add_processor in [True]:
    #     results = run_annotator(
    #         project_id=94,
    #         top_n=3,
    #         add_processor=add_processor,
    #         inference_service_url=None,
    #     )

    pass
