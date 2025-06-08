from datalabeling.common.config import PredictionConfig
from datalabeling.ml.interface import InferenceEngine, Annotator, load_engine
from datalabeling.ml.workers import ObjectDetectionSystem

from datalabeling.common.mlflow_utils import load_registered_model

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
    batch_size=8,
    overlap_ratio=0.2,
    confidence_threshold=0.2,
    inference_service_url="http://localhost:4141/predict",  # None "http://localhost:4141/predict"
    flight_height=180,
    sensor_height=24,
    gsd=None,
    nms_iou=0.5,
    verbose=True,
    # min_area=100,
    # max_area=None,
    cls_imgsz=128,
    # device="cuda:0",
)

ALIAS = "demo"  # -rt-batch8'
NAME = "labeler"


def run_inference_engine(img_path: str):
    engine, _ = load_engine(
        pred_config=config,
        roi_classifier_path=r"..\base_models_weights\roi_classifier.ckpt",
        roi_cls_is_features=True,
        roi_cls_label_map={0: "gt", 1: "tn"},
        roi_keep_classes=["gt"],
        detection_label_map={0: "wildlife"},
        feature_extractor_path="facebook/dinov2-with-registers-small",
        detection_model=None,
        mlflow_model_alias="demo",
        mlflow_model_name="labeler",
    )

    detections = engine.inference(images_paths=[img_path])

    return detections


def run_detector(
    image_paths: list,
):
    t1_start = perf_counter()

    detector = ObjectDetectionSystem(
        config=config, buffer_size=32, timeout=15, detection_label_map={0: "wildlife"}
    )
    # detector.set_processor(roi_processor=processor)
    # detector.set_model(
    #     model=None, path_weights=r"D:\datalabeling\base_models_weights\best.pt"
    # )

    results = detector.run(images_paths=image_paths)

    # results_url = None
    # results_url = Detector.predict_url(
    #     image_path=tile.image_path,
    #     inference_service_url="http://localhost:4141/predict",
    # )

    # print(tile.predictions)
    # print(results[0])
    # print(results_url[0][0])

    # print("Inference time improved: ", perf_counter() - t1_start)

    # t1_start = perf_counter()
    # results = detector.legacy_predict(tile=None,image_path=tile.image_path)
    # perf2 = perf_counter() - t1_start
    # print("Inference time SAHI: ", perf_counter() - t1_start)

    # print("speed up:", perf2/perf1)

    return results


def run_annotator(
    project_id=4,
    top_n=3,
    add_processor=True,
    inference_service_url=None,
    dotenv_path="../.env",
):
    # get annotator
    annotator = Annotator(
        config=config,
        dotenv_path=dotenv_path,
    )

    detector = Detector(config=config, detection_model=detection_model)
    annotator.set_detector(detector, model_tag=ALIAS)

    if add_processor:
        annotator.set_processor(image_processor=None, detection_processor=processor)

    annotator.upload_predictions(
        project_id=project_id, top_n=top_n, tag="-" + str(add_processor)
    )

    return "success"


if __name__ == "__main__":
    # image_path = r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images\0d1ba3c424ad4414ac37dbd0c93460ea_1_51_0_1024_640_1664.jpg"
    # image_path = r"D:\savmap_dataset_v2\raw\tmp\0a3ed15cfab4453795564140e8fde8ba.JPG"
    # image_path = r"D:\workspace\data\savmap_dataset_v2\raw\tmp\0a4a499660dc4e7c986779f8b6786f87.JPG"

    # tile = Tile(image_path=image_path, parent_image=image_path)

    images = Path(
        r"D:\workspace\data\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images"
    ).glob("*.JPG")
    results = run_detector(image_paths=images)

    # data = tile.detections_to_df()

    # detections = detection_model.model(
    #     Path(r"D:\PhD\Data per camp\DetectionDataset\savmap\images"), iou=0.5, batch=8
    # )

    # t1_start = perf_counter()

    # detections = run_inference_engine(image_path)

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

    # Inference using deployment
    # config = PredictionConfig(
    #             imgsz=800,
    #             tilesize=800,
    #             overlap_ratio=0.2,
    #             confidence_threshold=0.2,
    #             # min_area=100,
    #             # max_area=None,
    #             cls_imgsz=128,
    #             # device="cuda:0",
    #             )

    # model = YOLO(r"D:\datalabeling\base_models_weights\best.pt")

    # images = Path(r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images").glob('*')
    # sample_images = list(images)[:5]
    # sample_images = [Image.open(p) for p in sample_images]
    # preds = model.predict(sample_images,batch=5)

    pass
