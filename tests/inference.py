from datalabeling.common.config import PredictionConfig
from datalabeling.ml.interface import InferenceEngine
from datalabeling.ml.models import UltralyticsDetector, GroundingDinoDetector
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
    flight_height=180,
    sensor_height=24,
    gsd=None,
    nms_iou=0.5,
    verbose=False,
    # min_area=100,
    # max_area=None,
    cls_imgsz=98,
    # device="cuda:0",
)

ALIAS = "demo"  # -rt-batch8'
NAME = "labeler"


def run_inference_engine(image_paths: list[str]):
    detection_model = UltralyticsDetector(
        model_path="D:/datalabeling/base_models_weights/best.pt", config=config
    )

    engine, _ = InferenceEngine.load_engine(
        pred_config=config,
        roi_classifier_path=r"..\base_models_weights\roi_classifier.ckpt",
        roi_cls_is_features=True,
        roi_cls_label_map={0: "gt", 1: "tn"},
        roi_keep_classes=["gt"],
        detection_label_map={0: "wildlife"},
        feature_extractor_path="facebook/dinov2-with-registers-small",
        detection_model=detection_model,
        mlflow_model_alias="demo",
        mlflow_model_name="labeler",
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
        roi_classifier_path=r"..\base_models_weights\roi_classifier.ckpt",
        roi_cls_is_features=True,
        roi_cls_label_map={0: "gt", 1: "tn"},
        roi_keep_classes=["gt"],
        detection_label_map={0: "wildlife"},
        feature_extractor_path="facebook/dinov2-with-registers-small",
        detection_model=YOLO(r"D:\datalabeling\base_models_weights\best.pt"),
        mlflow_model_alias="demo",
        mlflow_model_name="labeler",
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
    # model = UltralyticsDetector(model_path="D:/datalabeling/base_models_weights/best.pt",config=config)

    model = GroundingDinoDetector(config=config)

    image = Image.open(path)

    image = torch.rand(1, 3, 800, 800)

    result = model.predict(
        image=image,
    )

    return result


def run_detector(
    image_paths: list,
):
    t1_start = perf_counter()

    detector = ObjectDetectionSystem(
        config=config, buffer_size=32, timeout=15, detection_label_map={0: "wildlife"}
    )
    # detector.set_processor(roi_processor=processor)

    model = UltralyticsDetector(
        model_path="D:/datalabeling/base_models_weights/best.pt", config=config
    )

    # model = GroundingDinoDetector(config=config)

    detector.set_model(model=model)

    results = detector.run(images_paths=image_paths)

    # results_url = None
    # results_url = Detector.predict_url(
    #     image_path=tile.image_path,
    #     inference_service_url="http://localhost:4141/predict",
    # )

    # print(tile.predictions)
    # print(results[0])
    # print(results_url[0][0])

    print("Inference time improved: ", perf_counter() - t1_start)

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
    dotenv_path="../.env",
):
    annotator, _ = InferenceEngine.load_engine(
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
    # image_path = img = r"D:\workspace\data\general_dataset\original-data\train\images\0af7b1ea3a107e511353adbaba10c2e55a0bddf2.JPG"

    image_path = r"D:\workspace\data\savmap_dataset_v2\annotated_py_paul\yolo_format\images\00a033fefe644429a1e0fcffe88f8b39_0_4_0_1024_640_1664.jpg"

    images = [image_path]

    # images = Path(
    #     r"D:\PhD\Data per camp\Dry season\Kapiri\Camp 3\Rep 2 - tiled"
    # ).glob("*.JPG")
    # images = list(images)[:20]

    # results = run_model(image_path)

    # results = run_detector(image_paths=images)

    # t1_start = perf_counter()

    results = run_inference_engine(images)

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
