from datalabeling.ml.models import Detector, ImageClassifier
from datalabeling.common.processor import get_processor, DetectionsPostprocessor
from datalabeling.common.annotation_utils import resize_bbox

from datalabeling.common.config import PredictionConfig
from datalabeling.ml.interface import InferenceEngine, Annotator

from datalabeling.common.mlflow_utils import load_registered_model

from ultralytics import YOLO
from sahi.models.ultralytics import UltralyticsDetectionModel

import torch
import numpy as np
from PIL import Image
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from time import perf_counter
from pathlib import Path


config = PredictionConfig(
    imgsz=800,
    tilesize=800,
    overlap_ratio=0.2,
    confidence_threshold=0.2,
    # min_area=100,
    # max_area=None,
    cls_imgsz=128,
    # device="cuda:0",
)

# get image classifier
path = r"./runs-classifier/best-v2.ckpt"
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

ALIAS='yolo11s-obb-v1' #-rt-batch8'
NAME='labeler'

# set detector
detection_model,version = load_registered_model(alias=ALIAS,
                                                name=NAME,
                                                load_unwrapped=True)
if isinstance(detection_model, YOLO):
    detection_model =  UltralyticsDetectionModel(
            model=detection_model,
            confidence_threshold=config.confidence_threshold,
            category_mapping={"0":"wildlife"},
            # image_size=config.imgsz,
            device=config.device,
        )

def run_inference_engine(img_path: str, num_classes: int = 2):
    # get inference engine
    engine = InferenceEngine(config=config)

    detector = Detector(config=config, detection_model=detection_model)
    # detector.set_detection_model(detection_model=None,
    #                              path_to_weights=""
    #                              )
    engine.set_detector(detector, model_tag=ALIAS)

    # set processors
    engine.set_processor(image_processor=None, detection_processor=processor)

    detections = engine.inference(
        image_path=img_path, image=None, inference_service_url=None
    )

    return detections


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
    image_path = r"..\.tmp\images\DJI_20240204125354_0144.JPG"

   

    t1_start = perf_counter() 

    detections = detection_model.model(Path(r"D:\PhD\Data per camp\DetectionDataset\savmap\images"),iou=0.5,batch=8)

    # detections  = run_inference_engine(image_path)

    t1_stop = perf_counter()
    print("Inference time: ", t1_stop - t1_start)
    print("detections:",detections)

    # image = imread(img_path)

    # for i, det in enumerate(selected):
    #     x1, x2, y1, y2 = resize_bbox(factor=3,
    #                                  x1=det.x_min,x2=det.x_max,
    #                                  y1=det.y_min,y2=det.y_max,
    #                                  img_height=image.shape[0],
    #                                  img_width=image.shape[1]
    #                                  )
    #     img = image[y1:y2,x1:x2]
    #     imsave(str(i) + "_example.jpg", img)

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
    # detections = Detector(None,config=config).predict(image_path=image_path,
    #                  inference_service_url="http://localhost:4141/predict"
    #                 )

    # print(detections)

    # model = YOLO(r"D:\datalabeling\base_models_weights\best.pt")

    # images = Path(r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images").glob('*')
    # sample_images = list(images)[:5]
    # sample_images = [Image.open(p) for p in sample_images]
    # preds = model.predict(sample_images,batch=5)

    pass
