from datalabeling.ml.train import ImageClassifier
from datalabeling.ml.models import Detector
from datalabeling.common.processor import get_processor, DetectionsPostprocessor
from datalabeling.common.annotation_utils import resize_bbox

from datalabeling.common.config import PredictionConfig
from datalabeling.ml.interface import InferenceEnginge

import torch
import numpy as np
from PIL import Image
from skimage.io import imread, imsave
import matplotlib.pyplot as plt


def run(img_path: str, num_classes: int = 2):
    config = PredictionConfig(
        imgsz=800,
        tilesize=800,
        overlap_ratio=0.2,
        confidence_threshold=0.2,
        # min_area=100,
        # max_area=None,
        cls_imgsz=128,
        device="cpu",
    )

    # get image classifier
    path = r"D:\datalabeling\src\tests\runs-classifier\best-v6.ckpt"
    model = ImageClassifier.load_from_checkpoint(
        path, cls_is_features=True, map_location="cpu"
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

    # get detector
    detector = Detector(
        path_to_weights=r"D:\datalabeling\base_models_weights\best.pt",
        confidence_threshold=config.confidence_threshold,
        overlap_ratio=config.overlap_ratio,
        tilesize=config.tilesize,
        imgsz=config.imgsz,
        use_sliding_window=config.use_sliding_window,
        device=config.device,
    )

    detections = detector.predict(image_path=img_path)

    image = Image.open(img_path)
    selected = processor.run(detections, image, box_size=96)

    return detections, selected


if __name__ == "__main__":
    img_path = r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images\0d1ba3c424ad4414ac37dbd0c93460ea_1_51_0_1024_640_1664.jpg"
    # img_path = r"D:\savmap_dataset_v2\raw\tmp\0a3ed15cfab4453795564140e8fde8ba.JPG"
    # img_path = r"D:\paul_data\DJI_20231002100957_0001.JPG"
    detections, selected = run(img_path)

    image = imread(img_path)

    for i, det in enumerate(selected):
        x1, x2, y1, y2 = resize_bbox(
            factor=3,
            x1=det.x_min,
            x2=det.x_max,
            y1=det.y_min,
            y2=det.y_max,
            img_height=image.shape[0],
            img_width=image.shape[1],
        )
        img = image[y1:y2, x1:x2]
        imsave(str(i) + "_example.jpg", img)

        # plt.show()
