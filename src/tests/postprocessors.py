from datalabeling.ml.train import ImageClassifier
from datalabeling.ml.models import Detector
from datalabeling.common.processor import (
    Classifier,
    DetectionsPostprocessor,
    FeatureExtractor,
)
import torch
import numpy as np
from PIL import Image
from skimage.io import imread, imsave
import matplotlib.pyplot as plt


def run_image_classifier(num_classes: int = 2):
    def format_det(pred_coco: dict) -> dict:
        bbox = pred_coco["bbox"]
        label = pred_coco["category_id"]
        class_ = pred_coco["category_name"]
        score = pred_coco["score"]

        return dict(
            x_min=int(bbox[0]),
            y_min=int(bbox[1]),
            x_max=int(bbox[0] + bbox[2]),
            y_max=int(bbox[1] + bbox[3]),
            class_=class_,
            label=label,
            score=score,
        )

    model = torch.nn.Sequential(
        torch.nn.LazyLinear(128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.2),
        torch.nn.LazyLinear(128),
        torch.nn.ReLU(),
        torch.nn.LazyLinear(num_classes),
    )

    path = r"D:\datalabeling\src\tests\569247532315291665\d31c200a1c064ab985e8315d5495345a\checkpoints\epoch=9-step=2990.ckpt"

    model = ImageClassifier.load_from_checkpoint(path, model=model, map_location="cpu")
    cls_ = Classifier(
        model,
        label_map={0: "gt", 1: "tn"},
        device="cpu",
        feature_extractor=FeatureExtractor(),
        imgsz=96,
    )

    processor = DetectionsPostprocessor(
        classifier=cls_,
        keep_classes=["gt"],
    )

    detector = Detector(
        path_to_weights=r"D:\datalabeling\base_models_weights\best.pt",
        confidence_threshold=0.25,
        overlap_ratio=0.2,
        tilesize=800,
        imgsz=800,
        use_sliding_window=True,
        device="cpu",
    )

    img_path = r"D:\savmap_dataset_v2\raw\images\0e06d6a1720a4b0190cd16a93f2f178d.JPG"

    results = detector.predict(image_path=img_path, return_coco=True)

    detections = [format_det(det) for det in results]

    image = imread(img_path, as_gray=False)
    selected = processor.run(detections, image, box_size=96)

    return detections, selected


if __name__ == "__main__":
    detections, selected = run_image_classifier()

    img_path = r"D:\savmap_dataset_v2\raw\images\0e06d6a1720a4b0190cd16a93f2f178d.JPG"

    image = imread(img_path)

    for i, det in enumerate(selected):
        img = image[det["y_min"] : det["y_max"], det["x_min"] : det["x_max"]]
        imsave(str(i) + "_example.jpg", img)

        # plt.show()
