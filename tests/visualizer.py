# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 14:42:46 2025

@author: FADELCO
"""

import os

from datalabeling.common.dataset_loader import LabelingDataset
from datalabeling.common.visualizer import FiftyOneVisualizer
from datalabeling.ml.interface import InferenceEngine
from datalabeling.common.config import PredictionConfig


def load_dataset(load_results=True):
    config = PredictionConfig(
        imgsz=640,
        tilesize=640,
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

    MODEL_PATH = "D:/datalabeling/base_models_weights/best.pt"
    roi_classifier_path = r"..\base_models_weights\roi_classifier.ckpt"
    roi_cls_is_features = True
    roi_cls_label_map = {0: "gt", 1: "tn"}
    roi_keep_classes = ["gt"]
    detection_label_map = {0: "wildlife"}
    feature_extractor_path = "facebook/dinov2-with-registers-small"

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

    save_tag = ""
    pred_results_dir = r"D:\workspace\data\savmap_dataset_v2\images_tmp"
    stem = f"predictions-{save_tag}" + save_tag if len(save_tag) > 0 else "predictions"
    save_path = os.path.join(pred_results_dir, stem + ".csv")

    if load_results:
        dataset.import_data(save_path)

    else:
        dataset.add_predictions(engine=engine, build=True)

    return dataset


if __name__ == "__main__":
    dataset = load_dataset(load_results=True)

    visualizer = FiftyOneVisualizer(
        dataset=dataset, dataset_name="savmap_detection_test", persistent=True
    )

    # visualizer.run(port=5151)

    visualizer.create_load_dataset()

    # while True:

    #     continue
