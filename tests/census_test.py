from datalabeling.common.census import (
    GPSOverlapStrategy,
    CentroidProximityRemovalStrategy,
    WildlifeCountingSystem,
    run_census,
)
from datalabeling.common.base import Tile, Detection
from datalabeling.common.dataset_loader import LabelingDataset
import random
import os
from datalabeling.ml.interface import InferenceEngine
from datalabeling.common.config import PredictionConfig, FlightSpecs
# import uuid

# EXAMPLE_DIR = r"D:\workspace\data\savmap_dataset_v2\raw\images"

EXAMPLE_DIR = r"D:\PhD\Data per camp\Dry season\Kapiri\Camp 2\Rep 1"


def make_detection(parent_image: str):
    """
    Create a random Detection object for a given parent image.
    Args:
        parent_image (str): Path to the parent image.
    Returns:
        Detection: A randomly generated detection.
    """
    x = random.randint(0, 800)
    y = random.randint(0, 800)
    w = random.randint(20, 50)
    h = random.randint(20, 50)
    score = random.random()
    class_name = "wildlife"

    return Detection(
        x_min=x,
        y_min=y,
        x_max=x + w,
        y_max=y + h,
        label=0,
        class_name=class_name,
        score=score,
        parent_image=parent_image,
    )


def load_dataset_from_dirs(image_dir: str):
    """
    Load a LabelingDataset from a directory of images, generating random predictions for each image.
    Args:
        image_dir (str): Directory containing images.
    Returns:
        LabelingDataset: Dataset with tiles and random predictions.
    """
    images_dirs = [image_dir]

    dataset = LabelingDataset.from_dirs(images_dirs)
    tiles = []
    for image_path in dataset.data["file_name"].unique():
        predictions = [make_detection(image_path) for _ in range(random.randint(1, 10))]
        tile = Tile(image_path=image_path, predictions=predictions)
        tiles.append(tile)

    dataset.tiles = tiles
    dataset.build(True)

    return dataset


def test_gps_overlap():
    """
    Test function to compute the overlap map for a dataset using GPSOverlapStrategy.
    Returns:
        dict: The overlap map between images.
    """
    dataset = load_dataset_from_dirs(EXAMPLE_DIR)
    tiles = dataset.tiles

    print(dataset.get_stats())

    overlap_strategy = GPSOverlapStrategy()
    overlap_map = overlap_strategy.find_overlapping_images(
        tiles, min_overlap_threshold=0.0
    )

    print(overlap_strategy.stats)

    return overlap_map


def test_count_system():
    """
    Test function to run the full wildlife census system pipeline:
    - Loads a dataset
    - Runs overlap and duplicate removal strategies
    - Returns the census system object
    Returns:
        WildlifeCensusSystem: The system after running the pipeline.
    """
    dataset = load_dataset_from_dirs(EXAMPLE_DIR)

    # Run GPSOverlapStrategy
    overlap_strategy = GPSOverlapStrategy()

    # Run DuplicateRemovalStrategy
    finder = CentroidProximityRemovalStrategy()

    # Run WildlifeCountingSystem
    census_system = WildlifeCountingSystem(overlap_strategy, finder)
    census_system.set_dataset(dataset)
    census_system.run(
        image_overlap_threshold=0.0,
        detection_iou_threshold=0.8,
        filepath="census_results.json",
    )

    return census_system


def test_inference_and_save_predictions():
    """
    Test function to run the inference engine on a dataset and save predictions to a CSV file.
    Returns:
        str: Path to the saved predictions CSV file.
    """

    # Run inference engine on the dataset
    config = PredictionConfig(
        imgsz=800,
        tilesize=800,
        batch_size=4,
        overlap_ratio=0.2,
        confidence_threshold=0.2,
        inference_service_url=None,
        flight_specs=FlightSpecs(
            flight_height=180,
            sensor_height=24,
        ),
        nms_iou=0.5,
        verbose=False,
        cls_imgsz=98,
    )

    ALIAS = "demo"
    NAME = "labeler"
    MODEL_PATH = None #"D:/datalabeling/base_models_weights/best.pt"
    roi_classifier_path = r"../base_models_weights/roi_classifier.ckpt"
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

    census_system = run_census(
        images_dir=[EXAMPLE_DIR],
        engine=engine,
        overlap_strategy="GPSOverlapStrategy",
        duplicate_removal_strategy="CentroidProximityRemovalStrategy",
        image_overlap_threshold=0.0,
        detection_iou_threshold=0.8,
        save_path="census_results.json",
        fiftyone_dataset_name="savmap_dataset_v2-raw-census-demo",
        fiftyone_persistent=False,
    )

    return census_system


if __name__ == "__main__":
    #overlap_map = test_gps_overlap()

    census_system = test_count_system()

    # test_inference_and_save_predictions()
