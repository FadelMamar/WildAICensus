from datalabeling.common.census import (
    GPSOverlapStrategy,
    CentroidProximityRemovalStrategy,
    WildlifeCensusSystem,
)
from datalabeling.common.base import Tile, Detection
from datalabeling.common.dataset_loader import LabelingDataset
import random
# import uuid


# Helper to create a mock tile with predictions
def make_detection(parent_image: str):
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
    dataset = load_dataset_from_dirs(r"D:\workspace\data\savmap_dataset_v2\raw\images")
    tiles = dataset.tiles

    overlap_strategy = GPSOverlapStrategy()
    overlap_map = overlap_strategy.find_overlapping_images(
        tiles, min_overlap_threshold=0.0
    )

    return overlap_map


def test_count_system():
    dataset = load_dataset_from_dirs(r"D:\workspace\data\savmap_dataset_v2\raw\images")

    # Run GPSOverlapStrategy
    overlap_strategy = GPSOverlapStrategy()

    # Run DuplicateRemovalStrategy
    finder = CentroidProximityRemovalStrategy()

    # Run WildlifeCensusSystem
    census_system = WildlifeCensusSystem(overlap_strategy, finder)
    census_system.set_dataset(dataset)
    census_system.run(image_overlap_threshold=0.0, detection_iou_threshold=0.8)

    return census_system


if __name__ == "__main__":
    # overlap_map = test_gps_overlap()

    census_system = test_count_system()
