#!/usr/bin/env python3
"""
Wildlife Detection System for Aerial Imagery with Overlap Handling

This system implements a comprehensive solution for counting wildlife in overlapping
aerial images, using GPS metadata for spatial analysis and confidence-based
duplicate removal.

Design Patterns Used:
- Strategy Pattern: For different overlap detection strategies
- Observer Pattern: For progress tracking and logging
- Factory Pattern: For creating detection processors
- Command Pattern: For batch processing operations
"""

from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from abc import ABC, abstractmethod
from enum import Enum
import json
import logging
from datetime import datetime
import math
from itertools import product
from collections import defaultdict
from tqdm import tqdm
import torch
from torchmetrics.functional.detection import complete_intersection_over_union
import copy
import os

from .base import Tile, Detection
from .dataset_loader import LabelingDataset
from .evaluation import ReportGenerator
from ..ml.interface import InferenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OverlapStrategy(ABC):
    """Abstract strategy for detecting overlapping images."""

    def __init__(
        self,
    ):
        self.stats = None
        pass

    @abstractmethod
    def find_overlapping_images(
        self, images: List[Tile], min_overlap_threshold: float = 0.0
    ) -> Dict[str, List[str]]:
        """Find overlapping images and return mapping of image_id -> overlapping_image_ids.
        Args:
            images (List[Tile]): List of Tile objects.
            min_overlap_threshold (float): Minimum IoU threshold to consider images as overlapping.
        Returns:
            Dict[str, List[str]]: Mapping from image ID to list of overlapping image IDs.
        """
        pass


class GPSOverlapStrategy(OverlapStrategy):
    """
    GPS-based overlap detection using geographic footprints.
    Provides methods to find overlapping images and compute statistics on the overlap map.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def _compute_iou(self, images: List[Tile]) -> np.ndarray:
        """Compute Intersection over Union (IoU) between all pairs of image tiles.
        Args:
            images (List[Tile]): List of Tile objects.
        Returns:
            np.ndarray: IoU matrix between all pairs of tiles.
        """

        boxes = [img.geo_box for img in images]
        boxes = torch.tensor(boxes)
        box_ious = complete_intersection_over_union(
            preds=boxes, target=boxes, aggregate=False
        ).numpy()

        return box_ious

    def find_overlapping_images(
        self, images: List[Tile], min_overlap_threshold: float = 0.0
    ) -> Dict[str, List[str]]:
        """Find overlapping images using precomputed IoU matrix.
        Args:
            images (List[Tile]): List of Tile objects.
            min_overlap_threshold (float): Minimum IoU threshold to consider images as overlapping.
        Returns:
            Dict[str, List[str]]: Map of image IDs to their overlapping neighbor image IDs.
        """
        overlap_map = defaultdict(list)
        ious = self._compute_iou(images=images)
        for i, img1 in enumerate(tqdm(images, desc="Finding overlapping images")):
            for j, img2 in enumerate(images):
                if i <= j:  # only consider above diagonal because it's symmetric
                    continue
                if ious[i, j] > min_overlap_threshold:
                    overlap_map[str(img1.image_path)].append(str(img2.image_path))

        overlap_map = dict(overlap_map)
        self.stats = self.overlap_map_stats(overlap_map)
        return overlap_map

    def overlap_map_stats(self, overlap_map: dict) -> dict:
        """Compute statistics on the overlap_map.
        Args:
            overlap_map (dict): Map of image IDs to their overlapping neighbor image IDs.
        Returns:
            dict: Statistics including number of images, average, max, min neighbors, and neighbor counts list.
        """
        num_images = len(overlap_map)
        neighbor_counts = [len(neighs) for neighs in overlap_map.values()]
        avg_neighbors = float(np.mean(neighbor_counts)) if neighbor_counts else 0.0
        max_neighbors = max(neighbor_counts) if neighbor_counts else 0
        min_neighbors = min(neighbor_counts) if neighbor_counts else 0
        return {
            "num_images": num_images,
            "avg_neighbors": avg_neighbors,
            "max_neighbors": max_neighbors,
            "min_neighbors": min_neighbors,
            # 'neighbor_counts': neighbor_counts,
        }


class DuplicateRemovalStrategy(ABC):
    """Abstract strategy for removing duplicate detections."""

    @abstractmethod
    def remove_duplicates(
        self,
        tiles: List[Tile],
        overlap_map: Dict[str, List[str]],
        iou_threshold: float = 0.8,
    ) -> List[Tile]:
        """Remove duplicate detections from tiles based on overlap map and IoU threshold.
        Args:
            tiles (List[Tile]): List of Tile objects.
            overlap_map (Dict[str, List[str]]): Overlap mapping between tiles.
            iou_threshold (float): IoU threshold for considering duplicates.
        Returns:
            List[Tile]: List of tiles with duplicates removed.
        """
        pass


class CentroidProximityRemovalStrategy(DuplicateRemovalStrategy):
    """
    Responsible for finding groups of duplicate predictions given a list of Tile objects with predictions.
    Can also return unique predictions using a specified duplicate removal strategy.
    """

    def __init__(
        self,
    ):
        pass

    def _compute_iou(
        self, detections_1: List[Detection], detections_2: List[Detection]
    ) -> np.ndarray:
        """Compute Intersection over Union (IoU) between two lists of detections.
        Args:
            detections_1 (List[Detection]): First list of detections.
            detections_2 (List[Detection]): Second list of detections.
        Returns:
            np.ndarray: IoU matrix between detections.
        """

        boxes = [det.geo_box for det in detections_1]
        boxes_2 = [det.geo_box for det in detections_2]
        boxes = torch.tensor(boxes)
        boxes_2 = torch.tensor(boxes_2)
        box_ious = complete_intersection_over_union(
            preds=boxes, target=boxes_2, aggregate=False
        )

        return box_ious.numpy()

    def _prune_duplicates_between_tiles(
        self, tile1: Tile, tile2: Tile, iou_threshold: float = 0.8
    ) -> Tuple[Tile, Tile]:
        """Prune duplicate detections between two tiles based on IoU and class name.
        Args:
            tile1 (Tile): First tile.
            tile2 (Tile): Second tile.
            iou_threshold (float): IoU threshold for considering duplicates.
        Returns:
            Tuple[Tile, Tile]: Tiles with duplicates pruned.
        """
        if not tile1.predictions or not tile2.predictions:
            return tile1, tile2

        ious = self._compute_iou(tile1.predictions, tile2.predictions)  # shape [N1, N2]
        keep1 = np.ones(len(tile1.predictions), dtype=bool)
        keep2 = np.ones(len(tile2.predictions), dtype=bool)

        # Only compare predictions of the same class
        for i, det1 in enumerate(tile1.predictions):
            for j, det2 in enumerate(tile2.predictions):
                if det1.class_name != det2.class_name:
                    ious[i, j] = -1  # Mark as invalid

        idxs1, idxs2 = np.where(ious > iou_threshold)
        for i, j in zip(idxs1, idxs2):
            det1 = tile1.predictions[i]
            det2 = tile2.predictions[j]
            # Only process if both are still marked to keep
            if not (keep1[i] and keep2[j]):
                continue
            if det1.distance_to_centroid > det2.distance_to_centroid:
                keep1[i] = False
            else:
                keep2[j] = False

        tile1.set_predictions(
            [det for i, det in enumerate(tile1.predictions) if keep1[i]]
        )
        tile2.set_predictions(
            [det for j, det in enumerate(tile2.predictions) if keep2[j]]
        )
        return tile1, tile2

    def remove_duplicates(
        self,
        tiles: List[Tile],
        overlap_map: Dict[str, List[str]],
        iou_threshold: float = 0.8,
    ) -> List[Tile]:
        """
        For every image (Tile), consider its neighbors' predictions (from overlap_map).
        For each prediction in the image, check all predictions in neighbor images.
        If class matches and IoU > threshold, save the pair (detection, neighbor_detection) in a list.
        Returns the list of such pairs.
        """
        image_path_to_tile = {str(tile.image_path): tile for tile in tiles}

        for image_path in tqdm(
            overlap_map.keys(), desc="Removing duplicates in Overlapping regions"
        ):
            tile = image_path_to_tile[image_path]
            for neighbor in overlap_map[image_path]:
                tile2 = image_path_to_tile[str(neighbor)]
                tile1, tile2 = self._prune_duplicates_between_tiles(
                    tile, tile2, iou_threshold
                )
                image_path_to_tile[str(tile1.image_path)] = tile1
                image_path_to_tile[str(tile2.image_path)] = tile2

        return list(image_path_to_tile.values())


class WildlifeCountingSystem:
    """Main system for wildlife detection and counting"""

    def __init__(
        self,
        overlap_strategy: OverlapStrategy,
        duplicate_removal_strategy: DuplicateRemovalStrategy,
    ):
        self.overlap_strategy = overlap_strategy
        self.duplicate_removal_strategy = duplicate_removal_strategy
        self.report_generator = ReportGenerator()
        self.images: List[Tile] = []
        self.overlap_map: Dict[str, List[str]] = {}

        self.stats = dict()

    def set_dataset(self, dataset: LabelingDataset):
        self.dataset = dataset
        self.stats["dataset_raw_stats"] = self.dataset.get_stats()

    def run(
        self,
        image_overlap_threshold: float = 0.0,
        detection_iou_threshold: float = 0.8,
        filepath="census_results.json",
    ):
        """Process image overlaps and remove duplicate detections"""

        logger.info("Finding overlapping images...")
        images = copy.deepcopy(self.dataset.tiles)
        self.overlap_map = self.overlap_strategy.find_overlapping_images(
            images, min_overlap_threshold=image_overlap_threshold
        )
        self.stats["overlap_stats"] = self.overlap_strategy.stats

        # detections to utm coords & geographic footprint
        for tile in self.dataset.tiles:
            tile.update_detection_gps()

        logger.info("Removing duplicate detections...")
        tiles = self.duplicate_removal_strategy.remove_duplicates(
            self.dataset.tiles, self.overlap_map, iou_threshold=detection_iou_threshold
        )

        # updating dataset with removed duplicates
        self.dataset.tiles = tiles
        self.dataset.build(force_rebuild=True)
        self.stats["dataset_pruned_stats"] = self.dataset.get_stats()

        file_path = filepath or "census_results.json"
        self.export_results(filepath=file_path)

    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive statistics about the detection process"""
        return self.stats

    def export_results(self, filepath: str):
        """Export results to JSON file"""
        results = {
            "metadata": {
                "processing_timestamp": datetime.now().isoformat(),
                "system_config": {
                    "overlap_strategy": type(self.overlap_strategy).__name__,
                    "duplicate_removal_strategy": type(
                        self.duplicate_removal_strategy
                    ).__name__,
                },
            },
            "statistics": self.get_statistics(),
        }

        if not Path(filepath).exists():
            Path(filepath).resolve().parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        data_path = Path(filepath).with_suffix(".csv").with_stem("dataset")
        self.dataset.save_data_csv(str(data_path))

        detections_path = Path(filepath).with_suffix(".csv").with_stem("detections")
        detections = self.dataset.export_detections_gps(save_path=str(detections_path))

        map_path = Path(filepath).with_suffix(".html").with_stem("map")
        self.report_generator.save_map_with_detections(
            detections,
            save_path=str(map_path),
        )

        logger.info(f"Results exported to {filepath}, {data_path}, {map_path}")


def get_overlap_strategy(overlap_strategy: str) -> OverlapStrategy:
    if overlap_strategy == "GPSOverlapStrategy":
        return GPSOverlapStrategy()
    else:
        raise ValueError(f"Invalid overlap strategy: {overlap_strategy}")


def get_duplicate_removal_strategy(
    duplicate_removal_strategy: str,
) -> DuplicateRemovalStrategy:
    if duplicate_removal_strategy == "CentroidProximityRemovalStrategy":
        return CentroidProximityRemovalStrategy()
    else:
        raise ValueError(
            f"Invalid duplicate removal strategy: {duplicate_removal_strategy}"
        )


def run_census(
    images_dir: list[str],
    engine: InferenceEngine,
    overlap_strategy: str,
    duplicate_removal_strategy: str,
    image_overlap_threshold: float = 0.0,
    detection_iou_threshold: float = 0.8,
    save_path: str = "census_results.json",
) -> WildlifeCountingSystem:
    assert isinstance(images_dir, list), "images_dir must be a list of strings"
    for a in images_dir:
        assert isinstance(a, str), (
            f"images_dir must be a list of strings, got {type(a)}"
        )

    dataset = LabelingDataset.from_dirs(images_dir)

    if engine is not None:
        dataset.add_predictions(engine, build=True)

    census_system = WildlifeCountingSystem(
        get_overlap_strategy(overlap_strategy),
        get_duplicate_removal_strategy(duplicate_removal_strategy),
    )
    census_system.set_dataset(dataset)
    census_system.run(image_overlap_threshold, detection_iou_threshold, save_path)

    return census_system


if __name__ == "__main__":
    # Example usage
    logger.info("Wildlife Detection System initialized")
