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
from collections import defaultdict

from .base import Tile, Detection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnimalType(Enum):
    """Enumeration of detectable animal types in savanna"""

    ELEPHANT = "elephant"
    ZEBRA = "zebra"
    GIRAFFE = "giraffe"
    WILDEBEEST = "wildebeest"
    LION = "lion"
    BUFFALO = "buffalo"
    UNKNOWN = "unknown"


@dataclass
class GPSCoordinate:
    """GPS coordinate with utility methods"""

    latitude: float
    longitude: float
    altitude: Optional[float] = None

    def distance_to(self, other: "GPSCoordinate") -> float:
        """Calculate distance in meters using Haversine formula"""
        R = 6371000  # Earth's radius in meters

        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        return R * c


class OverlapStrategy(ABC):
    """Abstract strategy for detecting overlapping images"""

    @abstractmethod
    def find_overlapping_images(self, images: List[DroneImage]) -> Dict[str, List[str]]:
        """Find overlapping images and return mapping of image_id -> overlapping_image_ids"""
        pass


class GPSOverlapStrategy(OverlapStrategy):
    """GPS-based overlap detection using geographic footprints"""

    def __init__(self, min_overlap_threshold: float = 0.1):
        self.min_overlap_threshold = min_overlap_threshold

    def find_overlapping_images(self, images: List[Tile]) -> Dict[str, List[str]]:
        overlap_map = defaultdict(list)

        for i, img1 in enumerate(images):
            footprint1 = img1.geographic_footprint

            for img2 in images[i + 1 :]:
                footprint2 = img2.geographic_footprint
                overlap_ratio = footprint1.overlap_ratio(footprint2)

                if overlap_ratio >= self.min_overlap_threshold:
                    overlap_map[img1.image_id].append(img2.image_id)
                    overlap_map[img2.image_id].append(img1.image_id)

        return dict(overlap_map)


class DuplicateRemovalStrategy(ABC):
    """Abstract strategy for removing duplicate detections"""

    @abstractmethod
    def remove_duplicates(
        self, detections: List[Detection], overlap_map: Dict[str, List[str]]
    ) -> List[Detection]:
        pass


class ConfidenceBasedRemovalStrategy(DuplicateRemovalStrategy):
    """Remove duplicates based on confidence scores and spatial proximity"""

    def __init__(
        self, spatial_threshold: float = 0.5, temporal_threshold: float = 300
    ):  # 5 minutes
        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold

    def remove_duplicates(
        self, detections: List[Detection], overlap_map: Dict[str, List[str]]
    ) -> List[Detection]:
        # Group detections by animal type
        grouped_detections = defaultdict(list)
        for detection in detections:
            grouped_detections[detection.animal_type].append(detection)

        unique_detections = []

        for animal_type, animal_detections in grouped_detections.items():
            unique_detections.extend(
                self._remove_duplicates_for_type(animal_detections, overlap_map)
            )

        return unique_detections

    def _remove_duplicates_for_type(
        self, detections: List[Detection], overlap_map: Dict[str, List[str]]
    ) -> List[Detection]:
        """Remove duplicates for a specific animal type"""
        unique_detections = []
        processed_ids = set()

        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        for detection in sorted_detections:
            if detection.detection_id in processed_ids:
                continue

            # Find potential duplicates in overlapping images
            duplicates = self._find_potential_duplicates(
                detection, detections, overlap_map
            )

            # Mark all duplicates as processed
            for dup in duplicates:
                processed_ids.add(dup.detection_id)

            # Keep the highest confidence detection
            best_detection = max(duplicates, key=lambda x: x.confidence)
            unique_detections.append(best_detection)
            processed_ids.add(detection.detection_id)

        return unique_detections

    def _find_potential_duplicates(
        self,
        detection: Detection,
        all_detections: List[Detection],
        overlap_map: Dict[str, List[str]],
    ) -> List[Detection]:
        """Find potential duplicate detections"""
        duplicates = [detection]

        # Get images that overlap with the detection's image
        overlapping_images = overlap_map.get(detection.image_id, [])

        for other_detection in all_detections:
            if (
                other_detection.detection_id != detection.detection_id
                and other_detection.animal_type == detection.animal_type
                and other_detection.image_id in overlapping_images
            ):
                # Check spatial proximity and temporal similarity
                if self._are_likely_same_object(detection, other_detection):
                    duplicates.append(other_detection)

        return duplicates

    def _are_likely_same_object(self, det1: Detection, det2: Detection) -> bool:
        """Determine if two detections are likely the same object"""
        # For now, we use bounding box IoU as a proxy for spatial proximity
        # In a real implementation, you'd project bounding boxes to geographic coordinates
        spatial_similarity = det1.bounding_box.iou(det2.bounding_box)

        # Check temporal proximity
        time_diff = abs((det1.timestamp - det2.timestamp).total_seconds())
        temporal_similarity = time_diff <= self.temporal_threshold

        return spatial_similarity >= self.spatial_threshold and temporal_similarity


class WildlifeCensusSystem:
    """Main system for wildlife detection and counting"""

    def __init__(
        self,
        overlap_strategy: OverlapStrategy,
        duplicate_removal_strategy: DuplicateRemovalStrategy,
    ):
        self.overlap_strategy = overlap_strategy
        self.duplicate_removal_strategy = duplicate_removal_strategy
        self.images: List[Tile] = []
        self.all_detections: List[Detection] = []
        self.unique_detections: List[Detection] = []
        self.overlap_map: Dict[str, List[str]] = {}

    def add_image(self, image: Tile):
        """Add a drone image to the system"""
        self.images.append(image)
        self.all_detections.extend(image.detections)
        logger.info(
            f"Added image {image.image_id} with {len(image.detections)} detections"
        )

    def process_overlaps(self):
        """Process image overlaps and remove duplicate detections"""
        logger.info("Processing image overlaps...")
        self.overlap_map = self.overlap_strategy.find_overlapping_images(self.images)

        total_overlaps = (
            sum(len(overlaps) for overlaps in self.overlap_map.values()) // 2
        )
        logger.info(f"Found {total_overlaps} overlapping image pairs")

        logger.info("Removing duplicate detections...")
        self.unique_detections = self.duplicate_removal_strategy.remove_duplicates(
            self.all_detections, self.overlap_map
        )

        logger.info(
            f"Reduced {len(self.all_detections)} detections to "
            f"{len(self.unique_detections)} unique detections"
        )

    def get_animal_counts(self) -> Dict[AnimalType, int]:
        """Get counts of each animal type"""
        counts = defaultdict(int)
        for detection in self.unique_detections:
            counts[detection.animal_type] += 1
        return dict(counts)

    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive statistics about the detection process"""
        return {
            "total_images": len(self.images),
            "total_raw_detections": len(self.all_detections),
            "total_unique_detections": len(self.unique_detections),
            "overlap_pairs": sum(
                len(overlaps) for overlaps in self.overlap_map.values()
            )
            // 2,
            "duplicate_removal_rate": 1
            - (len(self.unique_detections) / len(self.all_detections))
            if self.all_detections
            else 0,
            "animal_counts": self.get_animal_counts(),
            "average_confidence": np.mean(
                [d.confidence for d in self.unique_detections]
            )
            if self.unique_detections
            else 0,
        }

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
            "detections": [
                {
                    "detection_id": d.detection_id,
                    "animal_type": d.animal_type.value,
                    "confidence": d.confidence,
                    "image_id": d.image_id,
                    "timestamp": d.timestamp.isoformat(),
                    "bounding_box": {
                        "x1": d.bounding_box.x1,
                        "y1": d.bounding_box.y1,
                        "x2": d.bounding_box.x2,
                        "y2": d.bounding_box.y2,
                    },
                }
                for d in self.unique_detections
            ],
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results exported to {filepath}")


if __name__ == "__main__":
    # Example usage
    logger.info("Wildlife Detection System initialized")
