import fiftyone as fo
import fiftyone.zoo as foz
import os
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time

from .dataset_loader import LabelingDataset


class FiftyOneVisualizer:
    """
    A visualization module for images and annotations using FiftyOne.
    Supports various annotation types including bounding boxes, classifications, and segmentations.
    """

    def __init__(
        self,
        dataset: LabelingDataset,
        dataset_name: str = "visualization_dataset",
        persistent: bool = True,
    ):
        """
        Initialize the visualizer with a dataset name.

        Args:
            dataset_name: Name for the FiftyOne dataset
        """
        self.logger = logging.getLogger("FiftyOneVisualizer")

        self.dataset_name = dataset_name
        self.dataset = None

        self._data: pd.DataFrame = dataset.data
        self._labeled_dataset = dataset

        self.persistent = persistent

        self._setup_dataset()

    def _setup_dataset(self):
        """Setup or load the FiftyOne dataset."""
        try:
            # Try to load existing dataset
            self.dataset = fo.load_dataset(self.dataset_name)
            self.logger.info(f"Loaded existing dataset: {self.dataset_name}")
        except ValueError:
            # Create new dataset if it doesn't exist
            self.dataset = fo.Dataset(self.dataset_name, persistent=self.persistent)
            self.logger.info(f"Created new dataset: {self.dataset_name}")

    def add_images(
        self,
    ):
        """
        Add images with object detection annotations.
        """
        samples = []

        # df_annotations = self._data[self._data['is_annot'] == True]

        with fo.ProgressBar() as pb:
            for img_path, df_detections in pb(self._data.groupby("file_name")):
                if not os.path.exists(img_path):
                    self.logger.warning(
                        f"Warning: Image not found at {img_path}.Skipping"
                    )
                    continue

                sample = fo.Sample(filepath=img_path)

                if df_detections.dropna().empty:
                    self.logger.info(f"Image at {img_path} is a negative sample")
                    sample["is_positive"] = False
                    samples.append(sample)
                    continue

                # conf = bboxes['scores']
                class_name = df_detections["class"].tolist()

                # get bbox [x, y, w, h] in normalized coords [0,1]
                bboxes = df_detections[["x_min", "y_min"]].copy()
                bboxes.loc[:, "x_min"] /= df_detections.loc[:, "width"]
                bboxes.loc[:, "y_min"] /= df_detections.loc[:, "height"]
                bboxes["w"] = 0.0
                bboxes["h"] = 0.0
                bboxes.loc[:, "w"] = (
                    df_detections.loc[:, "x_max"] - df_detections.loc[:, "x_min"]
                ) / df_detections.loc[:, "width"]
                bboxes.loc[:, "h"] = (
                    df_detections.loc[:, "y_max"] - df_detections.loc[:, "y_min"]
                ) / df_detections.loc[:, "height"]
                bboxes = bboxes[["x_min", "y_min", "w", "h"]].to_numpy().tolist()

                # Convert detections to FiftyOne format
                fo_detections = []
                for i, box in enumerate(bboxes):
                    fo_detection = fo.Detection(
                        label=class_name[i],
                        bounding_box=box,
                        # confidence=det.get('confidence', None)
                    )
                    fo_detections.append(fo_detection)

                sample["gt"] = fo.Detections(detections=fo_detections)
                sample["is_positive"] = True
                samples.append(sample)

            self.dataset.add_samples(samples)
        self.logger.info(f"Added {len(samples)} samples.")

    def _run(
        self,
        port: int = 5151,
        auto_launch: bool = True,
        view_name: Optional[str] = None,
    ):
        """
        Launch the FiftyOne visualization interface.

        Args:
            port: Port number for the web interface
            auto_launch: Whether to automatically open browser
            view_name: Optional name for a specific view of the data
        """

        self.add_images()

        self.logger.info(f"Dataset contains {len(self.dataset)} samples")

        # Create a session
        session = fo.launch_app(self.dataset, port=port, auto=auto_launch)

        session.wait()
