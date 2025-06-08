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

        self._labeled_dataset = dataset

        self.persistent = persistent

    def load_dataset(
        self,
    ):
        self.dataset = self._labeled_dataset.to_fiftyone(
            dataset_name=self.dataset_name, persistent=self.persistent
        )

    def run(
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
