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

    def create_load_dataset(
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
        raise NotImplementedError()
        # self.load_dataset()

        # self.logger.info(f"Dataset contains {len(self.dataset)} samples")

        # # Create a session
        # session = fo.launch_app(self.dataset, port=port, auto=auto_launch)

        # session.wait()


# TODO debug
class FiftyOneLabelStudioIntegration:
    """
    A complete integration class for FiftyOne and Label Studio workflows.
    """

    def __init__(self, dataset_name: str):
        """
        Initialize the integration with a FiftyOne dataset.

        Args:
            dataset_name: Name of the FiftyOne dataset
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self._load_or_create_dataset()

    def _load_or_create_dataset(self):
        """Load existing or create new dataset."""
        try:
            self.dataset = fo.load_dataset(self.dataset_name)
            print(f"Loaded existing dataset: {self.dataset_name}")
        except ValueError:
            print(f"Dataset {self.dataset_name} not found. Please create it first.")

    def setup_label_studio_connection(
        self, api_key: str = None, server_url: str = "http://localhost:8080"
    ):
        """
        Setup Label Studio connection parameters.

        Args:
            api_key: Label Studio API key (can also use environment variable)
            server_url: Label Studio server URL
        """
        if api_key:
            os.environ["FIFTYONE_LABELSTUDIO_API_KEY"] = api_key

        if server_url:
            os.environ["FIFTYONE_LABELSTUDIO_URL"] = server_url

        print(f"Label Studio configured for: {server_url}")
        print(
            "API key configured via environment variable"
            if api_key
            else "Using existing API key from environment"
        )

    def create_annotation_project(
        self,
        anno_key: str,
        label_schema: dict,
        view: fo.DatasetView = None,
        project_name: str = None,
        launch_editor: bool = True,
    ):
        """
        Create annotation project in Label Studio.

        Args:
            anno_key: Unique identifier for this annotation run
            label_schema: Dictionary defining the labeling schema
            view: Specific view of dataset to annotate (if None, uses full dataset)
            project_name: Name for the Label Studio project
            launch_editor: Whether to automatically launch Label Studio editor
        """
        target_view = view if view is not None else self.dataset

        try:
            target_view.annotate(
                anno_key,
                backend="labelstudio",
                label_schema=label_schema,
                project_name=project_name,
                launch_editor=launch_editor,
            )

            print(f"✅ Annotation project created with key: {anno_key}")
            print(f"📊 Annotation info:")
            print(self.dataset.get_annotation_info(anno_key))

        except Exception as e:
            print(f"❌ Error creating annotation project: {e}")
            raise

    def load_annotations_from_labelstudio(
        self, anno_key: str, dest_field: str = None, cleanup: bool = False
    ):
        """
        Load completed annotations from Label Studio back into FiftyOne.

        Args:
            anno_key: Unique identifier for the annotation run
            dest_field: Optional destination field name (overrides schema)
            cleanup: Whether to clean up Label Studio tasks after loading
        """
        try:
            self.dataset.load_annotations(
                anno_key, dest_field=dest_field, cleanup=cleanup
            )

            print(f"✅ Annotations loaded successfully for run: {anno_key}")

            if cleanup:
                print("🧹 Label Studio tasks cleaned up")

        except Exception as e:
            print(f"❌ Error loading annotations: {e}")
            raise

    def get_annotation_status(self, anno_key: str):
        """Get status information about an annotation run."""
        try:
            info = self.dataset.get_annotation_info(anno_key)
            results = self.dataset.load_annotation_results(anno_key)

            print(f"📊 Annotation Run Status: {anno_key}")
            print(f"Backend: {info.get('backend', 'N/A')}")
            print(f"Label Schema: {info.get('label_schema', {})}")

            return {"info": info, "results": results}

        except Exception as e:
            print(f"❌ Error getting annotation status: {e}")
            return None
