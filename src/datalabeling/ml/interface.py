import json
import logging
import os
import traceback
from pathlib import Path
from time import time
from urllib.parse import quote, unquote
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv

# from label_studio_ml.utils import get_local_path
from label_studio_tools.core.utils.io import get_local_path

from label_studio_sdk.client import LabelStudio
from PIL import Image
from tqdm import tqdm

from ..common.processor import DetectionsPostprocessor, Processor
from ..common.config import PredictionConfig
from ..common.base import Detection, Tile
from .models import Detector
from ..common.mlflow_utils import load_registered_model
import torch

logger = logging.getLogger(__name__)


class InferenceEngine(object):
    def __init__(self, config: PredictionConfig):
        self.config = config

        self.detector = None
        self.image_processor = None
        self.detection_processor: DetectionsPostprocessor = None
        self.model_tag = "None"

    def set_detector(self, detector: Detector, model_tag: str):
        self.detector = detector
        self.model_tag = model_tag

    def set_processor(
        self,
        image_processor: Processor = None,
        detection_processor: DetectionsPostprocessor = None,
    ):
        self.image_processor = image_processor
        self.detection_processor = detection_processor

        for p in [self.detection_processor, self.image_processor]:
            if isinstance(p, torch.nn.Module):
                p.eval()

    def inference(
        self,
        tile: Tile = None,
    ) -> list[Detection]:
        assert tile.image_data is not None, "define 'image_data' field"

        if self.image_processor:
            tile.image_data = self.image_processor.run(tile.image_data)

        detections = self.detector.predict(
            tile=tile,
        )

        if len(detections) < 1:
            return []

        if self.detection_processor:
            detections = self.detection_processor.run(
                detections, image=tile.image_data, box_size=self.config.cls_imgsz
            )

        return detections

    def batch_inference(
        self,
        images_paths: list[str],
        tiles: list[Tile] = None,
        save_path: str = None,
        as_dataframe: bool = False,
        return_tiles: bool = False,
    ) -> pd.DataFrame | list[Tile]:
        results = {}

        logger.info("Batch inference...")

        if tiles:
            assert images_paths is None
            for tile in tqdm(tiles, desc="Batch inference..."):
                detections = self.inference(
                    image_path=None,
                    tile=tile,
                    image=None,
                )
                tile.predictions = detections
                assert tile.image_path, "tile.image_path is not defined!"
                # results[str(tile.image_path)] = detections
            if return_tiles:
                return tiles

        else:
            for path in tqdm(images_paths, desc="Batch inference..."):
                detections = self.inference(
                    image_path=path,
                    tile=None,
                    image=None,
                )
                results[str(path)] = detections

        if as_dataframe or save_path:
            results = self.detector._format_results_as_dataframe(results=results)
            if save_path:
                results.to_csv(save_path, index=False)

        return results


class Annotator(InferenceEngine):
    def __init__(
        self,
        config: PredictionConfig,
        dotenv_path: str = None,
    ):
        super().__init__(config)

        # # Load environment variables
        if dotenv_path is not None:
            load_dotenv(dotenv_path=dotenv_path)
        else:
            logging.warning(
                msg="Pass argument `dotenv_path` to access label studio API"
            )

        # # label studio client
        LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
        API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
        self.labelstudio_client = LabelStudio(
            base_url=LABEL_STUDIO_URL, api_key=API_KEY
        )

        # LS label config
        self.from_name = "label"
        self.to_name = "image"
        self.label_type = "rectanglelabels"

    def upload_predictions(
        self,
        project_id: int,
        top_n: int = 0,  # download_resources: bool = True,
        tag: str = "",
    ) -> None:
        """Uploads predictions using label studio API.
        Make sure to set the API key and url inside .env

        Args:
            project_id (int): project id from Label studio
            top_n (int): top n tasks to be uploaded in descending order of task_id. Default 0 which disables the feature.
        """
        # Select project
        project = self.labelstudio_client.projects.get(id=project_id)

        # Upload predictions for each task
        tasks = self.labelstudio_client.tasks.list(
            project=project.id,
        )
        for i, task in enumerate(tasks):
            if top_n > 0:
                if i > top_n:
                    break

            task_id = task.id
            img_url = task.data["image"]

            try:
                # using unquote to deal with special characters
                img_path = get_local_path(
                    unquote(img_url),
                    download_resources=False,
                    hostname=os.getenv("LABEL_STUDIO_URL"),
                )

                if not Path(img_path).exists():
                    img_path = get_local_path(
                        unquote(img_url),
                        download_resources=True,
                        hostname=os.getenv("LABEL_STUDIO_URL"),
                    )

            except Exception:
                traceback.print_exc()
                logger.warn(f"Failed to load {img_path}. Skipping...")
                continue

            logger.info(f"Uploading predictions for: {img_path}")

            img = Image.open(img_path)
            predictions = self.inference(image=img)

            img_width, img_height = img.size

            formatted_pred = [
                pred.to_ls(
                    from_name=self.from_name,
                    to_name=self.to_name,
                    label_type=self.label_type,
                    img_height=img_height,
                    img_width=img_width,
                )
                for pred in predictions
            ]
            conf_scores = [pred["score"] for pred in formatted_pred]
            max_score = 0.0
            if len(conf_scores) > 0:
                max_score = max(conf_scores)

            self.labelstudio_client.predictions.create(
                task=task_id,
                score=max_score,
                result=formatted_pred,
                model_version=self.model_tag + tag,
            )

            img.close()
