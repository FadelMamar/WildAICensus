import logging
import os
import traceback
from pathlib import Path
from time import time
from urllib.parse import quote, unquote
import pandas as pd
from dotenv import load_dotenv

# from label_studio_ml.utils import get_local_path
from label_studio_tools.core.utils.io import get_local_path

from label_studio_sdk.client import LabelStudio
from PIL import Image

from .workers import ObjectDetectionSystem, GPSUtils
from ..common.processor import DetectionsPostprocessor
from ..common.config import PredictionConfig
from ..common.base import Detection, Tile


logger = logging.getLogger(__name__)


class InferenceEngine(object):
    def __init__(self, config: PredictionConfig):
        self.config = config

        self.detector = None
        self.image_processor = None
        self.detection_processor = None
        self.model_tag = "None"

    def set_detector(self, detector: ObjectDetectionSystem, model_tag: str):
        assert isinstance(detector, ObjectDetectionSystem)
        self.detector = detector
        self.model_tag = model_tag

    def set_processor(
        self,
        detection_processor: DetectionsPostprocessor = None,
    ):
        self.detector.set_processor(roi_processor=detection_processor)

    def inference(
        self,
        images_paths: list[str],
        tiles: list[Tile] = None,
        return_tiles: bool = False,
        return_as_df: bool = False,
    ) -> list[Detection]:
        assert isinstance(images_paths, list)
        """Multithreaded detector"""

        paths = images_paths
        if paths is None:
            paths = [t.image_path for t in tiles]

        self.detector.run(images_paths=paths)
        detections = [out["final_detections"] for out in self.detector.outputs]

        if tiles is not None:
            # if tiles are provided,
            for i, tile in enumerate(tiles):
                tile.predictions = detections[i]

        if return_tiles:
            return tiles

        if return_as_df:
            results = {}
            for i, image_path in enumerate(paths):
                results.update({str(image_path): detections[i]})
            return self._format_results_as_dataframe(results)

        return detections

    def _format_results_as_dataframe(
        self, results: dict[str, list[Detection]]
    ) -> pd.DataFrame:
        if len(results) < 1:
            return pd.DataFrame()

        unravel_dict = []
        for img_path, detections in results.items():
            if len(detections) < 1:
                continue

            for det in detections:
                unravel_dict.append(dict(file_name=img_path, **det.to_dict()))

        dfs = pd.DataFrame.from_dict(unravel_dict)

        dfs["bbox_w"] = dfs["x_max"] - dfs["x_min"]
        dfs["bbox_h"] = dfs["y_max"] - dfs["y_min"]

        # converting gps coords to decimal
        dfs[["img_Latitude", "img_Longitude", "img_Elevation"]] = (
            dfs["image_gps_loc"].apply(GPSUtils.to_decimal).apply(pd.Series)
        )
        dfs[["Latitude", "Longitude", "Elevation"]] = (
            dfs["gps_loc"].apply(GPSUtils.to_decimal).apply(pd.Series)
        )

        return dfs


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

            predictions = self.inference(
                images_paths=[img_path],
                tiles=None,
                return_tiles=False,
                return_as_df=False,
            )

            img_width, img_height = Image.open(img_path).size

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
