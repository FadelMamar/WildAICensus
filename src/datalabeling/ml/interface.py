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
from label_studio_ml.utils import get_local_path
from label_studio_sdk.client import LabelStudio
from PIL import Image
from tqdm import tqdm

from ..common.processor import Processor
from ..common.config import Detection, PredictionConfig
from .models import Detector
from ..common.mlflow_utils import load_registered_model

logger = logging.getLogger(__name__)


class InferenceEnginge(object):
    def __init__(self, config: PredictionConfig):
        self.config = config

        self.detector = None
        self.image_processor = None
        self.detection_processor = None

    def set_detector(self, detector: Detector):
        self.detector = detector

    def set_processor(
        self, image_processor: Processor = None, detection_processor: Processor = None
    ):
        self.image_processor = image_processor
        self.detection_processor = detection_processor

    def inference(
        self,
        image_path: str = None,
        image: Image.Image = None,
        inference_service_url=None,
    ) -> list[Detection]:
        if image_path:
            image = Image.open(image_path)
            image_path = None

        if self.image_processor:
            image = self.image_processor.run(np.asarray(image))
            image = Image.fromarray(image)

        detections = self.detector.predict(
            image=image,
            image_path=image_path,
            inference_service_url=inference_service_url,
            override_tilesize=self.config.tilesize,
        )

        if self.detection_processor:
            cfg = dict(image=image, box_size=self.config.cls_imgsz)
            detections = self.detection_processor.run(detections, **cfg)

        return detections

    def batch_inference(
        self,
        images_paths: list[str],
        save_path: str = None,
        as_dataframe: bool = False,
        inference_service_url=None,
    ) -> pd.DataFrame:
        results = {}

        for image_path in images_paths:
            detections = self.predict(
                image_path=image_path,
                image=None,
                inference_service_url=inference_service_url,
            )
            results[str(image_path)] = detections

        if as_dataframe or save_path:
            results = self.detector._format_results_as_dataframe(results=results)
            if save_path:
                results.to_csv(save_path, index=False)

        return results


class Annotator(InferenceEnginge):
    def __init__(
        self,
        config: PredictionConfig,
        dotenv_path: str = None,
        path_to_weights: str = None,
        mlflow_model_alias: str = "default",
        mlflow_model_name: str = "labeler",
        tag="",
    ):
        super().__init__(config)

        # Load environment variables
        if dotenv_path is not None:
            load_dotenv(dotenv_path=dotenv_path)
        else:
            logging.warning(
                msg="Pass argument `dotenv_path` to access label studio API"
            )

        # label studio client
        LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
        API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
        self.labelstudio_client = LabelStudio(
            base_url=LABEL_STUDIO_URL, api_key=API_KEY
        )

        print(API_KEY)

        if path_to_weights is None:
            TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
            mlflow.set_tracking_uri(TRACKING_URI)
            model, self.modelversion = load_registered_model(
                alias=mlflow_model_alias,
                name=mlflow_model_name,
                mlflow_tracking_url=TRACKING_URI,
                tag_to_append=tag,
            )

            self.detector = Detector(
                config=config,
                detection_model=model.unwrap_python_model().detection_model,
            )

        else:
            self.detector = Detector(config=config, detection_model=None)
            self.detector.set_detection_model(
                detection_model=None, path_to_weights=path_to_weights
            )
            self.modelversion = Path(path_to_weights).stem + tag

        # LS label config
        self.from_name = "label"
        self.to_name = "image"
        self.label_type = "rectanglelabels"

    def upload_predictions(
        self, project_id: int, top_n: int = 0, download_resources: bool = True
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
                    unquote(img_url), download_resources=download_resources
                )
            except Exception:
                traceback.print_exc()
                img_path = get_local_path(
                    img_url, download_resources=download_resources
                )

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
                model_version=self.modelversion,
            )

            img.close()
