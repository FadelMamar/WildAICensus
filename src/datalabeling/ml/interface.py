import logging
import os
import traceback
from pathlib import Path
from time import time
from urllib.parse import quote, unquote
import pandas as pd
from dotenv import load_dotenv
from typing import Sequence

# from label_studio_ml.utils import get_local_path
from label_studio_tools.core.utils.io import get_local_path

from label_studio_sdk.client import LabelStudio
from PIL import Image

from .workers import ObjectDetectionSystem, GPSUtils
from .models import ImageClassifier, YOLO
from ..common.processor import DetectionsPostprocessor, get_processor, FeatureExtractor
from ..common.config import PredictionConfig
from ..common.base import Detection, Tile
from ..common.mlflow_utils import load_registered_model


logger = logging.getLogger(__name__)


class InferenceEngine(object):
    def __init__(self, config: PredictionConfig):
        self.config = config

        self.detector = None
        self.image_processor = None
        self.detection_processor = None
        self.model_tag = "None"

        # LS label config
        self.labelstudio_client: LabelStudio = None
        self.from_name = "label"
        self.to_name = "image"
        self.label_type = "rectanglelabels"

    def set_detector(self, detector: ObjectDetectionSystem, model_tag: str):
        assert isinstance(detector, ObjectDetectionSystem), (
            "Received {type(detector)} instead of ObjectDetectionSystem"
        )
        self.detector = detector
        self.model_tag = model_tag

    def set_processor(
        self,
        detection_processor: DetectionsPostprocessor = None,
    ):
        self.detector.set_processor(roi_processor=detection_processor)

    def set_ls_client(self, dotenv_path: str):
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

        if LABEL_STUDIO_URL is None:
            raise ValueError("env variable LABEL_STUDIO_URL is not set.")
        if API_KEY is None:
            raise ValueError("env variable API_KEY is not set.")

        self.labelstudio_client = LabelStudio(
            base_url=LABEL_STUDIO_URL, api_key=API_KEY
        )

        return None

    def inference(
        self,
        images_paths: Sequence[str],
        tiles: list[Tile] = None,
        return_tiles: bool = False,
        return_as_df: bool = False,
    ) -> list[Detection] | list[Tile] | pd.DataFrame:
        """Multithreaded detector"""

        paths = images_paths
        if paths is None:
            paths = [t.image_path for t in tiles]

        detections = self.detector.run(images_paths=paths)

        if tiles is not None:
            # if tiles are provided,
            for i, tile in enumerate(tiles):
                tile.predictions = detections[i]

        if return_tiles:
            assert tiles is not None, (
                "This is likely an errro. tiles have not been set."
            )
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

    def _upload_single_task(self, task, tag: str = ""):
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

        except Exception:
            traceback.print_exc()
            logger.warning(f"Failed to load {img_path}. Skipping...")

        return None

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

            self._upload_single_task(task=task, tag=tag)


def load_engine(
    pred_config: PredictionConfig,
    roi_classifier_path: str = r"..\base_models_weights\roi_classifier.ckpt",
    roi_cls_is_features: bool = True,
    roi_cls_label_map: dict = {0: "gt", 1: "tn"},
    roi_keep_classes: list = ["gt"],
    detection_label_map: dict = None,  # {0: "wildlife"},
    feature_extractor_path: str = "facebook/dinov2-with-registers-small",
    detection_model: YOLO = None,
    mlflow_model_alias: str = "demo",
    mlflow_model_name: str = "labeler",
    set_ls_client: bool = False,
    dot_env_path: str = None,
) -> tuple[InferenceEngine, FeatureExtractor]:
    if detection_model is None:
        logger.info(
            f"Loading model from mlflow name={mlflow_model_name}/alias={mlflow_model_alias} "
        )
        detection_model, metadata = load_registered_model(
            alias=mlflow_model_alias,
            name=mlflow_model_name,
            mlflow_tracking_url="http://localhost:5000",
            load_unwrapped=True,
        )
        logger.info(f"model's metadata={metadata}")

    # build roi postprocessor
    feature_extractor = get_processor("feature_extractor")(
        hf_model_path=feature_extractor_path
    )

    roi_processor = None
    try:
        model = ImageClassifier.load_from_checkpoint(
            roi_classifier_path,
            cls_is_features=roi_cls_is_features,
            map_location=pred_config.device,
        )

        roi_classifier = get_processor("classifier")(
            model,
            label_map=roi_cls_label_map,
            device=pred_config.device,
            feature_extractor=feature_extractor,
            imgsz=pred_config.cls_imgsz,
        )
        roi_processor = DetectionsPostprocessor(
            keep_classes=roi_keep_classes,
        )
        roi_processor.set_classifier(roi_classifier)
    except:
        traceback.print_exc()
        logger.warning("Roi classifier is not loaded.")

    # build object detection system
    detection_label_map = (
        getattr(detection_model, "names", None)
        or detection_label_map
        or {0: "wildlife"}
    )
    detector = ObjectDetectionSystem(
        config=pred_config, detection_label_map=detection_label_map
    )
    detector.set_model(model=detection_model, task="detect", path_weights=None)
    detector.set_processor(roi_processor=roi_processor)

    engine = InferenceEngine(config=pred_config)
    engine.set_detector(detector=detector, model_tag=mlflow_model_alias)

    if set_ls_client:
        engine.set_ls_client(dotenv_path=dot_env_path)

    return engine, feature_extractor
