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

from .workers import ObjectDetectionSystem
from .models import ImageClassifier, Detector, build_detector
from ..common.processor import DetectionsPostprocessor, get_processor
from ..common.config import PredictionConfig
from ..common.base import Detection, Tile
from ..common.mlflow_utils import load_registered_model
from ..common.annotation_utils import GPSUtils

logger = logging.getLogger(__name__)


class InferenceEngine(object):
    def __init__(self, config: PredictionConfig):
        """
        Initialize the InferenceEngine with a prediction configuration.

        Args:
            config (PredictionConfig): Prediction configuration object.
        """
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
        """
        Set the object detection system and model tag for inference.

        Args:
            detector (ObjectDetectionSystem): The detection system to use.
            model_tag (str): Tag or version of the model.
        """
        assert isinstance(detector, ObjectDetectionSystem), (
            "Received {type(detector)} instead of ObjectDetectionSystem"
        )
        self.detector = detector
        self.model_tag = model_tag

    def set_processor(
        self,
        detection_processor: DetectionsPostprocessor = None,
    ):
        """
        Set the detection postprocessor for ROI or other post-processing.

        Args:
            detection_processor (DetectionsPostprocessor, optional): Postprocessor to use.
        """
        self.detector.set_processor(roi_processor=detection_processor)

    def set_ls_client(self, dotenv_path: str):
        """
        Set up the Label Studio client using environment variables from a .env file.

        Args:
            dotenv_path (str): Path to the .env file containing API credentials.
        """
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
    ) -> Sequence[list[Detection]] | list[Tile] | pd.DataFrame:
        """
        Run multithreaded inference on a list of image paths or tiles.

        Args:
            images_paths (Sequence[str]): List of image paths to run inference on.
            tiles (list[Tile], optional): List of Tile objects to update with predictions.
            return_tiles (bool, optional): If True, return tiles with predictions.
            return_as_df (bool, optional): If True, return results as a DataFrame.

        Returns:
            Sequence[list[Detection]] | list[Tile] | pd.DataFrame: Detections, tiles, or DataFrame depending on arguments.
        """

        if tiles is None:
            tiles = [
                Tile(image_path=p, flight_specs=self.config.flight_specs)
                for p in images_paths
            ]

        logger.info(f"Running inference on {len(tiles)} tiles.")

        detections = self.detector.run(tiles=tiles)

        if len(detections) != len(tiles):
            raise ValueError(
                "Number of detections does not match number of images. {} != {}".format(
                    len(detections), len(tiles)
                )
            )

        if tiles is not None:
            for i, tile in enumerate(tiles):
                tile.set_predictions(detections[i])

        if return_tiles:
            assert tiles is not None, (
                "This is likely an error. 'tiles' is not provided."
            )
            return tiles

        if return_as_df:
            results = {}
            for i, tile in enumerate(tiles):
                results.update({str(tile.image_path): detections[i]})
            return self._format_results_as_dataframe(results)

        return detections

    def _format_results_as_dataframe(
        self, results: dict[str, list[Detection]]
    ) -> pd.DataFrame:
        """
        Convert detection results to a pandas DataFrame, including bounding box and GPS info.

        Args:
            results (dict[str, list[Detection]]): Mapping from image path to detections.

        Returns:
            pd.DataFrame: DataFrame with detection and image metadata.
        """
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
        """
        Upload predictions for a single Label Studio task.

        Args:
            task: Label Studio task object.
            tag (str, optional): Additional tag for the model version.
        """
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
        """
        Upload predictions to Label Studio for a given project.

        Args:
            project_id (int): Project ID in Label Studio.
            top_n (int, optional): Only upload for the top N tasks. Default 0 (all).
            tag (str, optional): Additional tag for the model version.
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

    @classmethod
    def load_engine(
        cls,
        pred_config: PredictionConfig,
        roi_classifier_path: str = None,
        roi_cls_is_features: bool = True,
        roi_cls_label_map: dict = {0: "gt", 1: "tn"},
        roi_keep_classes: list = ["gt"],
        detection_label_map: dict = {0: "wildlife"},
        feature_extractor_path: str = "facebook/dinov2-with-registers-small",
        model_path: str = None,
        detection_model_type: str = "ultralytics",
        text_instruction: str = "detect wildlife species",
        mlflow_model_alias: str = "demo",
        mlflow_model_name: str = "labeler",
        set_ls_client: bool = False,
        buffer_size=256,
        timeout=300,
        dot_env_path: str = None,
    ) -> tuple:
        """
        Load and configure an InferenceEngine with all required models and processors.

        Args:
            pred_config (PredictionConfig): Prediction configuration.
            roi_classifier_path (str, optional): Path to ROI classifier checkpoint.
            roi_cls_is_features (bool, optional): Whether ROI classifier uses features.
            roi_cls_label_map (dict, optional): Label map for ROI classifier.
            roi_keep_classes (list, optional): Classes to keep after ROI classification.
            detection_label_map (dict, optional): Label map for detection model.
            feature_extractor_path (str, optional): Path or name of feature extractor.
            model_path (str, optional): Path to detection model weights.
            detection_model_type (str, optional): Type of detection model.
            text_instruction (str, optional): Instruction for detection model.
            mlflow_model_alias (str, optional): MLflow model alias.
            mlflow_model_name (str, optional): MLflow model name.
            set_ls_client (bool, optional): Whether to set up Label Studio client.
            dot_env_path (str, optional): Path to .env file for Label Studio.

        Returns:
            tuple: (InferenceEngine, feature_extractor)
        """
        if (model_path is None) and (pred_config.inference_service_url is None):
            logger.info(
                f"Loading model from mlflow name={mlflow_model_name}/alias={mlflow_model_alias} "
            )
            model, metadata = load_registered_model(
                alias=mlflow_model_alias,
                name=mlflow_model_name,
                mlflow_tracking_url="http://localhost:5000",
                load_unwrapped=True,
            )
            logger.info(f"model's metadata={metadata}")

            logger.info(f"{model.__class__.__name__} loaded successfully.")

            pred_config.batch_size = metadata.get("batch", pred_config.batch_size)
            pred_config.tilesize = metadata.get("tilesize", pred_config.tilesize)
            pred_config.imgsz = pred_config.tilesize

            detection_model = build_detector(
                detection_model_type=metadata["detection_model_type"],
                model_path=None,
                model=model,
                config=pred_config,
                text_instruction=text_instruction,
            )
        else:
            detection_model = build_detector(
                detection_model_type=detection_model_type,
                model_path=model_path,
                model=None,
                config=pred_config,
                text_instruction=text_instruction,
            )

        # build roi postprocessor
        feature_extractor = get_processor("feature_extractor")(
            hf_model_path=feature_extractor_path
        )

        roi_processor = None
        if roi_classifier_path is not None:
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

        # build object detection system
        detection_label_map = (
            getattr(detection_model, "names", None)
            or detection_label_map
            or {0: "wildlife"}
        )
        detector = ObjectDetectionSystem(
            config=pred_config,
            detection_label_map=detection_label_map,
            buffer_size=buffer_size,
            timeout=timeout,
        )
        detector.set_model(model=detection_model)
        detector.set_processor(roi_processor=roi_processor)

        engine = cls(config=pred_config)
        engine.set_detector(detector=detector, model_tag=mlflow_model_alias)

        if set_ls_client:
            engine.set_ls_client(dotenv_path=dot_env_path)

        return engine, feature_extractor
