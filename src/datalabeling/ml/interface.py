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


class Annotator(object):
    def __init__(
        self,
        dotenv_path: str = None,
        path_to_weights: str = None,
        mlflow_model_alias: str = "start",
        mlflow_model_name: str = "detector",
        tilesize: int = 640,
        overlapratio: float = 0.1,
        device: str = None,
        use_sliding_window: bool = True,
        is_yolo_obb: bool = False,
        confidence_threshold: float = 0.1,
        tag_to_append: str = "",
    ):
        """_summary_

        Args:
            path_to_weights (str, optional): path to weights. Defaults to None.
            mlflow_model_alias (str, optional): mflow registered model alias. Defaults to "start".
            mlflow_model_name (str, optional): model name. Defaults to "detector".
            confidence_threshold (float, optional): Detection threshold. Defaults to 0.35.
        """

        # Load environment variables
        if dotenv_path is not None:
            load_dotenv(dotenv_path=dotenv_path)
            # Connect to the Label Studio API and check the connection
            LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
            API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
            self.labelstudio_client = LabelStudio(
                base_url=LABEL_STUDIO_URL, api_key=API_KEY
            )
        else:
            logging.warning(
                msg="Pass argument `dotenv_path` to access label studio API"
            )

        ## Load model from path
        self.tilesize = tilesize
        self.overlapratio = overlapratio
        self.sahi_prostprocess = "NMS"

        self.path_to_weights = path_to_weights
        if self.path_to_weights is None:
            TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
            mlflow.set_tracking_uri(TRACKING_URI)
            client = mlflow.MlflowClient()
            name = mlflow_model_name
            alias = mlflow_model_alias
            version = client.get_model_version_by_alias(name=name, alias=alias).version
            self.modelversion = f"{name}:{version}" + tag_to_append
            self.modelURI = f"models:/{name}/{version}"
            self.model = mlflow.pyfunc.load_model(self.modelURI)
        else:
            self.model = Detector(
                path_to_weights=path_to_weights,
                confidence_threshold=confidence_threshold,
                overlap_ratio=self.overlapratio,
                tilesize=self.tilesize,
                device=device,
                use_sliding_window=use_sliding_window,
                is_yolo_obb=is_yolo_obb,
            )
            self.modelversion = Path(path_to_weights).stem + tag_to_append
        # LS label config
        self.from_name = "label"
        self.to_name = "image"
        self.label_type = "rectanglelabels"

    def predict(self, image: Image.Image) -> dict:
        """prediction using Sahi or not depending on self.use_sliding_window

        Args:
            image (Image.Image): image from PIL

        Returns:
            dict: prediction in coco annotation format
        """
        try:
            return self.model.predict(image, return_coco=True)
        except Exception:
            return self.model.predict(image)

    def predict_directory(
        self,
        path_to_dir: str = None,
        images_paths: list[str] = None,
        return_gps: bool = False,
        return_coco: bool = True,
        as_dataframe: bool = False,
        save_path: str = None,
    ):
        try:
            results = self.model.predict_directory(
                path_to_dir=path_to_dir,
                images_paths=images_paths,
                return_gps=return_gps,
                return_coco=return_coco,
                as_dataframe=as_dataframe,
                save_path=save_path,
            )

        except:
            results = (
                self.model.unwrap_python_model().detection_model.predict_directory(
                    path_to_dir=path_to_dir,
                    images_paths=images_paths,
                    return_gps=return_gps,
                    return_coco=return_coco,
                    as_dataframe=as_dataframe,
                    save_path=save_path,
                )
            )

        return results

    def format_prediction(self, pred: dict, img_height: int, img_width: int) -> dict:
        """_summary_

        Args:
            pred (dict): prediction in coco format
            img_height (int): image height
            img_width (int): image width

        Returns:
            dict: Label studio formated prediction
        """
        # formatting the prediction to work with Label studio
        x, y, width, height = pred["bbox"]
        label = pred["category_name"]
        score = pred["score"]
        if not isinstance(score, float):
            score = 0.0
        template = {
            "from_name": self.from_name,
            "to_name": self.to_name,
            "type": self.label_type,
            "original_width": img_width,
            "original_height": img_height,
            "image_rotation": 0,
            "value": {
                self.label_type: [
                    label,
                ],
                "x": x / img_width * 100,
                "y": y / img_height * 100,
                "width": width / img_width * 100,
                "height": height / img_height * 100,
                "rotation": 0,
            },
            "score": score,
        }
        return template

    def upload_predictions(self, project_id: int, top_n: int = 0) -> None:
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
                img_path = get_local_path(unquote(img_url), download_resources=False)
            except Exception:
                traceback.print_exc()
                img_path = get_local_path(img_url, download_resources=False)

            logger.info(f"Uploading predictions for: {img_path}")

            img = Image.open(img_path)
            prediction = self.predict(img)
            img_width, img_height = img.size
            formatted_pred = [
                self.format_prediction(pred, img_height=img_height, img_width=img_width)
                for pred in prediction
            ]
            conf_scores = [pred["score"] for pred in prediction]
            max_score = 0.0
            if len(conf_scores) > 0:
                max_score = max(conf_scores)

            self.labelstudio_client.predictions.create(
                task=task_id,
                score=max_score,
                result=formatted_pred,
                model_version=self.modelversion,
            )
            # project.create_prediction(
            #     task_id=task_id,
            #     score=max_score,
            #     result=formatted_pred,
            #     model_version=self.modelversion,
            # )

            img.close()

    def build_upload_json(
        self,
        path_img_dir: str,
        root: str,
        #   project_id:int=None,
        pattern="*.JPG",
        bulk_predictions: list[dict] = None,
        save_json_path: str = None,
    ) -> list[dict]:
        """Build Label studio json for data uploading

        Args:
            path_img_dir (str): directory with images of interest
            root (str): root specified in environment variable LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT or LOCAL_FILES_DOCUMENT_ROOT.
            pattern (str, optional): image extension pattern. It is passed to glob. Defaults to "*.JPG".
            bulk_predictions (dict, optional): dictionary with pre-computed detections. The key are the image file names. Defaults to None.
            save_json_path (str, optional): path to which the json file should be saved. Defaults to None.

        Returns:
            list[dict]: predictions in label sutdio format
        """
        directory_preds = list()

        # Upload predictions for each task
        for image_path in Path(path_img_dir).glob(pattern):
            img_path_as_bytes = bytes(image_path.relative_to(Path(root)))  # .as_posix()
            img_path_as_url = quote(img_path_as_bytes)

            img_url = f"/data/local-files/?d={img_path_as_url}"
            pred = {
                "data": {"image": img_url},
                "predictions": [],
            }
            image_path = Path(get_local_path(img_url))
            # get predictions
            if bulk_predictions is None:
                start = time()
                image = Image.open(image_path)
                predictions = self.predict(image)
                logger.info(f"Prediction time:{time() - start:.3f} seconds.")
                # format predictions
                img_width, img_height = image.size
                formatted_pred = [
                    self.format_prediction(
                        pred, img_height=img_height, img_width=img_width
                    )
                    for pred in predictions
                ]
            else:
                predictions = bulk_predictions[image_path.name]
                formatted_pred = [
                    self.format_prediction(
                        pred, img_height=pred["height"], img_width=pred["width"]
                    )
                    for pred in predictions
                ]
            conf_scores = [pred["score"] for pred in predictions]
            # store predictions
            if len(conf_scores) > 0:
                pred["predictions"].append(
                    {
                        "result": formatted_pred,
                        "model_version": self.modelversion,
                        "score": max(conf_scores),
                    }
                )
            else:
                pred["predictions"].append(
                    {
                        "result": formatted_pred,
                        "model_version": self.modelversion,
                        "score": 0.0,
                    }
                )
            # update buffer
            directory_preds.append(pred)

        if save_json_path is not None:
            with open(Path(save_json_path), "w") as file:
                json.dump(directory_preds, file, indent=2)

        return directory_preds

    @staticmethod
    def get_project_stats(
        labelstudio_client: LabelStudio, project_id: int, annotator_id=0
    ):
        project = labelstudio_client.projects.get(id=project_id)

        images_count = dict()

        # Iterating
        tasks = labelstudio_client.tasks.list(
            project=project.id,
        )
        labels = []
        for task in tasks:
            try:
                result = task.annotations[annotator_id]["result"]
            except Exception:
                # traceback.print_exc()
                continue

            img_labels = []
            for annot in result:
                img_labels = annot["value"]["rectanglelabels"] + img_labels
            labels = labels + img_labels

            # update stats holder
            for label in set(img_labels):
                if label in images_count.keys():
                    images_count[label] += 1
                else:
                    images_count[label] = 1

        instances = {
            f"{k}": [
                labels.count(k),
            ]
            for k in set(labels)
        }
        images_count = {
            k: [
                v,
            ]
            for k, v in images_count.items()
        }

        instances_count = pd.DataFrame.from_dict(instances)
        images_count = pd.DataFrame.from_dict(images_count)

        return instances_count, images_count
