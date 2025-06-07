from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Sequence
import os, json
import logging
import torch
from pathlib import Path
import mlflow
from ultralytics.engine.results import Results as UltralyticsResults
from torchvision.transforms import PILToTensor
from torchvision.ops import nms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset

from fastapi import HTTPException


def load_registered_model(
    alias,
    name,
    load_unwrapped: bool = True,
):
    client = mlflow.MlflowClient()

    version = client.get_model_version_by_alias(name=name, alias=alias).version
    modelversion = f"{name}:{version}"
    modelURI = f"models:/{name}/{version}"

    dwnd_location = Path(os.environ.get("WEIGHTS_PATH", "./model_weights")) / f"{name}"
    dwnd_location = dwnd_location / str(version)
    if dwnd_location.exists():
        model = mlflow.pyfunc.load_model(str(dwnd_location))
    else:
        dwnd_location.mkdir(parents=True, exist_ok=True)
        model = mlflow.pyfunc.load_model(modelURI, dst_path=str(dwnd_location))

    metadata = dict(version=modelversion)
    metadata.update(model.metadata.metadata)

    if load_unwrapped:
        model = model.unwrap_python_model().model

    return model, metadata


class Detector(object):
    def __init__(
        self,
        device: str = None,
    ):
        self.logger = logging.getLogger("Detector")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.mlflow_model_name = os.environ.get("MODEL_NAME", "labeler")
        self.mlflow_model_alias = os.environ.get("MODEL_ALIAS", "demo")

        self.tracking_url = os.environ.get(
            "MLFLOW_TRACKING_URI", "http://mlflow_service:5000"
        )

        # self.nms_iou = float(os.environ.get("NMS_IOU", 0.5))
        # self.label_map = json.loads(os.environ.get("LABEL_MAP", "{}"))
        # self.tilesize = None
        # self.batch_size = None
        # self.overlap_ratio = float(os.environ.get("OVERLAP_RATIO", 0.2))

        self.model = None
        self.modelURI = None
        self.model_metadata = None

        self._set_model()

    def _set_model(self):
        mlflow.set_tracking_uri(self.tracking_url)

        self.model, self.model_metadata = load_registered_model(
            alias=self.mlflow_model_alias,
            name=self.mlflow_model_name,
            load_unwrapped=True,
        )

        self.logger.info(
            f"Loading model from: {self.mlflow_model_name}",
        )
        self.logger.info(
            f"Metadata: {self.model_metadata}",
        )

        self.batch_size = self.model_metadata.get("batch") or int(
            os.environ.get("BATCH_SIZE", 1)
        )
        self.tilesize = self.model_metadata.get("tilesize") or int(
            os.environ.get("TILESIZE", 800)
        )

        # warmup
        self.logger.info(
            f"Running warmup with batch_size={self.batch_size} and tilesize={self.tilesize}"
        )
        self.model(
            torch.zeros((self.batch_size, 3, self.tilesize, self.tilesize)),
            verbose=False,
        )

    def predict(
        self, images, iou_nms: float = 0.5, conf: float = 0.2, verbose: bool = False
    ) -> List[dict]:
        try:
            if isinstance(images, torch.Tensor):
                if images.shape[0] != self.model_metadata["batch"]:
                    msg = f"Expected {self.model_metadata['batch']}, but received {images.shape[0]}"
                    raise ValueError(msg)

            results = self.model(
                images,
                verbose=verbose,
                conf=conf,
                imgsz=self.model_metadata["tilesize"],
                iou=iou_nms,
                device=self.device,
            )

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        out = [self._trim_result(o) for o in results]

        return out

    def _trim_result(self, results: List[UltralyticsResults]) -> List[dict]:
        trimmed = []
        for res in results:
            if res.obb is not None:
                boxes = res.obb
            else:
                boxes = res.boxes

            o = dict(
                bbox=boxes.xyxy.cpu().tolist(),
                label=boxes.cls.cpu().flatten().tolist(),
                score=boxes.conf.cpu().flatten().tolist(),
            )
            trimmed.append(o)

        return trimmed
