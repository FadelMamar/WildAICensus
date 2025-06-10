import litserve as ls
import os
import torch
import logging
import traceback

# from .utils import Detector
# import json
from fastapi import HTTPException

# from PIL import Image
from dataclasses import dataclass
from typing import List  # , Optional, Sequence
import os
import logging

# import numpy as np
import torch
from pathlib import Path
import mlflow
from ultralytics.engine.results import Results as UltralyticsResults
import base64
from io import BytesIO
from PIL import Image

# import time
# from torchvision.transforms import PILToTensor
# from torchvision.ops import nms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset


logger = logging.getLogger("Predictor")


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

        self.model = None
        self.modelURI = None
        self.model_metadata = None

        self._set_model()

    def load_registered_model(
        self,
        alias,
        name,
        load_unwrapped: bool = True,
    ):
        client = mlflow.MlflowClient()

        version = client.get_model_version_by_alias(name=name, alias=alias).version
        modelversion = f"{name}:{version}"
        modelURI = f"models:/{name}/{version}"

        dwnd_location = (
            Path(os.environ.get("WEIGHTS_PATH", "./model_weights")) / f"{name}"
        )
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

    def _set_model(self):
        mlflow.set_tracking_uri(self.tracking_url)

        self.model, self.model_metadata = self.load_registered_model(
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

    def _pad_if_needed(self, batch: torch.Tensor, out_shape: tuple) -> torch.Tensor:
        # if batch size is less than expected, pad with zeros

        assert len(out_shape) == len(batch.shape)
        assert len(out_shape) == 4

        condition = any([a < b for a, b in zip(batch.shape, out_shape)])

        if condition:
            b, c, h, w = batch.shape
            padded = torch.zeros(out_shape)
            padded[:b, :c, :h, :w] = batch.clone()
            batch = padded

        return batch

    def _predict_tensor(
        self, images: torch.Tensor, conf: float, iou_nms: float, verbose: bool = False
    ) -> list:
        batchsize = self.model_metadata["batch"]
        tilesize = self.model_metadata["tilesize"]
        images = TensorDataset(images)
        loader = DataLoader(images, batch_size=batchsize, shuffle=False)

        # if images.shape[0] != self.model_metadata['batch']:
        #     msg = f"Expected {self.model_metadata['batch']}, but received {images.shape[0]}"
        #     raise ValueError(msg)

        results = []
        for (batch,) in loader:
            num_images, n_channels = batch.shape[:2]

            padded = self._pad_if_needed(
                batch, out_shape=(batchsize, n_channels, tilesize, tilesize)
            )
            res = self.model(
                padded,
                verbose=verbose,
                conf=conf,
                imgsz=tilesize,
                iou=iou_nms,
                device=self.device,
            )
            results.extend(res[:num_images])

        return results

    def predict(
        self, images, iou_nms: float = 0.5, conf: float = 0.2, verbose: bool = False
    ) -> List[dict]:
        try:
            if isinstance(images, torch.Tensor):
                results = self._predict_tensor(
                    images, conf=conf, iou_nms=iou_nms, verbose=verbose
                )
            else:
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

        out = []
        for res in results:
            out.extend(self._trim_result(res))

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


class MyModelAPI(ls.LitAPI):
    def setup(
        self,
        device,
    ):
        """
        One-time initialization: load your model here.
        `device` is e.g. 'cuda:0' or 'cpu'.
        """
        logger.info(f"Device: {device}")
        self.model = Detector(
            device=device,
        )

    async def decode_request(self, request: dict) -> dict:
        """
        Convert the JSON payload into model inputs.
        For example, extract and preprocess an image or numeric data.
        """

        output = dict()

        try:
            img_data = request.get("images", None)
            img_tensor = request.get("tensor", None)

            assert (img_data is not None) + (img_tensor is not None), (
                "Provide Exactly One of them."
            )

            if img_data is not None:
                decoded_images = []
                for data in img_data:
                    img = base64.b64decode(data)
                    img = Image.open(BytesIO(img))
                    decoded_images.append(img)
                    output["images"] = decoded_images

            elif img_tensor is not None:
                tensor_bytes = base64.b64decode(img_tensor)
                tensor_bytes = bytearray(tensor_bytes)
                shape = request.get("shape")
                img_tensor = torch.frombuffer(
                    tensor_bytes, dtype=torch.float32
                ).reshape(shape)
                output["images"] = img_tensor

            else:
                print("Provide field 'images' or 'tensor'.")
                raise ValueError("No image data found in request")

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))

        return output

    async def predict(self, x: dict) -> dict:
        """
        Run the model forward pass.
        Input `x` is the output of decode_request.
        """

        try:
            results = self.model.predict(
                x.get("images"),
                iou_nms=x.get("iou_nms", 0.5),
                conf=x.get("conf", 0.2),
                verbose=x.get("verbose", False),
            )
            out = dict(detections=results)
            return out
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")

    async def encode_response(self, output: dict):
        """
        Wrap the model output in a JSON-serializable dict.
        """
        logger.debug("sending response...")
        return output
