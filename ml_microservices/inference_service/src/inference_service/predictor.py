import litserve as ls
import os
import torch
import logging

from .utils import Detector

logger = logging.getLogger(__file__)


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
            mlflow_model_name=os.environ.get("MODEL_NAME", "labeler"),
            mlflow_model_alias=os.environ.get("MODEL_ALIAS", "demo"),
            use_sliding_window=True,
            confidence_threshold=0.15,
            overlap_ratio=0.2,
            tilesize=960,
            imgsz=960,
            device=device,
            tracking_url=os.environ.get(
                "MLFLOW_TRACKING_URI", "http://mlflow_service:5000"
            ),
        )

        logger.info("creating model...")

    def decode_request(self, request: dict) -> dict:
        """
        Convert the JSON payload into model inputs.
        For example, extract and preprocess an image or numeric data.
        """
        import base64
        from io import BytesIO
        from PIL import Image

        try:
            img_data = request["image"]

            if not isinstance(img_data, str):
                raise ValueError("Invalid base64 format")

            image_bytes = base64.b64decode(img_data)
            img = Image.open(BytesIO(image_bytes))

        except Exception as e:
            raise ValueError(f"Image decoding failed: {str(e)}")

        return {"image": img}

    def predict(self, x: dict):
        """
        Run the model forward pass.
        Input `x` is the output of decode_request.
        """

        logger.info("computing predictions...")

        results = self.model.predict(**x)
        out = dict(detections=results)

        return out

    def encode_response(self, output: dict):
        """
        Wrap the model output in a JSON-serializable dict.
        """
        logger.debug("sending response...")
        return output
