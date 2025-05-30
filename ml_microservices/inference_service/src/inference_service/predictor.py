import litserve as ls
import os
import torch
import logging
import traceback
from .utils import Detector
import json

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
            device=device,
        )

        print("Loading model...")

    def decode_request(self, request: dict) -> dict:
        """
        Convert the JSON payload into model inputs.
        For example, extract and preprocess an image or numeric data.
        """
        import base64
        from io import BytesIO
        from PIL import Image

        try:
            img_data = request.get("images")

            if img_data is None:
                raise ValueError("No image data found in request")

            decoded_images = []
            for data in img_data:
                img = base64.b64decode(data)
                img = Image.open(BytesIO(img))
                decoded_images.append(img)

        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Image decoding failed: {str(e)}")

        return {"images": decoded_images}

    def predict(self, x: dict) -> dict:
        """
        Run the model forward pass.
        Input `x` is the output of decode_request.
        """

        logger.info("Running inference...")

        try:
            results = self.model.predict(**x)
            out = dict(detections=results)
            return out
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")

    def encode_response(self, output: dict):
        """
        Wrap the model output in a JSON-serializable dict.
        """
        logger.debug("sending response...")
        return output
