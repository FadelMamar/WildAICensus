from PIL import Image
import os
import logging
import torch
from pathlib import Path
from sahi.predict import get_sliced_prediction
import mlflow

logger = logging.getLogger(__name__)


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

    if load_unwrapped:
        try:
            model = model.unwrap_python_model().detection_model
        except:
            model = model.unwrap_python_model().classifier

    return model, modelversion


class Detector(object):
    def __init__(
        self,
        mlflow_model_name: str,
        mlflow_model_alias: str,
        use_sliding_window: bool = True,
        confidence_threshold: float = 0.15,
        overlap_ratio: float = 0.2,
        tilesize: int | None = 960,
        imgsz: int = 960,
        device: str = None,
        tracking_url: str = "http://mlflow_service:5000",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sahi_prostprocess = "NMS"

        self.confidence_threshold = confidence_threshold
        self.overlap_ratio = overlap_ratio
        self.tilesize = tilesize
        self.imgsz = imgsz
        self.use_sliding_window = use_sliding_window

        self.tracking_url = tracking_url
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_model_alias = mlflow_model_alias

        # LS label config
        self.from_name = "label"
        self.to_name = "image"
        self.label_type = "rectanglelabels"

        self.model = None
        self.modelURI = None
        self.modelversion = None

    def _set_model(self):
        mlflow.set_tracking_uri(self.tracking_url)

        self.model, _ = load_registered_model(
            alias=self.mlflow_model_alias, name=self.mlflow_model_name
        )

    def predict(
        self,
        image: Image.Image,
    ):
        """Run sliced predictions

        Args:
            image (Image): input image

        Returns:
            dict: predictions in coco format
        """

        if self.model is None:
            self._set_model()

        assert isinstance(image, Image.Image), (
            f"image should be instance of Image.Image, got {type(image)}"
        )

        with torch.no_grad():
            result = get_sliced_prediction(
                image,
                self.model,
                slice_height=self.tilesize,
                slice_width=self.tilesize,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio,
                postprocess_type="NMS",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=0.5,
                verbose=False,
            )

        out = result.to_coco_annotations()

        return out
