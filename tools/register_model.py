"""Creates/gets an MLflow experiment and registers a detection model to the Model Registry."""

# import argparse
import fire
from dataclasses import dataclass
from sys import version_info
from sahi.models.ultralytics import UltralyticsDetectionModel
from ultralytics import YOLO
from sahi.predict import get_prediction, get_sliced_prediction
from pathlib import Path
import torch
import mlflow
from datargs import parse
import platform
from datalabeling.ml.models import ImageClassifier


def get_experiment_id(name: str):
    """Gets mlflow experiments id

    Args:
        name (str): mlflow experiment name

    Returns:
        str: experiment id
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id


class DetectorWrapper(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
    ):
        super(DetectorWrapper, self).__init__()
        self.model = None

    def load_context(self, context):
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"

        path = Path(context.artifacts["path"]).resolve()

        if platform.system().lower() != "windows":
            path = path.as_posix().replace("\\", "/")

        self.model = YOLO(path, task="detect")


class RoiClassifierWrapper(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
    ):
        super(RoiClassifierWrapper, self).__init__()

        self.model = None

    def load_context(self, context):
        path = Path(context.artifacts["path"]).resolve()

        if platform.system().lower() != "windows":
            path = path.as_posix().replace("\\", "/")

        self.model = torch.jit.load(path, map_location="cpu")
        self.model.eval()


@dataclass
class Args:
    exp_name: str  # MLflow experiment name
    model: str  # Path to saved PyTorch model
    model_name: str  # Registered model name

    mlflow_tracking_uri: str = "http://localhost:5000"

    confidence_threshold: float = 0.1
    overlap_ratio: float = 0.1
    tilesize: int = 2000
    imgsz: int = 1280
    nms_iou: float = 0.5  # used when use_sliding_window=False

    use_sliding_window: bool = False


PYTHON_VERSION = "{major}.{minor}.1".format(
    major=version_info.major, minor=version_info.minor
)

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        "python>=3.11",
        "pip",
        {
            "pip": [
                "mlflow>=2.13.2",
                "pillow",
                "ultralytics",
                "sahi",
                "torch>=2.0.0",
            ],
        },
    ],
    "name": "wildai_env",
}


class RegisterDetector(object):
    def __init__(
        self,
        weights: str,
        name: str = "labeler",
        imgsz: int = 800,
        mlflow_tracking_uri: str = "http://localhost:5000",
    ):
        model_path = Path(weights).resolve()
        YOLO(model_path, task="detect").export(
            format="torchscript", imgsz=imgsz, optimize=True, nms=True, device="cpu"
        )
        torchscript_path = model_path.with_suffix(".torchscript")

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        artifacts = {"path": str(torchscript_path)}

        exp_id = get_experiment_id(name)

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.pyfunc.log_model(
                "finetuned",
                python_model=DetectorWrapper(),
                conda_env=conda_env,
                artifacts=artifacts,
                registered_model_name=name,
            )


class RegisterRoiClassifier(object):
    def __init__(
        self,
        weights: str = r"D:\datalabeling\base_models_weights\roi_classifier.ckpt",
        num_classes: int = 2,
        cls_is_features: bool = True,
        cls_imgsz: int = 128,
        cls_embed_dim: int = 384,
        name: str = "classifier",
        mlflow_tracking_uri: str = "http://localhost:5000",
        save_path: str = "roi_classifier_torchscript.pt",
    ):
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        model_path = Path(weights).resolve()
        model = ImageClassifier.load_from_checkpoint(
            model_path, num_classes=num_classes, cls_is_features=cls_is_features
        ).model

        # initialize weights
        if cls_is_features:
            model(torch.zeros(1, cls_embed_dim))
        else:
            model(torch.zeros(1, 3, cls_imgsz, cls_imgsz))

        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(save_path)  # Save

        artifacts = {"path": save_path}

        exp_id = get_experiment_id(name)

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.pyfunc.log_model(
                "finetuned",
                python_model=RoiClassifierWrapper(),
                conda_env=conda_env,
                artifacts=artifacts,
                registered_model_name=name,
            )


class Register(object):
    def register_detector(
        self,
        weights_path: str,
        name: str = "labeler",
        imgsz: int = 800,
        mlflow_tracking_uri: str = "http://localhost:5000",
    ):
        RegisterDetector(
            weights=weights_path,
            name=name,
            imgsz=imgsz,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

    def register_classifier(
        self,
        weights_path,
        num_classes: int = 2,
        cls_is_features: bool = True,
        cls_imgsz: int = 128,
        cls_embed_dim: int = 384,
        name: str = "classifier",
        mlflow_tracking_uri: str = "http://localhost:5000",
        save_path: str = "roi_classifier_torchscript.pt",
    ):
        RegisterRoiClassifier(
            weights=weights_path,
            num_classes=num_classes,
            cls_is_features=cls_is_features,
            cls_imgsz=cls_imgsz,
            cls_embed_dim=cls_embed_dim,
            name=name,
            mlflow_tracking_uri=mlflow_tracking_uri,
            save_path=save_path,
        )


if __name__ == "__main__":
    fire.Fire(Register)
