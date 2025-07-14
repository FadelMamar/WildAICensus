"""Creates/gets an MLflow experiment and registers a detection model to the Model Registry."""

# import argparse
import fire
from sys import version_info

from ultralytics import YOLO

from pathlib import Path
import torch
import mlflow
import platform
import os

os.environ["TORCH_XNNPACK_DISABLE"] = "1"


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
        self.artifacts = None

    def load_context(self, context):
        path = Path(context.artifacts["path"]).resolve()

        if platform.system().lower() != "windows":
            path = path.as_posix().replace("\\", "/")

        self.model = YOLO(path, task="detect")

        self.artifacts = context.artifacts


class RoiClassifierWrapper(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
    ):
        super(RoiClassifierWrapper, self).__init__()

        self.model = None
        self.artifacts = None

    def load_context(self, context):
        path = Path(context.artifacts["path"]).resolve()

        if platform.system().lower() != "windows":
            path = path.as_posix().replace("\\", "/")

        self.model = torch.jit.load(path, map_location="cpu")
        self.model.eval()

        self.artifacts = context.artifacts


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


class Register(object):
    def register_detector(
        self,
        weights_path: str,
        name: str = "labeler",
        export_format: str = "torchscript",
        imgsz: int = 800,
        batch: int = 8,
        device: str = "cpu",
        mlflow_tracking_uri: str = "http://localhost:5000",
        dynamic: bool = False,
        task: str = "detect",
    ):
        model_path = Path(weights_path).resolve()
        export_path = model_path
        if export_format != "pt":
            YOLO(model_path, task=task).export(
                format=export_format,
                imgsz=imgsz,
                optimize=device == "cpu",
                nms=True,
                dynamic=dynamic,
                batch=batch,
                device=device,
            )

            if export_format == "openvino":
                export_path = model_path.with_name(f"{model_path.stem}_openvino_model")
            else:
                export_path = model_path.with_suffix(f".{export_format}")

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        artifacts = {
            "path": str(export_path),
        }
        metadata = {"batch": batch, "tilesize": imgsz, "task": task}

        exp_id = get_experiment_id(name)

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.pyfunc.log_model(
                "finetuned",
                python_model=DetectorWrapper(),
                conda_env=conda_env,
                artifacts=artifacts,
                registered_model_name=name,
                metadata=metadata,
            )

    def register_classifier(
        self,
        weights_path,
        num_classes: int = 2,
        cls_is_features: bool = True,
        batch: int = 8,
        cls_imgsz: int = 128,
        cls_embed_dim: int = 384,
        name: str = "classifier",
        mlflow_tracking_uri: str = "http://localhost:5000",
    ):
        from datalabeling.ml.models import ImageClassifier

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        model_path = Path(weights_path).resolve()
        model = ImageClassifier.load_from_checkpoint(
            model_path, num_classes=num_classes, cls_is_features=cls_is_features
        ).model

        model = model.cpu()  # Ensure model is on CPU for export

        # initialize weights
        if cls_is_features:
            model(torch.zeros(batch, cls_embed_dim))
        else:
            model(torch.zeros(batch, 3, cls_imgsz, cls_imgsz))

        model_scripted = torch.jit.script(model)  # Export to TorchScript
        save_path = str(model_path.with_suffix(".torchscript"))
        model_scripted.save(save_path)  # Save

        artifacts = {
            "path": save_path,
        }

        metadata = {
            "num_classes": num_classes,
            "cls_is_features": cls_is_features,
            "cls_imgsz": cls_imgsz,
            "batch": batch,
            "cls_embed_dim": cls_embed_dim,
        }

        exp_id = get_experiment_id(name)

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.pyfunc.log_model(
                "finetuned",
                python_model=RoiClassifierWrapper(),
                conda_env=conda_env,
                artifacts=artifacts,
                registered_model_name=name,
                metadata=metadata,
            )


if __name__ == "__main__":
    fire.Fire(Register)
