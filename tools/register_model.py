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
        tilesize: int = 640,
        confidence_threshold: float = 0.1,
        overlap_ratio: float = 0.1,
        imgsz: int = 640,
        use_sliding_window: bool = True,
        nms_iou: bool = 0.5,
    ):
        """_summary_

        Args:
            tilesize (int, optional): _description_. Defaults to 640.
            confidence_threshold (float, optional): _description_. Defaults to 0.1.
            overlap_ratio (float, optional): _description_. Defaults to 0.1.
            sahi_postprocess (str, optional): _description_. Defaults to 'NMS'.
        """
        super(DetectorWrapper, self).__init__()
        self.tilesize = tilesize
        self.confidence_threshold = confidence_threshold
        self.overlapratio = overlap_ratio
        self.imgsz = imgsz
        self.use_sliding_window = use_sliding_window
        self.nms_iou = nms_iou
        self.detection_model = None

    def load_context(self, context):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        path = Path(context.artifacts["path"]).resolve()

        if platform.system().lower() != "windows":
            path = path.as_posix().replace("\\", "/")

        self.detection_model = UltralyticsDetectionModel(
            model=YOLO(path, task="detect"),
            confidence_threshold=self.confidence_threshold,
            image_size=self.imgsz,
            device=device,
        )

    # def predict(self, context, img):
    #     if self.use_sliding_window:
    #         tilesize = self.tilesize
    #         result = get_sliced_prediction(
    #             img,
    #             self.detection_model,
    #             slice_height=tilesize,
    #             slice_width=tilesize,
    #             overlap_height_ratio=self.overlapratio,
    #             overlap_width_ratio=self.overlapratio,
    #             postprocess_type="NMS",
    #             postprocess_match_metric="IOU",
    #             postprocess_match_threshold=self.nms_iou,
    #             verbose=False,
    #         )
    #     else:
    #         result = get_prediction(
    #             image=img,
    #             detection_model=self.detection_model,
    #             shift_amount=[0, 0],
    #             full_shape=None,
    #             postprocess=None,
    #             verbose=False,
    #         )

    #     result = result.to_coco_annotations()

    #     return result


class RoiClassifierWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, num_classes=2, cls_is_features=True):
        super(RoiClassifierWrapper, self).__init__()

        self.num_classes = num_classes
        self.cls_is_features = cls_is_features
        self.classifier = None

    def load_context(self, context):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        path = Path(context.artifacts["path"]).resolve()

        if platform.system().lower() != "windows":
            path = path.as_posix().replace("\\", "/")

        self.classifier = torch.jit.load(path, map_location=device)
        self.classifier.eval()


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
        weights,
        model_name="labeler",
        exp_name="labeler",
        tilesize: int = 800,
        confidence_threshold: float = 0.1,
        overlap_ratio: float = 0.2,
        imgsz: int = 800,
        use_sliding_window: bool = True,
        nms_iou: bool = 0.5,
        mlflow_tracking_uri="http://localhost:5000",
    ):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.registered_model_name = model_name
        self.exp_name = exp_name

        self.model = DetectorWrapper(
            tilesize=tilesize,
            confidence_threshold=confidence_threshold,
            overlap_ratio=overlap_ratio,
            use_sliding_window=use_sliding_window,
            nms_iou=nms_iou,
            imgsz=imgsz,
        )

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        artifacts = {"path": str(Path(weights).resolve())}

        exp_id = get_experiment_id(self.exp_name)

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.pyfunc.log_model(
                "finetuned",
                python_model=self.model,
                conda_env=conda_env,
                artifacts=artifacts,
                registered_model_name=self.registered_model_name,
            )


class RegisterRoiClassifier(object):
    def __init__(
        self,
        weights=r"D:\datalabeling\base_models_weights\roi_classifier.ckpt",
        num_classes=2,
        cls_is_features=True,
        name: str = "classifier",
        mlflow_tracking_uri: str = "http://localhost:5000",
        save_path: str = "Roi_classifier_torchscript.pt",
    ):
        from datalabeling.ml.models import ImageClassifier

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        model = ImageClassifier.load_from_checkpoint(
            weights, num_classes=num_classes, cls_is_features=cls_is_features
        ).model

        # initialize weights
        if cls_is_features:
            model(torch.zeros(1, 384))
        else:
            model(torch.zeros(1, 3, 128, 128))

        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(save_path)  # Save

        artifacts = {"path": save_path}

        exp_id = get_experiment_id(name)

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.pyfunc.log_model(
                "finetuned",
                python_model=RoiClassifierWrapper(
                    num_classes=num_classes, cls_is_features=cls_is_features
                ),
                conda_env=conda_env,
                artifacts=artifacts,
                registered_model_name=name,
            )


class Register(object):
    def register_detector(self, detector_weights):
        RegisterDetector(detector_weights)

    def register_classifier(
        self,
    ):
        RegisterRoiClassifier()


if __name__ == "__main__":
    fire.Fire(Register)
