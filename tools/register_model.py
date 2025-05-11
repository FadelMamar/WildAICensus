"""Creates/gets an MLflow experiment and registers a detection model to the Model Registry."""

# import argparse
from dataclasses import dataclass
from sys import version_info
from sahi.models.ultralytics import UltralyticsDetectionModel
from ultralytics import YOLO
from sahi.predict import get_prediction, get_sliced_prediction
from pathlib import Path
import torch
import mlflow
from datargs import parse


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

        path = Path(context.artifacts["path"]).resolve().as_posix()
        path = path.replace("\\", "/")

        self.detection_model = UltralyticsDetectionModel(
            model=YOLO(path, task="detect"),
            confidence_threshold=self.confidence_threshold,
            image_size=self.imgsz,
            device=device,
        )

    def predict(self, context, img):
        if self.use_sliding_window:
            tilesize = self.tilesize
            result = get_sliced_prediction(
                img,
                self.detection_model,
                slice_height=tilesize,
                slice_width=tilesize,
                overlap_height_ratio=self.overlapratio,
                overlap_width_ratio=self.overlapratio,
                postprocess_type="NMS",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=self.nms_iou,
                verbose=False,
            )
        else:
            result = get_prediction(
                image=img,
                detection_model=self.detection_model,
                shift_amount=[0, 0],
                full_shape=None,
                postprocess=None,
                verbose=False,
            )

        result = result.to_coco_annotations()

        return result


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


def main():
    args = parse(Args)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    artifacts = {"path": str(Path(args.model).resolve())}

    model = DetectorWrapper(
        tilesize=args.tilesize,
        confidence_threshold=args.confidence_threshold,
        overlap_ratio=args.overlap_ratio,
        use_sliding_window=args.use_sliding_window,
        nms_iou=args.nms_iou,
        imgsz=args.imgsz,
    )

    exp_id = get_experiment_id(args.exp_name)

    # cloudpickle.register_pickle_by_value(model_wrapper)

    with mlflow.start_run(experiment_id=exp_id):
        mlflow.pyfunc.log_model(
            "finetuned",
            python_model=model,
            conda_env=conda_env,
            artifacts=artifacts,
            registered_model_name=args.model_name,
        )


if __name__ == "__main__":
    main()
