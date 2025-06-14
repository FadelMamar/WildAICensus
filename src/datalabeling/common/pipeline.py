import logging, os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from label_studio_sdk.client import LabelStudio
import pandas as pd
from ultralytics import SAM
from ultralytics.data.dataset import YOLOConcatDataset, YOLODataset

from ..ml.train import TrainingManager
from ..ml.models import Detector
from .annotation_utils import check_label_format
from .annotation_utils import convert_obb_to_dota as o2d
from .annotation_utils import convert_obb_to_yolo as o2y
from .annotation_utils import convert_yolo_to_coco as y2coco
from .annotation_utils import convert_yolo_to_obb as y2o
from .annotation_utils import create_yolo_seg_directory as yolo2seg
from .config import (
    DataConfig,
    LabelConfig,
    TrainingConfig,
    EvaluationConfig,
    TilingConfig,
)
from .dataset_loader import (
    DataPreparation,
    ClassificationDatasetBuilder,
    LabelingDataset,
)
from .io import load_yaml

logger = logging.getLogger(__name__)


class PipelineStep(ABC):
    """
    Base class for pipeline steps. Each step performs a transformation
    on a shared context dict.
    """

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> None:
        pass


class CheckLabelFormatStep(PipelineStep):
    def run(self, context: Dict[str, Any]) -> None:
        df: pd.DataFrame = context.get("labels_df")
        result = context.get("check_label_format")(df)
        context["label_format"] = result


class YoloToObbStep(PipelineStep):
    def __init__(self, yolo_labels_dir: str, obb_labels_dir: str, skip: bool = True):
        self.yolo_dir = yolo_labels_dir
        self.obb_dir = obb_labels_dir
        self.skip = skip

    def run(self, context: Dict[str, Any]) -> None:
        y2o(
            yolo_labels_dir=self.yolo_dir,
            output_dir=self.obb_dir,
            skip=self.skip,
        )
        context["obb_dir"] = self.obb_dir


class ObbToYoloStep(PipelineStep):
    def __init__(self, obb_labels_dir: str, yolo_labels_dir: str, skip: bool = True):
        self.obb_dir = obb_labels_dir
        self.yolo_dir = yolo_labels_dir
        self.skip = skip

    def run(self, context: Dict[str, Any]) -> None:
        o2y(
            obb_labels_dir=self.obb_dir,
            output_dir=self.yolo_dir,
            skip=self.skip,
        )
        context["yolo_dir"] = self.yolo_dir


class ObbToDotaStep(PipelineStep):
    def __init__(
        self,
        obb_img_dir: str,
        dota_dir: str,
        label_map: Dict[int, str],
        skip: bool = True,
        clear_old: bool = True,
    ):
        self.obb_dir = obb_img_dir
        self.dota_dir = dota_dir
        self.label_map = label_map
        self.skip = skip
        self.clear_old = clear_old

    def run(self, context: Dict[str, Any]) -> None:
        o2d(
            obb_img_dir=self.obb_dir,
            output_dir=self.dota_dir,
            label_map=self.label_map,
            skip=self.skip,
            clear_old_labels=self.clear_old,
        )
        context["dota_dir"] = self.dota_dir


class YoloToSegStep(PipelineStep):
    def __init__(
        self,
        data_config_yaml: str,
        model_sam: SAM,
        device: str = "cuda",
        copy_images_dir: bool = True,
        clear: bool = True,
    ):
        self.data_config_yaml = data_config_yaml
        self.model_sam = model_sam
        self.device = device
        self.copy_images_dir = copy_images_dir
        self.clear = clear

    def run(self, context: Dict[str, Any] = None) -> None:
        yolo2seg(
            data_config_yaml=self.data_config_yaml,
            model_sam=self.model_sam,
            device=self.device,
            copy_images_dir=self.copy_images_dir,
            clear=self.clear,
        )


class YoloToCocoStep(PipelineStep):
    def __init__(
        self,
        coco_dir: str,
        # imgsz:int,
        data_config: Dict[str, Any],
        split: str = "val",
        clear: bool = False,
    ):
        self.dataset = None
        self.coco_dir = coco_dir
        self.data_config = data_config
        self.split = split
        self.clear = clear

        # load dataset
        datasets = []
        for path in self.data_config[split]:
            img_dir = os.path.join(self.data_config["path"], path)
            d = YOLODataset(
                img_path=img_dir,
                task="detect",
                data={"names": self.data_config["names"]},
                augment=False,
                # imgsz=imgsz,
                classes=None,
            )
            datasets.append(d)

        self.dataset = YOLOConcatDataset(datasets)

    def run(self, context: Dict[str, Any]) -> None:
        y2coco(
            dataset=self.dataset,
            output_dir=self.coco_dir,
            data_config=self.data_config,
            split=self.split,
            clear_data=self.clear,
        )
        context["coco_dir"] = self.coco_dir


class LabelstudioToYolo(PipelineStep):
    def __init__(
        self,
        dataset_config: DataConfig,
        label_config: LabelConfig,
        image_dir,
        max_workers: int = 1,
        ls_client=None,
        ls_xml_config: str = None,
        project_id: int = None,
        tiling_config: TilingConfig = None,
    ):
        self.ls_to_yolo = DataPreparation(dataset_config, label_config)
        self.image_dir = image_dir
        self.ls_client = ls_client
        self.max_workers = max_workers
        self.ls_xml_config = ls_xml_config
        self.project_id = project_id
        self.labelstudio_client = None

        self.tiling_config = tiling_config
        self.dataset_config = dataset_config
        self.label_config = label_config

        if self.project_id is not None:
            assert self.ls_client is not None, "Provide labelstudio client"

    def run(self, context: Dict[str, Any] = None) -> None:
        if self.project_id is not None:
            dataset = LabelingDataset.from_ls(
                self.ls_client,
                project_id=self.project_id,
                config=self.tiling_config,
                top_n=0,
                max_workers=self.max_workers,
                load_existing_metadata=False,
            )
            dataset.slice_and_save_as_yolo(
                self.dataset_config, self.label_config, max_workers=self.max_workers
            )

        else:
            self.ls_to_yolo.run(
                ls_xml_config=self.ls_xml_config,
                image_dir=self.image_dir,
                ls_client=self.ls_client,
            )


class ClassificationDataExport(PipelineStep):
    def __init__(
        self,
        detector: Detector,
        eval_config: EvaluationConfig,
        source_dirs: list[str],
        output_dir: str,
        bbox_resize_factor: float = 2.0,
        save_true_negatives=True,
        feature_extractor=None,
        tn_kwargs=dict(w=50, h=50, number=3),
    ):
        self.handler = ClassificationDatasetBuilder(
            detector,
            eval_config,
            source_dirs=source_dirs,
            output_dir=output_dir,
            feature_extractor=feature_extractor,
        )
        self.bbox_resize_factor = bbox_resize_factor
        self.save_true_negatives = save_true_negatives
        self.tn_kwargs = tn_kwargs

    def run(self, context: Dict[str, Any] = None) -> None:
        self.handler.run(
            strategy="gt",
            save_true_negatives=self.save_true_negatives,
            tn_kwargs=self.tn_kwargs,
            bbox_resize_factor=self.bbox_resize_factor,
        )


class ModelTrainingStep(PipelineStep):
    def __init__(
        self,
        training_cfg: TrainingConfig,
        model_type: str = "ultralytics",
        herdnet_loss=None,
        herdnet_training_backend: str = "original",
        classifier_training_backend: str = "ultralytics",
    ):
        self.training_cfg = training_cfg
        self.model_type = model_type
        self.herdnet_loss = herdnet_loss
        self.herdnet_training_backend = herdnet_training_backend
        self.classifier_training_backend = classifier_training_backend

    def run(self, context: Dict[str, Any] = None) -> None:
        trainer = TrainingManager(
            args=self.training_cfg,
            herdnet_loss=self.herdnet_loss,
            herdnet_training_backend=self.herdnet_training_backend,
            model_type=self.model_type,
            classifier_training_backend=self.classifier_training_backend,
        )
        trainer.run()


class Pipeline:
    """
    Orchestrates a sequence of PipelineSteps operating on a shared context.
    """

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def run(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context: Dict[str, Any] = initial_context or {}
        # Inject raw functions
        context.setdefault("check_label_format", check_label_format)
        for step in self.steps:
            step.run(context)
        return context
