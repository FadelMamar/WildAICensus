import logging
import os, json
from pathlib import Path
import lightning as L
import torch
from tqdm import tqdm
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
import numpy as np
import mlflow
from lightning.pytorch.loggers import MLFlowLogger

from ultralytics import RTDETR, YOLO
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import platform
from datalabeling.common.mlflow_utils import load_registered_model
from datalabeling.common.config import TrainingConfig
from datalabeling.ml.models import ImageClassifier
from datalabeling.common.io import ClassifierDataModule

from datalabeling.ml.utils import (
    get_data_cfg_paths_for_cl,
    get_data_cfg_paths_for_HN,
    remove_label_cache,
    CustomTrainer,
    CustomYOLO,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    # model type
    is_single_cls: bool = False
    is_rtdetr: bool = False
    task: str = "detect"  # "detect" "obb" "segment"
    model_type: str = "ultralytics"  # "ultralytics", "herdnet", "classifier"

    # active learning flags
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_model_alias: str = None
    mlflow_model_name: str = None

    # training data
    yolo_yaml: str = None  # os.path.join(CUR_DIR,'../../../data/data_config.yaml')
    yolo_arch_yaml: str = None

    ultralytics_pos_weight: float = 1.0

    # detector
    object_detector_arch: str = "yolo"  # "yolo", "rtdetr", "custom_yolo"
    custom_yolo_kwargs: dict = None  # used only for custom yolo arch

    # training flags
    imgsz: int = 800
    path_weights: str = None
    lr0: float = 1e-4
    lrf: float = 1e-2
    warmup_epochs: int = 3
    batchsize: int = 32
    epochs: int = 50
    seed = 41
    optimizer: str = "AdamW"
    optimizer_momentum: float = 0.99
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 10
    val: str = "True"
    yolo_iou_val: float = 0.6
    dfl: float = 1.5
    cls: float = 0.5
    box: float = 7.5

    # classifier
    cls_num_classes: int = 2
    cls_label_smoothing: float = 0.0
    cls_thrs: float = 0.5
    cls_workdir: str = "runs-classifier"
    cls_data_dir: str = None
    cls_monitor_metric: str = "val_f1score"
    cls_monitor_mode: str = "max"
    cls_auto_augment: str = "augmix"
    cls_is_features: bool = False
    cls_tn_ratio: float = 1.0
    cls_training_backend: str = "pl"  # pl, sk, ultralytics

    # pretraining
    use_pretraining: bool = False
    ptr_data_config_yaml: str = None
    ptr_tilesize: int = 640
    ptr_batchsize: int = 32
    ptr_epochs: int = 10
    ptr_lr0: float = 1e-4
    ptr_lrf: float = 1e-1
    ptr_freeze: int = None

    # continual learning flags
    use_continual_learning: bool = False
    cl_ratios: Sequence[float] = (1.0,)  # ratio = num_empty/num_non_empty
    cl_epochs: Sequence[int] = (20,)
    cl_freeze: Sequence[int] = (0,)
    cl_lr0s: Sequence[float] = (5e-5,)
    cl_save_dir: str = None  # should be given!
    cl_data_config_yaml: str = None
    cl_batch_size: int = 16

    # hard negative data sampling learning mode
    use_hn_learning: bool = False
    hn_save_dir: str = None
    hn_data_config_yaml: str = None
    hn_imgsz: int = 1280  # used to resize the input image
    hn_tilesize: int = 1280  # used for sliding window based detections
    hn_num_epochs: int = 10
    hn_freeze: int = 20
    hn_lr0: float = 5e-5
    hn_lrf: float = 1e-1
    hn_batch_size: int = 16
    hn_is_yolo_obb: bool = False
    hn_use_sliding_window = True  # can't change thru cli
    hn_overlap_ratio: float = 0.2
    hn_map_thrs: float = (
        0.35  # mAP threshold. lower than it is considered sample of interest
    )
    hn_score_thrs: float = 0.7
    hn_confidence_threshold: float = 0.25
    hn_ratio: int = 20  # ratio = num_empty/num_non_empty. Higher allows to look at all saved empty images
    hn_uncertainty_thrs: float = 5  # helps to select those with high uncertainty
    hn_uncertainty_method: str = "entropy"
    hn_load_results: bool = False

    # regularization
    dropout: float = 0.0  # used only for classification tasks
    weight_decay: float = 5e-4

    # transfer learning
    freeze: int = None

    # lr scheduling
    cos_annealing: bool = True

    # run and project name MLOps
    run_name: str = "debug"
    project_name: str = "wildAI"
    tag: Sequence[str] = ("",)

    # data augmentation https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters
    rotation_degree: float = 45.0
    mixup: float = 0.0
    shear: float = 10.0
    copy_paste: float = 0.0
    erasing: float = 0.0
    scale: float = 0.0
    fliplr: float = 0.5
    flipud: float = 0.5
    hsv_h: float = 0.0
    hsv_s: float = 0.3
    hsv_v: float = 0.3
    translate: float = 0.2
    mosaic: float = 0.0
    multi_scale: bool = False


def load_ultralytics_model_class(
    object_detector_arch: str, path: str, task: str = "detect", **kwargs
):
    if object_detector_arch == "rtdetr":
        return RTDETR(path)
    if object_detector_arch == "yolo":
        return YOLO(model=path, task=task)
    if object_detector_arch == "custom_yolo":
        return CustomYOLO(model=path, **kwargs)
    else:
        raise NotImplementedError(
            f"object_detector_arch `{object_detector_arch}` is not supported."
        )


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


class TrainingManager:
    def __init__(
        self,
        args: TrainingConfig,
    ):
        self.args = args

        assert self.args.task in ["detect", "obb", "segment"]

        assert args.model_type in ["detector", "classifier"], (
            f"this model_type ``{args.model_type}`` is not supported."
        )
        assert args.cls_training_backend in ["pl", "ultralytics"]

        # assert isinstance(self.args.mlflow_model_alias,str), f"Expected type 'str', received {type(self.args.mlflow_model_alias)}"

        self.model = self._load_model()

    def _load_model(self):
        alias = self.args.mlflow_model_alias
        if alias:
            model, metadata = load_registered_model(
                alias=alias,
                name=self.args.mlflow_model_name,
                mlflow_tracking_url=self.args.mlflow_tracking_uri,
                load_unwrapped=True,
            )
            # self.args.path_weights = model.detection_model.model.ckpt_path
            modeluri = metadata.get("modeluri")
            logger.info(f"Loading {modeluri}")
            return model

        elif not (self.args.path_weights or self.args.yolo_arch_yaml):
            raise ValueError("Provide 'path_weights' or 'yolo_arch_yaml'.")

        if self.args.model_type == "detector":
            return self._load_ultralytics_model()
        elif self.args.model_type == "classifier":
            return self._load_classifier_model()
        else:
            raise NotImplementedError

    def _load_ultralytics_model(self):
        model = None

        path = self.args.path_weights
        if self.args.yolo_arch_yaml:
            path = self.args.yolo_arch_yaml

        model = load_ultralytics_model_class(
            object_detector_arch=self.args.object_detector_arch,
            path=path,
            **self.args.custom_yolo_kwargs,
        )

        if self.args.path_weights and self.args.yolo_arch_yaml:
            model = model.load(self.args.path_weights)

        return model

    def _load_classifier_model(
        self,
    ):
        if self.args.cls_training_backend == "ultralytics":
            return self._load_ultralytics_model()

        # using pl
        model = ImageClassifier(
            cls_is_features=self.args.cls_is_features,
            epochs=self.args.epochs,
            num_classes=self.args.cls_num_classes,
            threshold=self.args.cls_thrs,
            label_smoothing=self.args.cls_label_smoothing,
            lr=self.args.lr0,
            lrf=self.args.lrf,
            weight_decay=self.args.weight_decay,
        )

        return model

    def run(self):
        if self.args.model_type == "detector":
            self._run_ultralytics()

        elif self.args.model_type == "classifier":
            self._run_classifier()

        else:
            raise NotImplementedError

    def log(self, registered_model_name: str):
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

        artifacts = {
            "path": "al.pt",
        }
        metadata = {"batch": None, "tilesize": self.args.imgsz, "task": self.args.task}

        exp_id = get_experiment_id(registered_model_name)

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.pyfunc.log_model(
                "finetuned",
                python_model=DetectorWrapper(),
                conda_env=conda_env,
                artifacts=artifacts,
                registered_model_name=registered_model_name,
                metadata=metadata,
            )

        logger.info("Logging is successful.")

    def _run_classifier(
        self,
    ):
        if self.args.cls_training_backend == "ultralytics":
            self._train_ultralytics(data_cfg=self.args.cls_data_dir)

        # using pytorch lightning
        datamodule = ClassifierDataModule(
            data_dir=self.args.cls_data_dir,
            batch_size=self.args.batchsize,
            num_workers=os.cpu_count() // 4,
            img_size=self.args.imgsz,
            is_features=self.args.cls_is_features,
            tn_ratio=self.args.cls_tn_ratio,
        )

        datamodule.setup("fit")

        self.model.set_label_class_map(class_to_label_map=datamodule.class_to_idx)

        # loggers and callbacks
        mlf_logger = MLFlowLogger(
            experiment_name=self.args.project_name,
            run_name=self.args.run_name,
            tracking_uri=self.args.mlflow_tracking_uri,
            log_model=False,
            # artifact_location='checkpoints',
            checkpoint_path_prefix="checkpoints",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.args.cls_workdir,
            monitor=self.args.cls_monitor_metric,
            mode=self.args.cls_monitor_mode,
            filename="best",
            save_weights_only=True,
            save_last=True,
            save_top_k=1,
        )
        lr_callback = LearningRateMonitor(logging_interval="epoch")
        callbacks = [
            checkpoint_callback,
            lr_callback,
            EarlyStopping(
                monitor=self.args.cls_monitor_metric,
                patience=self.args.patience,
                min_delta=1e-4,
                mode=self.args.cls_monitor_mode,
            ),
        ]

        # trainer
        trainer = L.Trainer(
            max_epochs=self.args.epochs,
            logger=mlf_logger,
            precision="bf16-mixed",
            callbacks=callbacks,
            detect_anomaly=False,
            accelerator="auto",
        )

        # fit
        trainer.fit(self.model, datamodule=datamodule)

    def _run_ultralytics(self):
        self.model.info()

        if self.args.use_pretraining:
            self._pretraining()

        if self.args.use_continual_learning:
            self._continual_learning()

        if self.args.use_hn_learning:
            self._hard_negative_learning()

        if not (
            self.args.ptr_data_config_yaml
            or self.args.use_continual_learning
            or self.args.use_hn_learning
        ):
            self._train_ultralytics()

    def _train_ultralytics(
        self, data_cfg=None, imgsz=None, batchsize=None, resume=False
    ):
        args = self.args

        assert args.val in ["True", "False"]

        cfg = dict()
        if self.args.object_detector_arch != "rtdetr":
            os.environ["pos_weight"] = json.dumps(self.args.ultralytics_pos_weight)
            if self.args.object_detector_arch == "yolo":
                cfg = dict(trainer=CustomTrainer)

        self.model.train(
            data=data_cfg or args.yolo_yaml,
            epochs=args.epochs,
            imgsz=imgsz or args.imgsz,
            device=args.device,
            freeze=args.freeze,
            name=args.run_name,
            workers=0,
            single_cls=args.is_single_cls,
            lr0=args.lr0,
            lrf=args.lrf,
            momentum=args.optimizer_momentum,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            dfl=args.dfl,
            cls=args.cls,
            box=args.box,
            dropout=args.dropout,
            batch=batchsize or args.batchsize,
            val=args.val == "True",
            plots=True,
            cos_lr=args.cos_annealing,
            deterministic=False,
            cache=False,
            optimizer=args.optimizer,
            project=args.project_name,
            patience=args.patience,
            multi_scale=args.multi_scale,
            degrees=args.rotation_degree,
            mixup=args.mixup,
            scale=args.scale,
            iou=args.yolo_iou_val,
            mosaic=args.mosaic,
            augment=False,
            erasing=args.erasing,
            copy_paste=args.copy_paste,
            shear=args.shear,
            fliplr=args.fliplr,
            flipud=args.flipud,
            perspective=0.0,
            hsv_s=args.hsv_s,
            hsv_h=args.hsv_h,
            hsv_v=args.hsv_v,
            translate=args.translate,
            auto_augment=args.cls_auto_augment,
            exist_ok=True,
            seed=args.seed,
            resume=resume,
            **cfg,
        )

    def _pretraining(self):
        args = self.args
        assert os.path.exists(args.ptr_data_config_yaml), (
            "provide --ptr-data-config-yaml"
        )
        logger.info("\n\n------------ Pretraining ----------\n")
        remove_label_cache(args.ptr_data_config_yaml)
        args.run_name += f"-PTR_freeze_{args.freeze}"
        args.epochs = args.ptr_epochs
        args.lr0 = args.ptr_lr0
        args.lrf = args.ptr_lrf
        args.freeze = args.ptr_freeze
        self._train_ultralytics(
            data_cfg=args.ptr_data_config_yaml,
            imgsz=args.ptr_tilesize,
            batchsize=args.ptr_batchsize,
        )

    def _hard_negative_learning(self, img_glob_pattern: str = "*"):
        args = self.args
        assert args.hn_save_dir, "Provide --hn-save-dir"
        logger.info(
            "\n\n------------ Hard Negative Sampling Learning Strategy ----------\n"
        )
        remove_label_cache(args.hn_data_config_yaml)
        args.run_name += f"-HN_freeze_{args.freeze}"
        cfg_path = get_data_cfg_paths_for_cl(
            ratio=args.hn_ratio,
            data_config_yaml=args.hn_data_config_yaml,
            cl_save_dir=args.hn_save_dir,
            seed=args.seed,
            split="train",
            pattern_glob=img_glob_pattern,
        )
        hn_cfg_path = get_data_cfg_paths_for_HN(
            args=args,
            data_config_yaml=cfg_path,
            model=self.model,
            split="train",
        )
        args.lr0 = args.hn_lr0
        args.lrf = args.hn_lrf
        args.freeze = args.hn_freeze
        args.epochs = args.hn_num_epochs

        self._train_ultralytics(
            data_cfg=hn_cfg_path, imgsz=args.hn_imgsz, batchsize=args.hn_batch_size
        )

    def _continual_learning(self, img_glob_pattern: str = "*"):
        args = self.args
        assert os.path.exists(args.cl_data_config_yaml), "Provide --cl-data-config-yaml"
        logger.info("\n\n------------ Continual Learning ----------\n")
        remove_label_cache(args.cl_data_config_yaml)

        for flag in (args.cl_ratios, args.cl_epochs, args.cl_freeze):
            assert len(flag) == len(args.cl_lr0s), (
                f"All cl_* flags should match length. {len(flag)} != {len(args.cl_lr0s)}"
            )

        original_run_name = args.run_name
        for lr, ratio, num_epochs, freeze in zip(
            args.cl_lr0s, args.cl_ratios, args.cl_epochs, args.cl_freeze, strict=False
        ):
            cl_cfg_path = get_data_cfg_paths_for_cl(
                ratio=ratio,
                data_config_yaml=args.cl_data_config_yaml,
                cl_save_dir=args.cl_save_dir,
                seed=args.seed,
                split="train",
                pattern_glob=img_glob_pattern,
            )
            args.run_name = f"{original_run_name}-CL_emptyRatio_{ratio}_freeze_{freeze}"
            args.freeze = freeze
            args.lr0 = lr
            args.epochs = num_epochs
            self._train_ultralytics(data_cfg=cl_cfg_path, batchsize=args.cl_batch_size)
