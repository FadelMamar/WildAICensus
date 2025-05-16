import logging
import os
from pathlib import Path
import json
from typing import Sequence
import lightning as L
import torch
import yaml
from animaloc.eval import HerdNetEvaluator, PointsMetrics
from animaloc.eval.lmds import HerdNetLMDS
from animaloc.models import HerdNet, LossWrapper
from tqdm import tqdm
from animaloc.train.losses import FocalLoss
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
import numpy as np
import mlflow
from lightning.pytorch.loggers import MLFlowLogger
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC
from torch.utils.data import DataLoader
from ultralytics import RTDETR, YOLO
from torchvision import models

from ..common.config import TrainingConfig
from ..common.io import HerdnetData, ClassifierDataModule
from ..common.mlflow_utils import load_registered_model

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .utils import (
    get_data_cfg_paths_for_cl,
    get_data_cfg_paths_for_HN,
    remove_label_cache,
    CustomTrainer,
)
from .models import (ImageClassifier,
                     HerdnetTrainer,
                     get_image_classifier_module)

logger = logging.getLogger(__name__)


class TrainingManager:
    def __init__(
        self,
        args: TrainingConfig,
        herdnet_loss: list = None,
        herdnet_training_backend: str = "original",
        classifier_training_backend: str = "pl",
        model_type: str = "ultralytics",
    ):
        self.args = args
        self.herdnet_loss = herdnet_loss
        self.model_type = model_type
        self.herdnet_training_backend = herdnet_training_backend
        self.classifier_training_backend = classifier_training_backend

        assert model_type in ["ultralytics", "herdnet", "classifier"], (
            f"this model_type ``{model_type}`` is not supported."
        )

        assert classifier_training_backend in ["pl", "ultralytics", "sk"]

        assert herdnet_training_backend in ["original", "pl"], (
            "the provided backend is not supported."
        )

        self.model = self._load_model()

    def _load_model(self):
        if self.args.mlflow_model_alias is not None:
            name = self.args.run_name
            alias = self.args.mlflow_model_alias            
            model, version = load_registered_model(alias=alias,name=name,mlflow_tracking_url=self.args.mlflow_tracking_uri,load_unwrapped=True)
            self.args.path_weights = model.detection_model.model.ckpt_path
            logger.info(f"Loading model registered with alias: {alias}")

        if self.model_type == "ultralytics":
            return self._load_ultralytics_model()

        elif self.model_type == "herdnet":
            return self._load_herdnet()

        elif self.model_type == "classifier":
            return self._load_classifier_model()

        else:
            raise NotImplementedError

    def _load_ultralytics_model(self):
        model = None

        path = self.args.path_weights
        if self.args.yolo_arch_yaml:
            path = self.args.yolo_arch_yaml

        if self.args.is_rtdetr:
            model = RTDETR(path)
        else:
            model = YOLO(path, task=self.args.task, verbose=False)

        if self.args.path_weights and self.args.yolo_arch_yaml:
            model = model.load(self.args.path_weights)

        return model

    def _load_classifier_model(
        self,
    ):
        if self.classifier_training_backend == "ultralytics":
            return self._load_ultralytics_model()

        elif self.classifier_training_backend == "sk":
            # pipe = make_pipeline(StandardScaler(),SGDClassifier(loss='hinge',n_jobs=4))
            return SGDClassifier(loss="hinge", n_jobs=4)

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

    def _load_herdnet(
        self,
    ):
        if self.herdnet_loss is None:
            ce_weights = (
                torch.Tensor(self.args.herdnet_ce_weight).to(self.args.device)
                if self.args.herdnet_ce_weight is not None
                else None
            )
            self.herdnet_loss = [
                {
                    "loss": FocalLoss(reduction="mean"),
                    "idx": 0,
                    "idy": 0,
                    "lambda": 1.0,
                    "name": "focal_loss",
                },
                {
                    "loss": CrossEntropyLoss(reduction="mean", weight=ce_weights),
                    "idx": 1,
                    "idy": 1,
                    "lambda": 1.0,
                    "name": "ce_loss",
                },
            ]

        mode = "both"
        if self.herdnet_training_backend == "original":
            mode = "module"

        # Load herdnet object
        if self.args.path_weights is None:
            self.model = HerdNet(
                pretrained=False,  # load dla weights if True
                down_ratio=self.args.herdnet_down_ratio,
                num_classes=self.args.herdnet_num_classes,
            )
            self.model = LossWrapper(self.model, losses=self.herdnet_loss, mode=mode)

        else:
            self.model = HerdNet(
                pretrained=False,
                down_ratio=self.args.herdnet_down_ratio,
                num_classes=self.args.herdnet_ptr_model_classes,
            )
            self.model = LossWrapper(self.model, losses=self.herdnet_loss, mode=mode)
            checkpoint = torch.load(
                self.args.path_weights, map_location=self.args.device, weights_only=True
            )

            try:
                success = self.model.load_state_dict(
                    checkpoint["model_state_dict"], strict=True
                )
                logger.info(f"Loading ckpt: {self.args.path_weights}")

            except Exception:
                success = self.model.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
                logger.info("Warning! load_state_dict_strict is being set to False")

            logger.info(success)
            if self.args.herdnet_num_classes != self.args.herdnet_ptr_model_classes:
                logger.info(
                    "Classification head of herdnet will be modified"
                    f"to handle {self.args.herdnet_num_classes} classes."
                )
                self.model.model.reshape_classes(self.args.herdnet_num_classes)

        return self.model.to(self.args.device)

    def run(self):
        if self.model_type == "ultralytics":
            self._run_ultralytics()

        elif self.model_type == "herdnet":
            if self.herdnet_training_backend == "original":
                self._run_herdnet_original()
            else:
                self._run_herdnet_pl()

        elif self.model_type == "classifier":
            self._run_classifier()

        else:
            raise NotImplementedError

    def _train_classifier_sklearn(
        self,
    ):
        logger.info("Training classifier using extracted features...")

        # data
        datamodule = ClassifierDataModule(
            data_dir=self.args.cls_data_dir,
            batch_size=self.args.batchsize,
            num_workers=os.cpu_count() // 4,
            img_size=None,
            is_features=True,
        )
        datamodule.setup("fit")
        classes = np.array(datamodule.train_dataset.classes)
        for e in tqdm(range(self.args.epochs), desc="epochs..."):
            for features, labels in datamodule.train_dataloader():
                self.model.partial_fit(
                    X=features.cpu().numpy(),
                    y=labels.cpu().long().numpy().ravel(),
                    classes=classes,
                )
        return self.model

    def _run_classifier(
        self,
    ):
        if self.classifier_training_backend == "ultralytics":
            self._train_ultralytics(data_cfg=self.args.cls_data_dir)

        elif self.classifier_training_backend == "sk":
            mlflow.sklearn.autolog(
                log_models=True, log_datasets=False, log_model_signatures=True
            )
            try:
                with mlflow.start_run() as run:
                    self._train_classifier_sklearn()
            except:
                mlflow.end_run()
                with mlflow.start_run() as run:
                    self._train_classifier_sklearn()

        # using pytorch lightning
        datamodule = ClassifierDataModule(
            data_dir=self.args.cls_data_dir,
            batch_size=self.args.batchsize,
            num_workers=os.cpu_count() // 4,
            img_size=self.args.imgsz,
            is_features=self.args.cls_is_features,
            tn_ratio=self.args.cls_tn_ratio
        )

        datamodule.setup("fit")

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
        assert self.args.task in ["detect", "obb", "segment"]
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

    def _run_herdnet_original(
        self,
    ):
        from animaloc.train import Trainer

        # setting up working dir
        work_dir = self.args.herdnet_work_dir
        work_dir = Path(work_dir) / (self.args.run_name)
        work_dir.mkdir(exist_ok=True, parents=True)

        # Data
        datamodule = HerdnetData(
            data_config_yaml=self.args.yolo_yaml,
            patch_size=self.args.imgsz,
            tr_batch_size=self.args.batchsize,
            val_batch_size=self.args.herdnet_val_batchsize,
            down_ratio=self.args.herdnet_down_ratio,
            train_empty_ratio=self.args.herndet_empty_ratio,
        )
        datamodule.setup("fit")
        num_classes = datamodule.num_classes

        # -- Evaluator
        metrics = PointsMetrics(radius=20, num_classes=num_classes)
        stitcher = None
        evaluator = HerdNetEvaluator(
            model=self.model,
            dataloader=datamodule.val_dataloader(),
            metrics=metrics,
            device_name=self.args.device,
            print_freq=100,
            stitcher=stitcher,
            work_dir=work_dir,
            header="validation",
        )

        # Trainer
        optimizer = Adam(
            params=self.model.parameters(),
            lr=self.args.lr0,
            weight_decay=self.args.weight_decay,
        )
        trainer = Trainer(
            model=self.model,
            train_dataloader=datamodule.train_dataloader(),
            val_dataloader=None,
            valid_freq=self.args.herdnet_valid_freq,
            print_freq=100,
            lr_milestones=self.args.herdnet_lr_milestones,
            optimizer=optimizer,
            auto_lr=True,
            device_name=self.args.device,
            num_epochs=self.args.epochs,
            evaluator=evaluator,
            work_dir=work_dir,
        )

        # train
        self.model = trainer.start(
            warmup_iters=self.args.herdnet_warmup_iters,
            checkpoints="best",
            select="max",
            validate_on="f1_score",
        )

    def _run_herdnet_pl(self):
        # lowerng matrix multiplication precision
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        accelerator = "auto"
        detect_anomaly = False

        normalization = "standard"  # "standard or min_max
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        check_val_every_n_epoch = 3
        num_sanity_val_steps = 10

        work_dir = Path(self.args.herdnet_work_dir)  # for HerdNet Trainer
        work_dir.mkdir(exist_ok=True, parents=False)

        # get cross entropy loss weights
        # Data
        datamodule = HerdnetData(
            data_config_yaml=self.args.yolo_yaml,
            patch_size=self.args.imgsz,
            tr_batch_size=self.args.batchsize,
            val_batch_size=self.args.herdnet_val_batchsize,
            down_ratio=self.args.herdnet_down_ratio,
            train_empty_ratio=self.args.herndet_empty_ratio,
        )
        datamodule.setup("fit")

        if self.args.herdnet_pl_ckpt is not None:
            herdnet_trainer = HerdnetTrainer.load_from_checkpoint(
                checkpoint_path=self.args.herdnet_pl_ckpt,
                lr=self.args.lr0,
                map_location=self.args.device,
                weight_decay=self.args.weight_decay,
                data_config_yaml=self.args.yolo_yaml,
                work_dir=work_dir,
            )

            logger.info(f"\nLoading checkpoint at {self.args.herdnet_pl_ckpt}\n")
        else:
            # Training logic
            herdnet_trainer = HerdnetTrainer(
                data_config_yaml=self.args.yolo_yaml,
                model=self.model,
                lr=self.args.lr0,
                weight_decay=self.args.weight_decay,
                work_dir=work_dir,
            )

        # continuous learning
        for empty_ratio, lr, freeze_ratio, epochs in zip(
            self.args.cl_ratios,
            self.args.cl_lr0s,
            self.args.cl_freeze,
            self.args.cl_epochs,
            strict=False,
        ):
            self.args.run_name = (
                self.args.run_name
                + f"-emptyRatio_{empty_ratio}-freezeRatio_{freeze_ratio}"
            )

            self._train_herdnet_pl(
                herdnet_trainer=herdnet_trainer,
                lr=lr,
                epochs=epochs,
                workdir=work_dir / self.args.run_name,
                freeze_ratio=freeze_ratio,
                empty_ratio=empty_ratio,
                normalization=normalization,
                mean=mean,
                std=std,
                num_sanity_val_steps=num_sanity_val_steps,
                check_val_every_n_epoch=check_val_every_n_epoch,
                detect_anomaly=detect_anomaly,
                accelerator=accelerator,
            )

    def _train_herdnet_pl(
        self,
        herdnet_trainer: L.LightningModule,
        lr: float,
        epochs: int,
        freeze_ratio: float,
        empty_ratio: float,
        normalization: tuple,
        mean: tuple,
        std: tuple,
        workdir: str,
        num_sanity_val_steps: int = 10,
        check_val_every_n_epoch: int = 3,
        detect_anomaly: bool = False,
        accelerator: str = "auto",
    ) -> L.LightningModule:
        # loggers and callbacks
        mlf_logger = MLFlowLogger(
            experiment_name=self.args.project_name,
            run_name=self.args.run_name,
            tracking_uri=self.args.mlflow_tracking_uri,
            log_model=True,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=workdir,
            monitor="val_f1-score",
            mode="max",
            filename="best.ckpt",
            save_weights_only=True,
            save_last=True,
            save_top_k=1,
        )
        lr_callback = LearningRateMonitor(logging_interval="epoch")
        callbacks = [
            checkpoint_callback,
            lr_callback,
            EarlyStopping(
                monitor="val_f1-score",
                patience=self.args.patience,
                min_delta=1e-4,
                mode="max",
            ),
        ]

        herdnet_trainer.hparams.lr = lr
        herdnet_trainer.hparams.epochs = epochs
        herdnet_trainer.hparams.lrr = self.args.lrf

        # Freeze params
        num_layers = len(list(herdnet_trainer.parameters()))
        for idx, param in enumerate(herdnet_trainer.parameters()):
            if idx / num_layers < freeze_ratio:
                param.requires_grad = False
            else:
                break
        logger.info(f"\n{int(num_layers * freeze_ratio)} layers have been frozen.\n")

        # Data
        datamodule = HerdnetData(
            data_config_yaml=self.args.yolo_yaml,
            patch_size=self.args.imgsz,
            tr_batch_size=self.args.cl_batch_size,
            val_batch_size=self.args.herdnet_val_batchsize,
            down_ratio=self.args.herdnet_down_ratio,
            train_empty_ratio=empty_ratio,
            normalization=normalization,  #
            mean=mean,
            std=std,
        )

        # Trainer
        trainer = L.Trainer(
            num_sanity_val_steps=num_sanity_val_steps,
            logger=mlf_logger,
            max_epochs=epochs,
            check_val_every_n_epoch=check_val_every_n_epoch,
            # accumulate_grad_batches=max(int(64 / args.batchsize), 1),
            precision="16-mixed",
            callbacks=callbacks,
            # gradient_clip_val=10,
            # gradient_clip_algorithm="value",
            detect_anomaly=detect_anomaly,
            accelerator=accelerator,
        )
        trainer.fit(
            model=herdnet_trainer,
            datamodule=datamodule,
        )

        # Reset param.requires_grad
        if freeze_ratio > 0:
            for param in herdnet_trainer.parameters():
                param.requires_grad = True

        return herdnet_trainer

    def _train_ultralytics(
        self, data_cfg=None, imgsz=None, batchsize=None, resume=False
    ):
        args = self.args

        assert args.val in ["True", "False"]

        cfg = dict()
        # TODO: debug
        # if not self.args.is_rtdetr:
        #     os.environ["pos_weight"] = json.dumps(self.args.ultralytics_pos_weight)
        #     cfg = dict(trainer=CustomTrainer)

        self.model.train(
            data=data_cfg or args.yolo_yaml,
            epochs=args.epochs,
            imgsz=imgsz or args.imgsz,
            device=args.device,
            freeze=args.freeze,
            name=args.run_name,
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
        hn_cfg_path = get_data_cfg_paths_for_HN(args=args, data_config_yaml=cfg_path)
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
