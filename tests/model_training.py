import os, json

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

from ultralytics import settings
import torch

# Update a setting
settings.update({"mlflow": False})

from datalabeling.common.config import TrainingConfig
from datalabeling.common.io import load_yaml

# from datalabeling.common.pipeline import ModelTrainingStep, Pipeline
from datalabeling.ml.train import TrainingManager
from pathlib import Path


def test_training_service(
    args: TrainingConfig,
    url: str = "http://localhost:5500/train",
    yolo_arch_yaml: str = "yolov8n-p2.yaml",
):
    import requests
    import base64

    train_config = {k: v for k, v in vars(args).items() if "herdnet" not in k}

    for key in ["yolo_yaml", "yolo_arch_yaml"]:
        if train_config[key]:
            p = train_config[key]
            file_name = Path(p).name
            if key == "yolo_arch_yaml":
                file_name = yolo_arch_yaml
            train_config[key] = (load_yaml(p), file_name)

    if train_config["path_weights"]:
        with open(train_config["path_weights"], "rb") as file:
            weight_data = base64.b64encode(file.read()).decode("utf-8")
        train_config["path_weights"] = weight_data

    payload = dict(train_config=train_config)

    # print(json.dumps(train_config,indent=2))

    res = requests.post(url=url, json=payload).json()

    print(res)


def test_training_routine(
    args: TrainingConfig,
):
    handler = TrainingManager(
        args=training_cfg,
        herdnet_loss=None,
        herdnet_training_backend="pl",  # original or pl
    )
    handler.run()


if __name__ == "__main__":
    ## Training configs
    training_cfg = TrainingConfig()
    # training_cfg.herdnet_work_dir = r"D:\datalabeling\.tmp"
    # training_cfg.run_name = "debug"
    # training_cfg.imgsz = 640
    # training_cfg.batchsize = 4
    # training_cfg.epochs = 15
    # training_cfg.herndet_empty_ratio = 0.0
    # training_cfg.herdnet_lr_milestones = [15, 25]
    # training_cfg.herdnet_val_batchsize = 4

    training_cfg.yolo_yaml = (
        r"D:\datalabeling\configs\yolo_configs\data\data_config.yaml"
    )
    training_cfg.yolo_arch_yaml = r"..\configs\yolo_configs\models\yolo11s.yaml"
    training_cfg.path_weights = r"../runs/mlflow/140168774036374062/045bfab3be854d68a0227eae07da35cc/artifacts/weights/best.pt"

    # training_cfg.mlflow_model_alias = "pt"
    # training_cfg.mlflow_model_name = "labeler"

    training_cfg.cls_label_smoothing = 0.0
    training_cfg.cls_num_classes = 3

    # training_cfg.cls_data_dir = r"..\.tmp\cls-features"
    # training_cfg.cls_is_features = True
    # training_cfg.project_name = "classifier"
    # training_cfg.run_name = "demo-RoIClassifier"
    # training_cfg.cls_training_backend = "pl"

    training_cfg.imgsz = 800  # not for cls_is_features

    training_cfg.model_type = "detector"  # detector, classifier, herdnet

    training_cfg.batchsize = 16
    training_cfg.epochs = 1

    training_cfg.task = "detect"  # ultralytics
    training_cfg.project_name = "wildAI-detection"
    training_cfg.run_name = "yolo11s-custom-fptp"

    training_cfg.lr0 = 3e-4
    training_cfg.lrf = 1e-2
    training_cfg.patience = 20

    training_cfg.object_detector_arch = "yolo"  # "yolo", "rtdetr", "custom_yolo"
    training_cfg.custom_yolo_kwargs = dict(
        count_regressor_layers=19,  # p4
        area_regressor_layers=16,
        mask_p3_layer_indx=16,
        mask_loss_weight=0.0,
        roi_classifier_layers={"p3": 16, "p4": 19},
        fp_tp_loss_weight=0.5,
        count_loss_weight=0.5,
        area_loss_weight=0.0,
        roi_scale_factor=[
            2.0,
        ],
    )

    training_cfg.ultralytics_pos_weight = None
    training_cfg.weight_decay = 5e-4

    training_cfg.warmup_epochs = 0
    training_cfg.dfl = 1.5  # 1.5
    training_cfg.cls = 0.5  # 0.5
    training_cfg.box = 7.5  # 7.5

    training_cfg.cl_batch_size = (32,)
    training_cfg.use_continual_learning = True
    training_cfg.cl_ratios = (2.5,)  # ratio = num_empty/num_non_empty
    training_cfg.cl_epochs = (15,)
    training_cfg.cl_freeze = (11,)
    training_cfg.cl_lr0s = (5e-5,)
    training_cfg.cl_save_dir = (
        r"D:\PhD\Data per camp\DetectionDataset\continuous_learning"
    )
    training_cfg.cl_data_config_yaml = str(Path(training_cfg.yolo_yaml).resolve())
    training_cfg.cl_batch_size = training_cfg.batchsize

    training_cfg.device = "cpu"  # "cuda:0"

    handler = TrainingManager(
        args=training_cfg,
        herdnet_loss=None,
        herdnet_training_backend="pl",  # original or pl
        classifier_training_backend="pl",  # sk, pl, ultralytics
        model_type="ultralytics",
    )

    # test_training_service(args=training_cfg)
