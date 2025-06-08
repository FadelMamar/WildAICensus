import os

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

from ultralytics import settings

# Update a setting
settings.update({"mlflow": True})

from datalabeling.common.config import TrainingConfig
from datalabeling.common.pipeline import ModelTrainingStep, Pipeline
from datalabeling.ml.train import TrainingManager
from pathlib import Path

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
        r"..\configs\yolo_configs\data\dataset_identification-detection.yaml"
    )
    training_cfg.yolo_arch_yaml =  r"..\configs\yolo_configs\models\yolo11s.yaml"
    training_cfg.path_weights = r"../runs/mlflow/140168774036374062/f5b7124be14c4c89b8edd26bcf7a9a76/artifacts/weights/best.pt"

    # training_cfg.cls_label_smoothing = 0.
    # training_cfg.cls_num_classes = 2    

    # training_cfg.cls_data_dir = r"D:\PhD\Data per camp\Classification\cls-features"
    # training_cfg.cls_is_features = True
    # training_cfg.project_name = "classifier"
    # training_cfg.run_name = "demo-RoIClassifier"

    training_cfg.imgsz = 800  # not for cls_is_features

    training_cfg.batchsize = 16
    training_cfg.epochs = 15

    training_cfg.task = "detect"  # ultralytics
    training_cfg.project_name = "wildAI-detection"
    training_cfg.run_name = "yolo-custom-heads"


    training_cfg.lr0 = 3e-4
    training_cfg.lrf = 1e-1
    training_cfg.patience = 20

    training_cfg.object_detector_arch = "custom_yolo"  # "yolo", "rtdetr", "custom_yolo"
    training_cfg.custom_yolo_kwargs = dict(
        count_regressor_layers=22,  # p5
        area_regressor_layers=16,
        roi_classifier_layers={"p3": 16, "p4": 19},
        fp_tp_loss_weight=3.,
        count_loss_weight=1.,
        area_loss_weight=1.0,
        roi_scale_factor=[
            2.0,
        ],
    )

    training_cfg.ultralytics_pos_weight = 5.
    training_cfg.weight_decay = 5e-4

    training_cfg.warmup_epochs = 0
    training_cfg.dfl = 1.5 # 1.5
    training_cfg.cls = .5 # 0.5
    training_cfg.box = 7.5 # 7.5

    training_cfg.cl_batch_size = (32,)
    training_cfg.use_continual_learning = True
    training_cfg.cl_ratios = (5.,)  # ratio = num_empty/num_non_empty
    training_cfg.cl_epochs = (20,)
    training_cfg.cl_freeze = (11,)
    training_cfg.cl_lr0s = (1e-4,)
    training_cfg.cl_save_dir = (
        r"D:\PhD\Data per camp\DetectionDataset\continuous_learning"
    )
    training_cfg.cl_data_config_yaml = Path(training_cfg.yolo_yaml).resolve()
    training_cfg.cl_batch_size = training_cfg.batchsize

    training_cfg.device = "cuda:0"

    handler = TrainingManager(
        args=training_cfg,
        herdnet_loss=None,
        herdnet_training_backend="pl",  # original or pl
        classifier_training_backend="pl",  # sk, pl, ultralytics
        model_type="ultralytics",
    )

    handler.run()
