from datalabeling.common.config import TrainingConfig
from datalabeling.common.pipeline import ModelTrainingStep, Pipeline
from datalabeling.ml.train import TrainingManager


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
    training_cfg.yolo_arch_yaml = (
        r"D:\datalabeling\configs\yolo_configs\models\yolov8-rtdetr-p1-p4.yaml"
    )
    training_cfg.path_weights = r"D:\datalabeling\base_models_weights\best.pt"

    # training_cfg.ultralytics_pos_weight = 10.0

    training_cfg.cls_label_smoothing = 0.
    training_cfg.cls_num_classes = 2
    training_cfg.weight_decay = 5e-4

    training_cfg.cls_data_dir = r"D:\PhD\Data per camp\Classification\cls-features"
    training_cfg.cls_is_features = True

    training_cfg.imgsz = 640  # not for cls_is_features

    training_cfg.batchsize = 32
    training_cfg.epochs = 15

    training_cfg.task = "detect"  # ultralytics

    training_cfg.is_rtdetr = False

    training_cfg.lr0 = 3e-4
    training_cfg.lrf = 1e-2
    training_cfg.patience = 20

    training_cfg.project_name = "classifier"
    training_cfg.run_name = "demo-RoIClassifier"

    handler = TrainingManager(
        args=training_cfg,
        herdnet_loss=None,
        herdnet_training_backend="pl",  # original or pl
        classifier_training_backend="pl",  # sk, pl, ultralytics
        model_type="classifier",
    )
    
    handler.run()
