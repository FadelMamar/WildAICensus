

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
    training_cfg.yolo_arch_yaml = (
        r"..\configs\yolo_configs\models\yolov8s-p2.yaml"
    )
    training_cfg.path_weights = "../runs/mlflow/140168774036374062/f5b7124be14c4c89b8edd26bcf7a9a76/artifacts/weights/best.pt"

    training_cfg.ultralytics_pos_weight = 10.0

    # training_cfg.cls_label_smoothing = 0.
    # training_cfg.cls_num_classes = 2
    
    training_cfg.weight_decay = 5e-4

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

    training_cfg.is_rtdetr = False

    training_cfg.lr0 = 3e-4
    training_cfg.lrf = 1e-2
    training_cfg.patience = 20
    
    training_cfg.cl_batch_size = (32,)
    training_cfg.use_continual_learning = True
    training_cfg.cl_ratios = (1.0,)  # ratio = num_empty/num_non_empty
    training_cfg.cl_epochs = (20,)
    training_cfg.cl_freeze = (0,)
    training_cfg.cl_lr0s = (1e-4,)
    training_cfg.cl_save_dir = r"D:\PhD\Data per camp\DetectionDataset\continuous_learning"
    training_cfg.cl_data_config_yaml = Path(training_cfg.yolo_yaml).resolve()
    training_cfg.cl_batch_size = training_cfg.batchsize
    
    training_cfg.device = 'cuda:0'    

    handler = TrainingManager(
        args=training_cfg,
        herdnet_loss=None,
        herdnet_training_backend="pl",  # original or pl
        classifier_training_backend="pl",  # sk, pl, ultralytics
        model_type="ultralytics",
    )
    
    handler.run()
