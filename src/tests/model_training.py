from datalabeling.common.config import TrainingConfig
from datalabeling.common.pipeline import ModelTraining, Pipeline
import torch


if __name__ == "__main__":
    
    torch.set_float32_matmul_precision('high')
    
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

    # training_cfg.yolo_yaml = (
    #     r"D:\datalabeling\configs\yolo_configs\data\data_config.yaml"
    # )
    # training_cfg.yolo_arch_yaml = (
    #     r"D:\datalabeling\configs\yolo_configs\models\yolov8-ghost-p2.yaml"
    # )
    # training_cfg.path_weights = None

    # training_cfg.ultralytics_pos_weight = 10.0
    
    training_cfg.project_name = 'classifier'
    training_cfg.run_name = 'debug'

    training_cfg.cls_label_smoothing = 0.0
    training_cfg.cls_num_classes = 2
    
    training_cfg.cls_auto_augment = 'augmix'
    
    # training_cfg.cls_train_dir = (
    #     r"D:\PhD\Data per camp\Classification\train"
    # )
    # training_cfg.cls_val_dir = r"D:\PhD\Data per camp\Classification\val"
    
    training_cfg.cls_data_dir = r"D:\PhD\Data per camp\Classification"
    
    training_cfg.path_weights = r"C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\base_models_weights\yolo11s-cls.pt"
    
    training_cfg.imgsz = 96
    training_cfg.batchsize = 64
    training_cfg.epochs = 10
    training_cfg.lr0 = 1e-3

    training_step = ModelTraining(
        training_cfg=training_cfg,
        herdnet_loss=None,
        herdnet_training_backend="pl",  # original or pl
        model_type="classifier",
    )

    pipe = Pipeline(
        steps=[
            training_step,
        ]
    )

    # uncomment to run
    pipe.run()
