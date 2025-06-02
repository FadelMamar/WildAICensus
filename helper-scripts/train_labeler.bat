call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat

:: set to 'offline' when having issues with internet, otherwise use 'online'
call wandb offline

@REM :: freeze layers for transfer-learning
@REM :: --freeze 22 @REM ::--mlflow-model-alias "last"^

@REM    use_continual_learning:bool=False
@REM     cl_ratios:Sequence[float]=(1.0,)
@REM     cl_epochs:Sequence[int]=(20,)
@REM     cl_freeze:Sequence[int]=(None,)
@REM     cl_lr0s:Sequence[float]=(1e-5,)
@REM     cl_save_dir:str=None # should be given!
@REM    cl_data_config_yaml:str=None # should be given!

@REM    use_hn_learning:bool=False, Default params below
@REM hn_save_dir:str=None
@REM hn_data_config_yaml:str=None
@REM hn_num_epochs:int=10
@REM hn_freeze:int=20
@REM hn_lr0:float=1e-4
@REM hn_lrf:float=1e-2
@REM hn_batch_size:int=32
@REM hn_is_yolo_obb:bool=False
@REM hn_use_sliding_window:bool=True
@REM hn_overlap_ratio:float=0.2
@REM hn_map_thrs:float=0.35 # mAP threshold. lower than it is considered sample of interest
@REM hn_score_thrs:float=0.7
@REM hn_confidence_threshold:float=0.1
@REM hn_ratio:int=20 # ratio = num_empty/num_non_empty. Higher allows to look at all saved empty images
@REM hn_uncertainty_thrs:float=5 # helps to select those with high uncertainty
@REM hn_uncertainty_method:str="entropy"

@REM :: obb

call wandb offline

@REM @REM convert datasets to yolo-obb
@REM call uv run  tools\build_dataset.py --yolo-to-obb --data-config-yaml "data\dataset_labeler.yaml" --skip

@REM call uv run tools\cli.py --start-training --batchsize 16  --weight-decay 0.0001 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.1 --patience 15 --is-detector ^
@REM     --scale 0.5 --mosaic 0.2 --copy-paste 0.2 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 2 ^
@REM     --height 800 --width 800^
@REM     --path-weights "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/771014640604815853/9ab053acdcbb486992fbe456e3701c88/artifacts/weights/best.pt" ^
@REM     --run-name "yolov11s_obb-CT" --project-name "labeler"^
@REM     --tag "continuous-training" ^
@REM     --cl-save-dir "D:\PhD\Data per camp\DetectionDataset\continuous_learning" --use-continual-learning ^
@REM     --cl-data-config-yaml "data\dataset_labeler.yaml" ^
@REM     --cl-ratios 7.5 ^
@REM     --cl-epochs 50  ^
@REM     --cl-freeze 0 ^
@REM     --cl-lr0s 0.0001 ^
@REM     --hn-save-dir "D:\PhD\Data per camp\DetectionDataset\hard_samples" --use-hn-learning ^
@REM     --hn-num-epochs 7 --hn-data-config-yaml "data\dataset_labeler.yaml" ^
@REM     --hn-is-yolo-obb

@REM --yolo-arch-yaml "configs\yolo_configs\models\yolo11-obb.yaml" --run-name "yolov8s-obb-custom-CL" ^

call set MLFLOW_TRACKING_URI=http://localhost:5000

call python tools\cli.py --batchsize 16  --weight-decay 0.005 --optimizer "AdamW" --optimizer-momentum 0.99 --lrf 0.01 --patience 20 --is-single-cls ^
    --scale 0.5 --mosaic 0.2 --copy-paste 0.0 --mixup 0.0 --rotation-degree 45. --erasing 0.0 --warmup-epochs 2 ^
    --ultralytics-pos-weight 10.0 ^
    --box 7.5 --cls 0.5 --dfl 1.5 ^
    --imgsz 800 ^
    --path-weights "runs/mlflow/140168774036374062/a59eda79d9444ff4befc561ac21da6b4/artifacts/weights/best.pt" ^
    --yolo-arch-yaml "configs\yolo_configs\models\yolov8s-p2.yaml" --run-name "yolo8s-custom-CL" ^
    --project-name "wildAI-detection"^
    --tag "CL" ^
    --cl-save-dir "D:\PhD\Data per camp\IdentificationDataset\continuous_learning" --use-continual-learning ^
    --cl-data-config-yaml "configs\yolo_configs\data\dataset_identification-detection.yaml" --cl-batch-size 16 ^
    --cl-ratios 0 1 2.5 7.5 ^
    --cl-epochs 20 20 10 7 ^
    --cl-freeze 0 0 10 23  ^
    --cl-lr0s 0.0001 0.0001 0.00005 0.00005

@REM call uv run tools\cli.py train-herdnet.py


:: Transfer learning from a model registered on mlflow
@REM call uv run tools\cli.py --start-training --batchsize 16  --epochs 20 --lr0 0.0001 --weight-decay 0.05 --lrf 0.01 --patience 10 --is-detector^
@REM     --scale 0.5 --mosaic 0.3 --copy-paste 0.2 --mixup 0.0 --rotation-degree 45. --erasing 0.0^
@REM     --height 640 --width 640^
@REM     --mlflow-model-alias "last"^
@REM     --data-config-yaml "D:\PhD\Data per camp\DetectionDataset\hard_samples\hard_samples.yaml"^
@REM     --run-name "obb-detector" --project-name "wildAI-detection"^
@REM     --tag "hard_samples" "transfer-learning"


@REM :: Transfer learning from a pth file
@REM call uv run tools\cli.py --start-training --batchsize 128  --epochs 20 --lr0 0.001 --lrf 0.01 --patience 5 --is-detector^
@REM     --scale 0.0 --weight-decay 0.0005 --freeze 23^
@REM     --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\data_config.yaml" ^
@REM     --path-weights "C:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow/140168774036374062/095bc5dc29c74399b19e8f4c60fa4fa1/artifacts/weights/best.pt" ^
@REM     --run-name "yolov5mu" --project-name "wildAI-detection"^
@REM     --tag "dataset-v4" "transferlearning"^

@REM @REM Convert datasets to yolo
@REM call uv run  tools\build_dataset.py --obb-to-yolo --data-config-yaml "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\data\dataset_labeler.yaml" --skip


@REM call deactivate

:: uncomment to disable immediate shutdown
shutdown -s
