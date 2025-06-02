
call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat


@REM HerdNet
@REM --engine original, lightning
call uv run tools\validate_herdnet.py --name "herdnet-identif" --batch-size 1 --data-config "configs\yolo_configs\dataset_identification.yaml" --splits "val" ^
            --imgsz 800 --weights "mlartifacts/934358897506090439/b93b0e55010348be89f55bb27b99fd91/artifacts/epoch=11-step=1740/epoch=11-step=1740.ckpt" ^
            --save-dir ".\runs_herdnet" --print-freq 500 --engine "original"


@REM Ultralytics

@REM yolov5s
@REM call python  tools\build_dataset.py --obb-to-yolo --data-config-yaml "configs\yolo_configs\dataset_identification-detection.yaml" --skip
@REM call python  tools\build_dataset.py --obb-to-yolo --data-config-yaml "configs\yolo_configs\dataset_identification.yaml" --skip

@REM call uv run tools\validate.py --splits "test" --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/e5e3bf93d34f48f1bb7d0a648530bb45/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.25 --name "yolo5-su#Det#mlflow-e5e3bf93d34f48f1bb7d0a648530bb45" --batch-size 16 --max-det 100 ^
@REM     --project-name "runs_yolo/new-test" --plots

@REM call uv run tools\validate.py --splits "test" --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/a59eda79d9444ff4befc561ac21da6b4/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.32 --name "yolo12-s#Det#mlflow-a59eda79d9444ff4befc561ac21da6b4" --batch-size 16 --max-det 100 ^
@REM     --project-name "runs_yolo/new-test" --plots

@REM call uv run tools\validate.py --splits "test" --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/d5533934c0084419ba2e0d136537d302/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.4 --name "yolo12-X#Det#mlflow-d5533934c0084419ba2e0d136537d302" --batch-size 16 --max-det 100 ^
@REM     --project-name "runs_yolo/new-test" --plots

@REM call uv run tools\validate.py --splits "test" --data-config "configs\yolo_configs\dataset_identification.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/87718ce84ce04dacac6ab8c92328eae7/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.2 --name "yolo5-su#Identif#mlflow-87718ce84ce04dacac6ab8c92328eae7" --batch-size 16 --max-det 100 ^
@REM     --project-name "runs_yolo/new-test" --plots

@REM call uv run tools\validate.py --splits "test" --data-config "configs\yolo_configs\dataset_identification.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/757205b53b454043a0a6481efb72fddd/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.2 --name "yolo12-X#Identif#mlflow-757205b53b454043a0a6481efb72fddd" --batch-size 16 --max-det 100 ^
@REM     --project-name "runs_yolo/new-test" --plots

@REM call uv run tools\validate.py --splits "test" --data-config "configs\yolo_configs\dataset_identification.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/e0ea49b51ce34cfe9de6b482a2180037/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.32 --name "yolo12-s#Identif#mlflow-e0ea49b51ce34cfe9de6b482a2180037" --batch-size 16 --max-det 100 ^
@REM     --project-name "runs_yolo/new-test" --plots


@REM call python  tools\build_dataset.py --yolo-to-obb --data-config-yaml "configs\yolo_configs\dataset_identification-detection.yaml" --skip
@REM call python  tools\build_dataset.py --yolo-to-obb --data-config-yaml "configs\yolo_configs\dataset_identification.yaml" --skip

@REM  YOLO-obb
@REM # --augment enables TestTime augmentation

@REM Detection

@REM call uv run tools\validate.py --splits "test" --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/4291304920cd40c28ff8456684045983/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.28 --name "yolo8-obb-s#Det#mlflow-4291304920cd40c28ff8456684045983" --batch-size 16 --max-det 100 ^
@REM     --project-name "runs_yolo/new-test" --plots

@REM call uv run tools\validate.py --splits "test" --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/f5b7124be14c4c89b8edd26bcf7a9a76/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.32 --name "yolo11-obb-s#Det#mlflow-f5b7124be14c4c89b8edd26bcf7a9a76" --batch-size 16 --max-det 100 ^
@REM     --project-name "runs_yolo/new-test" --plots

@REM call uv run tools\validate.py --splits "test" --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/685cbdce2d9b42658fa3ecb19c48261c/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.39 --name "yolo11-obb-X#Det#mlflow-685cbdce2d9b42658fa3ecb19c48261c" --batch-size 16 --max-det 100 ^
@REM     --project-name "runs_yolo/new-test" --plots


@REM @REM Identif
@REM call uv run tools\validate.py --splits "test" --data-config "configs\yolo_configs\dataset_identification.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/b883bd2b31f94f29807ea3b94e8ff8fc/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.2 --name "yolo8-obb-s#Identif#mlflow-b883bd2b31f94f29807ea3b94e8ff8fc" --batch-size 16 --max-det 100 ^
@REM     --project-name "runs_yolo/new-test" --plots

@REM call uv run tools\validate.py --splits "test" --data-config "configs\yolo_configs\dataset_identification.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/34c709364c0e46dcb72c526de34a7fa4/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.39 --name "yolo11-obb-s#Identif#mlflow-34c709364c0e46dcb72c526de34a7fa4" --batch-size 16 --max-det 100 ^
@REM     --project-name "runs_yolo/new-test" --plots

@REM call uv run tools\validate.py --splits "test" --data-config "configs\yolo_configs\dataset_identification.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/31df73dd8793452595f359ff3c7deb02/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.32 --name "yolo11-obb-X#Identif#mlflow-31df73dd8793452595f359ff3c7deb02" --batch-size 16 --max-det 100 ^
@REM     --project-name "runs_yolo/new-test" --plots



@REM call uv run tools\validate.py --splits "test" --is-detector --data-config "configs\yolo_configs\dataset_identification-detection.yaml" ^
@REM     --weights "runs/mlflow/140168774036374062/f5b7124be14c4c89b8edd26bcf7a9a76/artifacts/weights/best.pt" ^
@REM     --imgsz 800 --iou-threshold 0.6 --conf-threshold 0.25 --name "yolo11-obb-s#mlflow-f5b7124be14c4c89b8edd26bcf7a9a76" --batch-size 32 --max-det 100 ^
@REM     --project-name "runs_yolo" --save-txt --plots



@REM MMROTATE
call deactivate

call .venv-mmrotate\Scripts\activate.bat

@REM call uv run tools\validate_mmrotate.py "runs-mmrotate\oriented_rcnn_r50_fpn\oriented_rcnn_r50_fpn_identif#empty_0.1#freeze_0.75\oriented_rcnn_r50_fpn_1x_dota_le90.py" "runs-mmrotate\oriented_rcnn_r50_fpn\oriented_rcnn_r50_fpn_identif#empty_0.1#freeze_0.75\best_mAP_epoch_12.pth" ^
@REM         --eval "mAP" --work-dir "runs-mmrotate\test\oriented_rcnn_r50_fpn\oriented_rcnn_r50_fpn_identif#empty_0.1#freeze_0.75" --fuse-conv-bn

@REM call uv run tools\mmrotate_confusion_matrix.py "runs-mmrotate\oriented_rcnn_r50_fpn\oriented_rcnn_r50_fpn_identif#empty_0.1#freeze_0.75\oriented_rcnn_r50_fpn_1x_dota_le90.py" ^
@REM                                                 "runs-mmrotate\test\oriented_rcnn_r50_fpn\oriented_rcnn_r50_fpn_identif#empty_0.1#freeze_0.75\results.pkl" ^
@REM                                                 "runs-mmrotate\test\oriented_rcnn_r50_fpn\oriented_rcnn_r50_fpn_identif#empty_0.1#freeze_0.75" ^
@REM                                                 --score-thr 0.3 --tp-iou-thr 0.6 --nms-iou-thr 0.5

@REM call uv run tools\validate_mmrotate.py "runs-mmrotate\roi_trans_r50_fpn\roi_trans_r50_fpn_identif#empty_0.1#freeze_0.75\roi_trans_r50_fpn_1x_dota_ms_rr_le90.py" "runs-mmrotate\roi_trans_r50_fpn\roi_trans_r50_fpn_identif#empty_0.1#freeze_0.75\best_mAP_epoch_12.pth" ^
@REM         --eval "mAP" --work-dir "runs-mmrotate\test\roi_trans_r50_fpn\roi_trans_r50_fpn_identif#empty_0.1#freeze_0.75" --fuse-conv-bn

@REM call uv run tools\mmrotate_confusion_matrix.py "runs-mmrotate\roi_trans_r50_fpn\roi_trans_r50_fpn_identif#empty_0.1#freeze_0.75\roi_trans_r50_fpn_1x_dota_ms_rr_le90.py" ^
@REM                                                 "runs-mmrotate\test\roi_trans_r50_fpn\roi_trans_r50_fpn_identif#empty_0.1#freeze_0.75\results.pkl" ^
@REM                                                 "runs-mmrotate\test\roi_trans_r50_fpn\roi_trans_r50_fpn_identif#empty_0.1#freeze_0.75"  ^
@REM                                                 --score-thr 0.3 --tp-iou-thr 0.6 --nms-iou-thr 0.5

@REM call uv run tools\validate_mmrotate.py "runs-mmrotate\redet_re50_fpn\redet_re50_fpn_identif#empty_0.1#freeze_0.75\redet_re50_refpn_1x_dota_le90.py" "runs-mmrotate\redet_re50_fpn\redet_re50_fpn_identif#empty_0.1#freeze_0.75\best_mAP_epoch_13.pth" ^
@REM         --eval "mAP" --work-dir "runs-mmrotate\test\redet_re50_fpn\redet_re50_fpn_identif#empty_0.1#freeze_0.75" --fuse-conv-bn

@REM call uv run tools\mmrotate_confusion_matrix.py "runs-mmrotate\redet_re50_fpn\redet_re50_fpn_identif#empty_0.1#freeze_0.75\redet_re50_refpn_1x_dota_le90.py" ^
@REM                                                 "runs-mmrotate\test\redet_re50_fpn\redet_re50_fpn_identif#empty_0.1#freeze_0.75\results.pkl" ^
@REM                                                 "runs-mmrotate\test\redet_re50_fpn\redet_re50_fpn_identif#empty_0.1#freeze_0.75" ^
@REM                                                 --score-thr 0.3 --tp-iou-thr 0.6 --nms-iou-thr 0.5





@REM @REM  PaddlePaddle Detection
call deactivate

call .venv-paddle\Scripts\activate


@REM --json_eval  to compute metrics from json


@REM call uv run tools\validate_ppd.py -c "configs\ppd_configs\ppyoloe_plus_sod_crn_s_80e_visdrone.yml" ^
@REM         -o use_gpu=true weights="runs_ppd\ppyoloe_plus_sod_crn_s_80e_visdrone_identif-tr_empty_ratio_0.1_freeze_0.75\best_model\model.pdparams" PPYOLOEHead.nms.score_threshold=0.4 EvalReader.batch_size=16 ^
@REM         --classwise --amp --output_eval "runs_ppd\test\ppyoloe_plus_sod_crn_s_80e_visdrone_identif-tr_empty_ratio_0.1_freeze_0.75"

@REM call uv run tools\validate_ppd.py -c "configs\ppd_configs\ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.yml" ^
@REM         -o use_gpu=true weights="runs_ppd\ppyoloe_plus_sod_crn_l_largesize_80e_visdrone_identif-tr_empty_ratio_0.1_freeze_0.75\best_model\model.pdparams" PPYOLOEHead.nms.score_threshold=0.4 EvalReader.batch_size=16 ^
@REM         --classwise --amp --output_eval "runs_ppd\test\ppyoloe_plus_sod_crn_l_largesize_80e_visdrone_identif-tr_empty_ratio_0.1_freeze_0.75"
