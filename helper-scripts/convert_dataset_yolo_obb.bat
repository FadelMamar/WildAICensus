call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

call helper-scripts\activate_label-backend_env.bat

:: yolo to obb
@REM call python  tools\build_dataset.py --yolo-to-obb --data-config-yaml "data\dataset_conversion.yaml" --skip

:: obb to yolo
call python  tools\build_dataset.py --obb-to-yolo --data-config-yaml "data\dataset_labeler.yaml" --skip

call deactivate
