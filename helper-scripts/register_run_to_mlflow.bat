call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

@REM call helper-scripts\activate_label-backend_env.bat

call .venv-export\Scripts\activate

@REM --use-sliding-window adding this flag will enabled sahi inference

call set MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
call set AWS_SECRET_ACCESS_KEY=minioadmin
call set AWS_ACCESS_KEY_ID=minioadmin

call python tools\register_model.py register_detector "D:\datalabeling\base_models_weights\best.pt"^
        "labeler" "torchscript" 800 8 "cpu" "http://localhost:5000" "False" "detect"

@REM call python tools\register_model.py register_classifier "D:\datalabeling\base_models_weights\roi_classifier.ckpt"^
@REM         2 "True" 8 128 384 "classifier" "http://localhost:5000"


call deactivate
