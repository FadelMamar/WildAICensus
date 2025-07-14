call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

@REM call helper-scripts\activate_label-backend_env.bat

call .venv-export\Scripts\activate

@REM --use-sliding-window adding this flag will enabled sahi inference

call set MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
call set AWS_SECRET_ACCESS_KEY=minioadmin
call set AWS_ACCESS_KEY_ID=minioadmin

@REM call python tools\register_model.py register_detector "runs/mlflow/140168774036374062/a59eda79d9444ff4befc561ac21da6b4/artifacts/weights/best.pt"^
@REM         "labeler" "pt" 960 32 "cuda" "http://localhost:5000" "False" "detect"

call python tools\register_model.py register_classifier classifier\best.ckpt-v6.ckpt^
        2 "True" 8 128 384 "classifier" "http://localhost:5000"


call deactivate
