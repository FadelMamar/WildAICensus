call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call deactivate

@REM call helper-scripts\activate_label-backend_env.bat

call .venv-export\Scripts\activate

@REM --use-sliding-window adding this flag will enabled sahi inference

call set MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
call set AWS_SECRET_ACCESS_KEY=minioadmin
call set AWS_ACCESS_KEY_ID=minioadmin

call python tools\register_model.py register_detector "runs/mlflow/140168774036374062/f5b7124be14c4c89b8edd26bcf7a9a76/artifacts/weights/best.pt"^
        "labeler" "engine" 960 1 "cuda:0" "http://localhost:5000" "False" "detect"





call deactivate
