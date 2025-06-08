call deactivate

call cd D:\datalabeling\ml_microservices\inference_service

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\ml_microservices\inference_service"
call "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\.venv-export\Scripts\activate"


call set MODEL_NAME=labeler
call set MODEL_ALIAS=demo
call set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
call set AWS_ACCESS_KEY_ID=minioadmin
call set AWS_SECRET_ACCESS_KEY=minioadmin
call set MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
call set NUM_WORKERS=1
call set BATCH_SIZE=1
call set INFERENCE_PORT=4141
call python main.py

@REM call python app.py
