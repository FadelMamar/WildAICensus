call deactivate

call cd D:\datalabeling\ml_microservices\training_service
call "D:\datalabeling\.venv\Scripts\activate"

@REM call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\ml_microservices\training_service"
@REM call "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling\.venv-export\Scripts\activate"


@REM call set MODEL_NAME=labeler
@REM call set MODEL_ALIAS=pt
call set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
call set AWS_ACCESS_KEY_ID=minioadmin
call set AWS_SECRET_ACCESS_KEY=minioadmin
call set MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000

call python app/main.py
