
call cd D:\datalabeling\ml_microservices\inference_service

call .venv-inference\Scripts\activate

@REM call set RAY_ADDRESS="auto"
@REM call set RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
@REM call ray start --head --include-dashboard=false --port 4141

@REM call serve run main:entrypoint

call set MODEL_NAME=labeler
call set MODEL_ALIAS=demo
call set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
call python main.py

@REM call python app.py
