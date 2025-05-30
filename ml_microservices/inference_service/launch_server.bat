call deactivate

call cd D:\datalabeling\ml_microservices\inference_service

call .venv-inference\Scripts\activate


call set MODEL_NAME=labeler
call set MODEL_ALIAS=demo
call set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
call set AWS_ACCESS_KEY_ID=minioadmin
call set AWS_SECRET_ACCESS_KEY=minioadmin
call set MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
call set WEIGHTS_PATH=./model
call set NUM_WORKERS=1
call set OVERLAP_RATIO=0.2
call set NMS_IOU=0.5
call set BATCH_SIZE=8
call set TILE_SIZE=800

call python main.py

@REM call python app.py
