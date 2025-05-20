import logging

from datargs import parse
import os
# import mlflow

# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
# from ultralytics import settings
# # Update a setting
# settings.update({"mlflow": True})
# mlflow.set_tracking_uri("file:///c:/Users/Machine Learning/Desktop/workspace-wildAI/datalabeling/runs/mlflow")
# mlflow.set_tracking_uri("http://localhost:5000")

from datalabeling.common.config import TrainingConfig
from datalabeling.ml.train import TrainingManager
from datalabeling.common.io import load_yaml

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    args = parse(TrainingConfig)

    
    handler = TrainingManager(
        args=args,
        herdnet_loss=None,
        herdnet_training_backend="pl",  # original or pl
        classifier_training_backend="pl",  # sk, pl, ultralytics
        model_type="ultralytics",
    )
    
    handler.run()
