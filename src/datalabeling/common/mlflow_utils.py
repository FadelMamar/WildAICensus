import mlflow
import torch

from ..ml import Detector


def load_registered_model(
    alias,
    name,
    tag_to_append: str = "",
    mlflow_tracking_url="http://localhost:5000",
    load_unwrapped: bool = False,
):
    mlflow.set_tracking_uri(mlflow_tracking_url)

    client = mlflow.MlflowClient()

    version = client.get_model_version_by_alias(name=name, alias=alias).version
    modelversion = f"{name}:{version}" + tag_to_append
    modelURI = f"models:/{name}/{version}"

    model = mlflow.pyfunc.load_model(modelURI)

    metadata = dict(version=modelversion)
    metadata.update(model.metadata.metadata)

    if load_unwrapped:
        try:
            model = model.unwrap_python_model().model
        except:
            try:
                model = model.unwrap_python_model().detection_model
            except:
                model = model.unwrap_python_model().classifier

    return model, metadata


def get_experiment_id(name: str):
    """Gets mlflow experiments id

    Args:
        name (str): mlflow experiment name

    Returns:
        str: experiment id
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id
