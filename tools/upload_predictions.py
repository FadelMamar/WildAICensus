import fire
from typing import Sequence
from datalabeling.common.config import PredictionConfig
from datalabeling.ml.interface import load_engine


# TODO debug
class Uploader:
    def __init__(
        self,
        imgsz=800,
        tilesize=800,
        overlap_ratio=0.2,
        confidence_threshold=0.2,
        inference_service_url=None,
        roi_classifier_path: str = r"..\base_models_weights\roi_classifier.ckpt",
        # min_area=100,
        # max_area=None,
        cls_imgsz=98,
        device="cuda:0",
    ):
        self.config = PredictionConfig(
            imgsz=imgsz,
            tilesize=tilesize,
            overlap_ratio=overlap_ratio,
            confidence_threshold=confidence_threshold,
            inference_service_url=inference_service_url,
            # min_area=100,
            # max_area=None,
            cls_imgsz=cls_imgsz,
            device=device,
        )

        self.roi_classifier_path = roi_classifier_path

    def upload(
        self,
        project_id: int,
        aliases: Sequence[str],
        mlflow_model_name: str = "labeler",
        roi_cls_label_map: dict = {0: "gt", 1: "tn"},
        roi_keep_classes: list = ["gt"],
        detection_label_map: dict = {0: "wildlife"},
        feature_extractor_path="facebook/dinov2-with-registers-small",
        dot_env_path: str = "../.env",
    ):
        for alias in aliases:
            annotator, _ = load_engine(
                pred_config=self.config,
                roi_classifier_path=self.roi_classifier_path,
                roi_cls_is_features=True,
                roi_cls_label_map=roi_cls_label_map,
                roi_keep_classes=roi_keep_classes,
                detection_label_map=detection_label_map,
                feature_extractor_path=feature_extractor_path,
                detection_model=None,
                mlflow_model_alias=alias,
                mlflow_model_name=mlflow_model_name,
                set_ls_client=True,
                dot_env_path=dot_env_path,
            )

            annotator.upload_predictions(project_id=project_id)


if __name__ == "__main__":
    fire.Fire(Uploader)
