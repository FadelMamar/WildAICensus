# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 19:29:12 2025

@author: FADELCO
"""

# import fire

from tqdm import tqdm
import os


def load_herd_net():
    from datalabeling.common.io import HerdnetData

    data_config_yaml = r"..\configs\yolo_configs\data_config.yaml"
    patch_size = 640
    batchsize = 4
    down_ratio = 2
    train_empty_ratio = 0.0

    datamodule = HerdnetData(
        data_config_yaml=data_config_yaml,
        patch_size=patch_size,
        tr_batch_size=batchsize,
        val_batch_size=1,
        down_ratio=down_ratio,
        train_empty_ratio=train_empty_ratio,
    )

    datamodule.setup("fit")
    num_classes = datamodule.num_classes

    for batch_train in tqdm(
        datamodule.train_dataloader(), desc="Iterating thru train_dataloader"
    ):
        continue

    for batch_val in tqdm(
        datamodule.val_dataloader(), desc="Iterating thru val_dataloader"
    ):
        continue


def create_classification_data(strategies: list[str] = ["gt", "hn"], alias="demo"):
    from datalabeling.common.config import EvaluationConfig, PredictionConfig
    from datalabeling.ml.models import Detector
    from datalabeling.common.io import load_yaml
    from datalabeling.common.mlflow_utils import load_registered_model
    from datalabeling.common.dataset_loader import (
        ClassificationDatasetBuilder,
    )
    from datalabeling.common.processor import get_processor

    eval_config = EvaluationConfig()
    eval_config.score_threshold = 0.25
    eval_config.map_threshold = 0.3
    eval_config.uncertainty_method = "entropy"
    eval_config.uncertainty_threshold = 4
    eval_config.score_col = "max_scores"
    eval_config.tp_iou_threshold = 0.2

    pred_config = PredictionConfig(
        imgsz=800,
        tilesize=800,
        overlap_ratio=0.2,
        confidence_threshold=0.2,
        # min_area=100,
        # max_area=None,
        cls_imgsz=128,
        # device="cpu",
    )

    # =============================================================================
    eval_config.load_results = (
        True  # Set to True to load existing predictions if applicable
    )
    # =============================================================================

    detection_model, model_version = load_registered_model(
        alias=alias,
        name="labeler",
        mlflow_tracking_url="http://localhost:5000",
        load_unwrapped=True,
    )

    detector = Detector(detection_model=detection_model, config=pred_config)

    handler = ClassificationDatasetBuilder(
        eval_config,
    )

    feature_extractor = get_processor("feature_extractor")(
        hf_model_path="facebook/dinov2-with-registers-small"
    )

    yaml_path = r"..\configs\yolo_configs\data\data_config.yaml"
    cfg = load_yaml(yaml_path)

    root_dir = r"..\.tmp\cls-features-1"

    for split in ["train", "val"]:
        source_dirs = [os.path.join(cfg["path"], subset) for subset in cfg[split]]

        handler.set_dirs(
            source_dirs=source_dirs, output_dir=os.path.join(root_dir, split)
        )

        handler.run(
            strategies=strategies,
            save_true_negatives=True,
            feature_extractor=feature_extractor,
            detector=detector,
            bbox_resize_factor=1,  # resizes the bbox for tn,tp,fp
            tn_kwargs=dict(
                w=pred_config.cls_imgsz, h=pred_config.cls_imgsz, number=3
            ),  # to disable use {}
            tp_kwargs=dict(
                w=pred_config.cls_imgsz, h=pred_config.cls_imgsz
            ),  # or {} to use actual bbox
            hn_kwargs=dict(w=pred_config.cls_imgsz, h=pred_config.cls_imgsz),
        )


def load_classification_features_data():
    from datalabeling.common.io import ClassifierDataModule

    data = ClassifierDataModule(
        data_dir=r"D:\PhD\Data per camp\Classification\cls-features",
        batch_size=64,
        is_features=True,
        img_size=96,
        tn_ratio=1.0,
    )

    data.setup("fit")

    for tr_batch in tqdm(data.train_dataloader(), desc="train loader"):
        pass

    for val_batch in tqdm(data.val_dataloader(), desc="train loader"):
        pass

    # print("labels_map: ", data.labels_map)

    # feature, label = next(iter(loader))

    # return tr_batch, val_batch


def load_dataset_from_ls(
    data_dir: str, project_id=4, top_n=0, load_existing_metadata=True
):
    from label_studio_sdk.client import LabelStudio
    from dotenv import load_dotenv
    import os
    from datalabeling.common.config import TilingConfig
    from datalabeling.common.dataset_loader import LabelingDataset

    # # Load environment variables
    load_dotenv(dotenv_path="../.env")

    # # label studio client
    LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
    API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
    labelstudio_client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

    # project = labelstudio_client.projects.get(id=project_id)

    # if data_dir is None:
    #     data_dir = labelstudio_client.import_storage.local.get(project_id).path

    # collect tile metadata: gps coords
    config = TilingConfig(
        root=data_dir,
        overlapfactor=0.1,
        ratiowidth=0.5,
        ratioheight=0.33,
        rmheight=0.1,
        rmwidth=0.1,
        flight_height=180,
        sensor_height=7.4,
        gsd=2.26,
        dest=r"D:\savmap_dataset_v2\slipts_tmp",
        save_coords_only=True,  # set to False to save tiles i.e. patches
    )

    print(config)
    dataset = LabelingDataset.from_ls(
        labelstudio_client,
        project_id=project_id,
        config=config,
        top_n=top_n,
        load_existing_metadata=load_existing_metadata,
    )

    return dataset


if __name__ == "__main__":
    #     fire.Fire(
    #         {
    #             "create-features": create_classification_data,
    #             "load-features": load_classification_features_data,
    #             "load-herdnet": load_herd_net,
    #         }
    #     )

    # create_classification_data()

    # load_classification_features_data()

    dataset = load_dataset_from_ls(
        project_id=4,
        top_n=0,
        load_existing_metadata=False,
        data_dir=r"D:\savmap_dataset_v2\raw\images",
    )

    data = dataset.data
    gps_data = dataset.export_detections_gps()
