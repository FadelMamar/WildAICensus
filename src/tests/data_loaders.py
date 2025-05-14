# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 19:29:12 2025

@author: FADELCO
"""

from tqdm import tqdm
# from datalabeling.common.pipeline import ClassificationDataExport


def load_herd_net():
    from datalabeling.common.io import HerdnetData

    data_config_yaml = r"D:\datalabeling\configs\yolo_configs\data_config.yaml"
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


def load_classification_data():
    from datalabeling.common.config import EvaluationConfig
    from datalabeling.ml.models import Detector
    from datalabeling.common.dataset_loader import (
        ClassificationDatasetBuilder,
        FeatureExtractor,
    )

    eval_config = EvaluationConfig()
    eval_config.score_threshold = 0.25
    eval_config.map_threshold = 0.3
    eval_config.uncertainty_method = "entropy"
    eval_config.uncertainty_threshold = 4
    eval_config.score_col = "max_scores"
    eval_config.tp_iou_threshold = 0.2

    # =============================================================================
    eval_config.load_results = (
        True  # Set to True to load existing predictions if applicable
    )
    # =============================================================================

    detector = Detector(
        path_to_weights=r"D:\datalabeling\base_models_weights\best.pt",
        confidence_threshold=eval_config.score_threshold,
        overlap_ratio=0.2,
        tilesize=800,
        imgsz=800,
        use_sliding_window=False,
        # device="cpu",
    )

    handler = ClassificationDatasetBuilder(
        eval_config,
    )

    source_dirs = [
        # r"D:\PhD\Data per camp\DetectionDataset\delplanque_tiled_data\train_tiled\images",
        # r"D:\PhD\Data per camp\DetectionDataset\delplanque_tiled_data\val_tiled\images",
        # r"D:\PhD\Data per camp\DetectionDataset\WAID\val\images",
        # r"D:\PhD\Data per camp\DetectionDataset\savmap\images",
        r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images",
        # r"D:\general_dataset\tiled-data\val\images",
        r"D:\general_dataset\tiled-data\test\images",
    ]

    handler.set_dirs(
        source_dirs=source_dirs, output_dir=r"D:\datalabeling\.tmp\cls-features\train"
    )

    handler.run(
        strategy="gt",
        save_true_negatives=True,
        feature_extractor=FeatureExtractor(),
        detector=detector,
        bbox_resize_factor=1,  # resizes the bbox for tn,tp,fp
        tn_kwargs=dict(w=96, h=96, number=3),  # to disable use {}
        tp_kwargs=dict(w=96, h=96),  # or {} to use actual bbox
    )


def load_classification_features_data():
    from datalabeling.common.io import ClassifierDataModule

    data = ClassifierDataModule(
        data_dir=r"D:\datalabeling\.tmp\cls-features",
        batch_size=32,
        is_features=True,
        img_size=96,
    )

    data.setup("fit")

    for tr_batch in tqdm(data.train_dataloader(), desc="train loader"):
        pass

    for val_batch in tqdm(data.val_dataloader(), desc="train loader"):
        pass

    # print("labels_map: ", data.labels_map)

    # feature, label = next(iter(loader))

    # return tr_batch, val_batch


if __name__ == "__main__":
    pass

    # load_classification_data()

    # load_classification_features_data()
