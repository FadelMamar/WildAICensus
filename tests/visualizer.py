# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 14:42:46 2025

@author: FADELCO
"""

from datalabeling.common.dataset_loader import LabelingDataset
from datalabeling.common.visualizer import FiftyOneVisualizer


def load_dataset():
    images_dirs = [
        r"D:\workspace\data\savmap_dataset_v2\raw\images",
    ]

    dataset = LabelingDataset.from_yolo(
        images_dirs=images_dirs,
        paths=None,
        load_empty=True,
        max_workers=2,
        label_map={0: "wildlife"},
    )

    return dataset


if __name__ == "__main__":
    # dataset = load_dataset()

    # data = dataset.data

    images_dirs = [
        r"D:\workspace\data\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images",
    ]

    dataset = LabelingDataset.from_yolo(
        images_dirs=images_dirs,
        paths=None,
        load_empty=True,
        max_workers=2,
        label_map={0: "wildlife"},
    )

    visualizer = FiftyOneVisualizer(
        dataset=dataset, dataset_name="savmap_labeled", persistent=False
    )

    # visualizer.run(port=5151)

    visualizer.add_images()

    # while True:

    #     continue
