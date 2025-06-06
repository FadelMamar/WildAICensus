import json
import logging
import os
import traceback
from pathlib import Path

import yaml
from datargs import parse
from dotenv import load_dotenv

from datalabeling.common.config import DataConfig, LabelConfig
from datalabeling.common.pipeline import (
    LabelstudioToYolo,
    ObbToDotaStep,
    Pipeline,
    YoloToObbStep,
    ObbToYoloStep,
)
from datalabeling.common.io import load_yaml
import os, traceback


def load_datasets(data_config_yaml: str) -> list[str]:
    data_config = load_yaml(data_config_yaml)
    paths = list()
    root = data_config["path"]
    for split in ["train", "val", "test"]:
        try:
            for p in data_config[split]:
                path = os.path.join(root, p)
                paths.append(path)
        except Exception as e:
            print(f"Failed to load datasets for conversion {split} --> ", e)

    return paths


if __name__ == "__main__":
    load_dotenv(r"..\.env")

    logger = logging.getLogger(__name__)

    args = parse(DataConfig)

    # print(args)

    # TODO: update
    if args.build_yolo_dataset:
        pass
        # build_yolo_dataset(args=args)
        # print("Saving arguments to destination directory")
        # save_path = Path(args.dest_path_images).parent / "dataset_configs.json"
        # with open(save_path, "w") as file:
        #     configs = [
        #         "is_detector",
        #         "discard_labels",
        #         "ls_json_dir",
        #         "keep_labels",
        #         "coco_json_dir",
        #         "dest_path_images",
        #         "dest_path_labels",
        #         "clear_yolo_dir",
        #         "height",
        #         "width",
        #         "overlap_ratio",
        #         "save_all",
        #         "parse_ls_config",
        #         "min_visibility",
        #         "empty_ratio",
        #     ]
        #     configs = dict(
        #         zip(configs, [args.__dict__[k] for k in configs], strict=False)
        #     )
        #     json.dump(configs, file, indent=2)

    assert (args.yolo_to_obb + args.obb_to_yolo) < 2, "Both arguments can't be True"

    # convert yolo dataset to obb

    paths = load_datasets(args.data_config_yaml)
    steps = []
    data_config = load_yaml(args.data_config_yaml)

    for p in paths:
        try:
            p_new = p.replace("images", "labels")

            if args.yolo_to_obb or args.obb_to_dota:
                steps.append(
                    YoloToObbStep(
                        yolo_labels_dir=p_new,
                        obb_labels_dir=p_new,
                        skip=True,
                    )
                )

            if args.obb_to_yolo:
                steps.append(
                    ObbToYoloStep(
                        obb_labels_dir=p_new,
                        yolo_labels_dir=p_new,
                        skip=True,
                    )
                )

            if args.obb_to_dota:
                labels_output_dir = Path(p_new).parent / "dota_labels"
                steps.append(
                    ObbToDotaStep(
                        obb_img_dir=p_new,
                        dota_dir=labels_output_dir,
                        label_map=data_config["names"],
                        skip=True,
                        clear_old=args.clear_dota_labels,
                    ),
                )

            pipeline = Pipeline(steps)
            result_ctx = pipeline.run()

        except Exception:
            logger.warning(f"Failed for {p_new}")
            traceback.print_exc()

    # create yolo-seg labels
    if args.create_yolo_seg_dir:
        from ultralytics import SAM

        model_sam = SAM(args.sam_model_path)
        create_yolo_seg_directory(
            data_config_yaml=args.data_config_yaml,
            model_sam=model_sam,
            device=args.device,
            copy_images_dir=args.copy_images,
        )

    if args.yolo_to_coco:
        from ultralytics.data.dataset import YOLOConcatDataset, YOLODataset

        from datalabeling.train.utils import remove_label_cache

        with open(args.data_config_yaml, "r") as file:
            data_config = yaml.load(file, Loader=yaml.FullLoader)

        remove_label_cache(args.data_config_yaml)

        for split in ["val", "test", "train"]:
            datasets = list()
            try:
                for path in data_config[split]:
                    images_path = os.path.join(data_config["path"], path)
                    dataset = YOLODataset(
                        img_path=images_path,
                        task="detect",
                        data={"names": data_config["names"]},
                        augment=False,
                        imgsz=args.imgsz,
                        classes=None,
                    )
                    datasets.append(dataset)
                datasets = YOLOConcatDataset(datasets)

                convert_yolo_to_coco(
                    datasets,
                    output_dir=args.coco_output_dir,
                    data_config=data_config,
                    split=split,
                    clear_data=args.clear_coco_dir,
                )
            except Exception as e:
                print(e)
                continue
