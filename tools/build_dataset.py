import fire
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
from datalabeling.common.io import load_datasets
import os, traceback

import os
from datalabeling.ml.interface import load_engine
from dotenv import load_dotenv
import os


# TODO
def create_classification_data(
    yaml_path: str,
    strategies: list[str] = ["gt", "hn"],
    alias="demo",
):
    from datalabeling.common.config import EvaluationConfig, PredictionConfig
    from datalabeling.common.io import load_yaml
    from datalabeling.common.dataset_loader import (
        ClassificationDatasetBuilder,
    )

    eval_config = EvaluationConfig()
    eval_config.score_threshold = 0.25
    eval_config.map_threshold = 0.3
    eval_config.uncertainty_method = "entropy"
    eval_config.uncertainty_threshold = 4
    eval_config.score_col = "max_scores"
    eval_config.tp_iou_threshold = 0.2
    eval_config.load_results = (
        False  # Set to True to load existing predictions if applicable
    )

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

    handler = ClassificationDatasetBuilder(
        eval_config,
    )

    yaml_path = r"..\configs\yolo_configs\data\data_config.yaml"
    cfg = load_yaml(yaml_path)

    root_dir = r"D:\datalabeling\.tmp\cls-features"

    engine, feature_extractor = load_engine(pred_config)

    for split in ["train", "val"]:
        source_dirs = [os.path.join(cfg["path"], subset) for subset in cfg[split]]

        print(f"source_dirs: {source_dirs}")

        handler.set_dirs(
            source_dirs=source_dirs, output_dir=os.path.join(root_dir, split)
        )

        handler.run(
            strategies=strategies,
            save_true_negatives=True,
            feature_extractor=feature_extractor,
            detector=engine,
            bbox_resize_factor=1,
            tn_kwargs=dict(w=pred_config.cls_imgsz, h=pred_config.cls_imgsz, number=3),
            tp_kwargs=dict(w=pred_config.cls_imgsz, h=pred_config.cls_imgsz),
            hn_kwargs=dict(w=pred_config.cls_imgsz, h=pred_config.cls_imgsz),
        )


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
