from datalabeling.common.config import DataConfig, LabelConfig
from datalabeling.common.dataset_loader import (
    LabelHandler,
    ClassificationDatasetBuilder,
)
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



def ls_to_yolo():
    ## ---- Creating yolo dataset from Label studio labels
    dataset_config = DataConfig()
    dataset_config.dest_path_images = (
        r"D:\savmap_dataset_v2\annotated_py_paul\yolo_format\images"
    )
    dataset_config.dest_path_labels = (
        r"D:\savmap_dataset_v2\annotated_py_paul\yolo_format\labels"
    )

    dataset_config.clear_output = False

    dataset_config.coco_json_dir = r"D:\savmap_dataset_v2\annotated_py_paul\coco-format"
    dataset_config.ls_json_dir = r"D:\savmap_dataset_v2\annotated_py_paul\labelstudio"
    dataset_config.parse_ls_config = False
    dataset_config.load_coco_annotations = True

    dataset_config.slice_height = 640
    dataset_config.slice_width = 640
    dataset_config.min_area_ratio = 0.8
    dataset_config.empty_ratio = 1.0

    dataset_config.is_single_cls = True
    dataset_config.yolo_data_config_yaml = r"D:\datalabeling\configs\yolo_configs\data_config.yaml"  # needed for multiclass

    dataset_config.dotenv_path = r"D:\datalabeling\.env"

    dataset_config.save_all = False
    dataset_config.save_only_empty = False

    label_config = LabelConfig()
    label_config.label_map = r"D:\datalabeling\exported_annotations\label_mapping.json"
    label_config.keep = ("wildlife",)
    label_config.discard = ("other",)

    # pipeline = DataPreparation(dataset_config, label_config)
    # # -- uncomment to run
    # pipeline.run(ls_xml_config=r"D:\datalabeling\exported_annotations\label_studio_config.xml",
    #              image_dir=r"D:\savmap_dataset_v2\annotated_py_paul\images_splits",
    #              ls_client=None)

    steps = [
        LabelstudioToYolo(
            dataset_config=dataset_config,
            label_config=label_config,
            ls_xml_config=r"D:\datalabeling\exported_annotations\label_studio_config.xml",
            ls_client=None,
            image_dir=r"D:\savmap_dataset_v2\annotated_py_paul\images_splits",
        ),
    ]

    pipeline = Pipeline(steps)
    result_ctx = pipeline.run()


def yolo_to_obb_dota():
    label_handler = LabelHandler(config=LabelConfig())
    label_handler.config.label_map = (
        r"..\exported_annotations\label_mapping.json"
    )
    label_handler.load_map()
    
    
    data_config_yaml = r"..\configs\yolo_configs\data\dataset_identification-detection.yaml"
    
    paths = load_datasets(data_config_yaml)
    
    for p in paths:
        
        try:
            
            p = p.replace("images", "labels")
            

            steps = [
                # YoloToObbStep(
                #     yolo_labels_dir=r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\labels",
                #     obb_labels_dir=r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\labels",
                #     skip=True,
                # ),
                
                ObbToYoloStep(
                    obb_labels_dir=p,
                    yolo_labels_dir=p,
                    skip=True,
                )
                # ObbToDotaStep(
                #     obb_img_dir=r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images",
                #     dota_dir=r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format_dota",
                #     label_map={
                #         0: "wildlife",
                #     },
                #     skip=True,
                #     clear_old=False,
                # ),
            ]

            pipeline = Pipeline(steps)
            result_ctx = pipeline.run()
    
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    print("--" * 20)

    # # --- Reformatting a coco-path converted by label-studio converter
    # reformat_coco_file_paths(coco_path=r"D:\savmap_dataset_v2\annotated_py_paul\Savmap v2 splits-coco-original.json",
    #                          save_path=r"D:\savmap_dataset_v2\annotated_py_paul\Savmap v2 splits-coco-filtered.json",
    #                          img_dir=r"D:\savmap_dataset_v2\annotated_py_paul\images_splits"
    #                          )

    ## Uncomment to run -> converts Label studio data -> COCO -> YOLO
    # ls_to_yolo()

    ## Uncomment to run
    yolo_to_obb_dota()
