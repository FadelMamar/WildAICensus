import io
import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union, List
from urllib.parse import unquote
from typing import Sequence
import cv2
import geopy
import numpy as np
import pandas as pd

# import torch
import utm
import yaml
from dotenv import load_dotenv
from label_studio_sdk import Client
from PIL import Image

from sahi.utils.file import load_json
from skimage.io import imread, imsave
from tqdm import tqdm
from ultralytics import SAM
from ultralytics.data.dataset import YOLOConcatDataset, YOLODataset
from label_studio_sdk.client import LabelStudio


from .config import DataConfig
from .io import save_json, load_yolo_label

logger = logging.getLogger(__name__)


def check_label_format(loaded_df: pd.DataFrame) -> str:
    """checks label format

    Args:
        loaded_df (pd.DataFrame): target values

    Raises:
        NotImplementedError: when the format is not yolo or yolo-obb

    Returns:
        str: yolo or yolo-obb
    """

    # check bounds
    # names = list(loaded_df.columns)
    assert loaded_df.iloc[:, 1:].all().max() <= 1.0, "max value <= 1"
    assert loaded_df.iloc[:, 1:].all().min() >= 0.0, "min value >=0"

    num_features = len(loaded_df.columns)
    if num_features == 5:
        return "yolo"
    elif num_features == 9:
        return "yolo-obb"
    else:
        raise NotImplementedError(
            f"The number of features ({num_features}) in the label file is wrong. Check yolo or yolo-obb format from ultralytics."
        )


def slice_coco(
    coco_annotation_file_path: str,
    image_dir: str,
    output_coco_annotation_file_name: str,
    output_dir: Optional[str] = None,
    ignore_negative_samples: bool = False,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    min_area_ratio: float = 0.1,
    out_ext: Optional[str] = None,
    verbose: bool = False,
) -> List[Union[Dict, str]]:
    """
    from sahi
    Slice large images given in a directory, into smaller windows. If out_name is given export sliced images and coco file.

    Args:
        coco_annotation_file_pat (str): Location of the coco annotation file
        image_dir (str): Base directory for the images
        output_coco_annotation_file_name (str): File name of the exported coco
            datatset json.
        output_dir (str, optional): Output directory
        ignore_negative_samples (bool): If True, images without annotations
            are ignored. Defaults to False.
        slice_height (int): Height of each slice. Default 512.
        slice_width (int): Width of each slice. Default 512.
        overlap_height_ratio (float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        min_area_ratio (float): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix.
        verbose (bool, optional): Switch to print relevant values to screen.
            Default 'False'.

    Returns:
        coco_dict: dict
            COCO dict for sliced images and annotations
        save_path: str
            Path to the saved coco file
    """

    from sahi.utils.coco import Coco, create_coco_dict
    from sahi.slicing import slice_image
    from shapely.errors import TopologicalError

    # read coco file
    coco_dict: Dict = load_json(coco_annotation_file_path)
    # create image_id_to_annotation_list mapping
    coco = Coco.from_coco_dict_or_path(coco_dict)
    # init sliced coco_utils.CocoImage list
    sliced_coco_images: List = []

    # iterate over images and slice
    for idx, coco_image in enumerate(tqdm(coco.images, desc="slice_coco")):
        # get image path
        image_path: str = os.path.join(image_dir, coco_image.file_name)
        # get annotation json list corresponding to selected coco image
        # slice image
        try:
            slice_image_result = slice_image(
                image=image_path,
                coco_annotation_list=coco_image.annotations,
                output_file_name=f"{Path(coco_image.file_name).stem}_{idx}",
                output_dir=output_dir,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                min_area_ratio=min_area_ratio,
                out_ext=out_ext,
                verbose=verbose,
            )
            # append slice outputs
            sliced_coco_images.extend(slice_image_result.coco_images)
        except TopologicalError:
            logger.warning(
                f"Invalid annotation found, skipping this image: {image_path}"
            )

    # create and save coco dict
    coco_dict = create_coco_dict(
        sliced_coco_images,
        coco_dict["categories"],
        ignore_negative_samples=ignore_negative_samples,
    )
    save_path = ""
    if output_coco_annotation_file_name and output_dir:
        save_path = Path(output_dir) / (output_coco_annotation_file_name + "_coco.json")
        save_json(coco_dict, save_path)

    return coco_dict, save_path


def resize_bbox(factor: float, x1, x2, y1, y2, img_width, img_height):
    box_size = max(x2 - x1, y2 - y1)

    x1 = max(x1 - (factor - 1.0) * box_size / 2, 0)
    y1 = max(y1 - (factor - 1.0) * box_size / 2, 0)
    x2 = min(x2 + (factor - 1.0) * box_size / 2, img_width)
    y2 = min(y2 + (factor - 1.0) * box_size / 2, img_height)

    out = list(map(int, [x1, x2, y1, y2]))
    return out


def get_project_stats(labelstudio_client: LabelStudio, project_id: int, annotator_id=0):
    project = labelstudio_client.projects.get(id=project_id)

    images_count = dict()

    # Iterating
    tasks = labelstudio_client.tasks.list(
        project=project.id,
    )
    labels = []
    for task in tasks:
        try:
            result = task.annotations[annotator_id]["result"]
        except Exception:
            # traceback.print_exc()
            continue

        img_labels = []
        for annot in result:
            img_labels = annot["value"]["rectanglelabels"] + img_labels
        labels = labels + img_labels

        # update stats holder
        for label in set(img_labels):
            if label in images_count.keys():
                images_count[label] += 1
            else:
                images_count[label] = 1

    instances = {
        f"{k}": [
            labels.count(k),
        ]
        for k in set(labels)
    }
    images_count = {
        k: [
            v,
        ]
        for k, v in images_count.items()
    }

    instances_count = pd.DataFrame.from_dict(instances)
    images_count = pd.DataFrame.from_dict(images_count)

    return instances_count, images_count


def convert_yolo_to_obb(
    yolo_labels_dir: str, output_dir: str, skip: bool = True
) -> None:
    """Converts labels in yolo format to Oriented Bounding Box (obb) format.

    Args:
        yolo_labels_dir (str): directory with txt files following yolo format
        output_dir (str): output directory. It's a directory with txt files following guidelines at https://docs.ultralytics.com/datasets/obb/
    """

    cols = ["id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    names = ["id", "x", "y", "w", "h"]

    if not Path(yolo_labels_dir).exists():
        raise FileNotFoundError("Directory does not exist.")

    if not Path(output_dir).exists():
        logger.info(f"Creating directory {output_dir}.")
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Iterate through labels
    skip_count = 0
    count = 0
    for label_path in tqdm(Path(yolo_labels_dir).glob("*.txt"), desc="yolo->obb"):
        df = pd.read_csv(label_path, sep=" ", header=None)

        # check format is yolo
        if check_label_format(loaded_df=df) == "yolo":
            df.columns = names
            count += 1
        else:
            if skip:
                skip_count += 1
                continue
            else:
                raise ValueError(f"{label_path} does not follow yolo format.")

        for col in names[1:]:
            df[col] = df[col].astype(float)
        df = df.astype({"id": "int32"})

        df["w"] = 0.5 * df["w"]
        df["h"] = 0.5 * df["h"]
        # top left
        df["x1"] = df["x"] - df["w"]
        df["y1"] = df["y"] - df["h"]
        # top right
        df["x2"] = df["x"] + df["w"]
        df["y2"] = df["y"] - df["h"]
        # bottom right
        df["x3"] = df["x"] + df["w"]
        df["y3"] = df["y"] + df["h"]
        # bottom left
        df["x4"] = df["x"] - df["w"]
        df["y4"] = df["y"] + df["h"]

        # check bounds
        assert df[names[1:]].all().max() <= 1.0, "max value <= 1"
        assert df[names[1:]].all().min() >= 0.0, "min value >=0"

        # save file
        df[cols].to_csv(
            Path(output_dir) / label_path.name, sep=" ", index=False, header=False
        )

    logger.info(f"Skipping {skip_count} files.\nConverting {count} files.")
    return None


def convert_obb_to_yolo(
    obb_labels_dir: str, output_dir: str, skip: bool = True
) -> None:
    """Converts labels in Oriented Bounding Box (obb) format to yolo format

    Args:
        obb_labels_dir (str): directory with txt files following guidelines at https://docs.ultralytics.com/datasets/obb/
        output_dir (str): output directory
    """

    names = ["id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    cols = ["id", "x", "y", "w", "h"]

    if not Path(obb_labels_dir).exists():
        raise FileNotFoundError("Directory does not exist.")

    # Iterate through labels
    skip_count = 0
    count = 0
    for label_path in tqdm(Path(obb_labels_dir).glob("*.txt"), desc="obb->yolo"):
        df = pd.read_csv(label_path, sep=" ", header=None)

        # check format
        if check_label_format(loaded_df=df) == "yolo-obb":
            df.columns = names
            count += 1
        else:
            if skip:
                skip_count += 1
                continue
            else:
                raise ValueError(f"{label_path} does not follow yolo-obb format.")

        # center
        df["x"] = (df["x1"] + df["x2"]) / 2.0
        df["y"] = (df["y1"] + df["y4"]) / 2.0
        # width
        df["w"] = df["x2"] - df["x1"]
        # height
        df["h"] = df["y4"] - df["y1"]

        # check bounds
        assert df[names[1:]].all().max() <= 1.0, "max value <= 1"
        assert df[names[1:]].all().min() >= 0.0, "min value >=0"

        # make sure id is int
        df = df.astype({"id": "int32"})

        # save file
        df[cols].to_csv(
            Path(output_dir) / label_path.name, sep=" ", index=False, header=False
        )

    logger.info(f"Skipping {skip_count} files.\nConverting {count} files.")
    return None


def convert_obb_to_dota(
    obb_img_dir: str,
    output_dir: str,
    label_map: dict,
    skip: bool = True,
    clear_old_labels: bool = True,
):
    # https://github.com/open-mmlab/mmrotate/blob/main/docs/en/tutorials/customize_dataset.md

    names = ["id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    cols = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "label_name", "difficulty"]

    if not Path(obb_img_dir).exists():
        raise FileNotFoundError("Directory does not exist.")

    if Path(output_dir).exists() and clear_old_labels:
        shutil.rmtree(output_dir)
        logger.info("Deleting existing labels.")

    # create output dir
    assert str(output_dir) != str(obb_img_dir).replace("images", "labels"), (
        "Provide a directory different from yolo-obb labels."
    )
    Path(output_dir).mkdir(exist_ok=True, parents=False)

    # Iterate through labels
    skip_count = 0
    count = 0
    for img_path in tqdm(Path(obb_img_dir).glob("*"), desc="obb->dota"):
        # Load labels and check format
        label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        if not label_path.exists():
            continue
        df = pd.read_csv(label_path, sep=" ", header=None)

        # check format
        if check_label_format(loaded_df=df) == "yolo-obb":
            df.columns = names
            count += 1
        elif skip:
            skip_count += 1
            continue
        else:
            raise ValueError(f"{label_path} must follow yolo-obb format.")

        # load image dimension
        img = Image.open(img_path)
        img_width, img_height = img.size
        img.close()

        # scale annotations
        df.loc[:, names[1::2]] = df.loc[:, names[1::2]] * img_width
        df.loc[:, names[2::2]] = df.loc[:, names[2::2]] * img_height

        # add label names and difficulty
        df.sort_values(by="id", ascending=True, inplace=True)
        df["label_name"] = df["id"].map(label_map)
        df["difficulty"] = 0  # 1:difficult, 0:no_difficult

        # changing data types
        df = df.astype({k: "int32" for k in names})

        # save file
        df[cols].to_csv(
            Path(output_dir) / label_path.name, sep=" ", index=False, header=False
        )

    logger.info(f"Skipping {skip_count} files.\nConverting {count} files.")
    return None


def create_yolo_seg_directory(
    data_config_yaml: str,
    model_sam: SAM,
    device: str = "cpu",
    copy_images_dir: bool = True,
    clear: bool = True,
):
    from ultralytics.utils.ops import xywh2xyxy

    with open(data_config_yaml, "r") as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)

    # Move to device
    model_sam = model_sam.to(device)
    logger.info(f"Device: {model_sam.device}")

    # get available splits in data_config
    splits = [s for s in ["val", "train", "test"] if s in data_config.keys()]
    logger.info(f"Convertings splits: {splits}")

    # TODO: optimize, drop pd.read_csv
    def is_dir_yolo(labels_dir: str) -> None:
        for label_path in tqdm(
            Path(labels_dir).glob("*.txt"), desc="checking labels format"
        ):
            df = pd.read_csv(label_path, sep=" ", header=None)
            if check_label_format(loaded_df=df) == "yolo":
                continue
            else:
                raise ValueError("Annotations should be in the yolo format")
        return None

    def compute_save_segs(dataset: YOLODataset, save_dir: Path):
        #  Saving segmentations
        for label in tqdm(dataset.labels, desc=f"Creating yolo-seg for split={split}"):
            h, w = label["shape"]
            boxes = label["bboxes"]
            img_path = Path(label["im_file"])
            if len(boxes) == 0:  # skip empty labels
                continue
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
            im = cv2.imread(img_path)
            sam_results = model_sam(
                im, bboxes=xywh2xyxy(boxes), verbose=False, save=False, device=device
            )
            label["segments"] = sam_results[0].masks.xyn

        for label in dataset.labels:
            texts = []
            lb_name = Path(label["im_file"]).with_suffix(".txt").name
            txt_file = save_dir / lb_name
            cls = label["cls"]
            for i, s in enumerate(label["segments"]):
                if len(s) == 0:
                    continue
                line = (int(cls[i]), *s.reshape(-1))
                texts.append(("%g " * len(line)).rstrip() % line)
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines(text + "\n" for text in texts)

    def create_seg_labels_dir(images_dir: str):
        seg_dir = Path(images_dir).parent / "Segmentations"
        seg_Labels_dir = seg_dir / "labels"
        if seg_Labels_dir.exists():
            if clear:
                shutil.rmtree(seg_Labels_dir)
                logger.info(f"Deleting existing segmentation labels : {seg_Labels_dir}")
        seg_Labels_dir.mkdir(exist_ok=True, parents=True)

        if copy_images_dir:
            if (seg_dir / "images").exists():
                shutil.rmtree(seg_dir / "images")
                logger.info(f"Deleting directory: {seg_dir / 'images'}")
            shutil.copytree(images_dir, seg_dir / "images")
            logger.info(f"Copying {images_dir} into {seg_dir}")

        return seg_Labels_dir

    for split in splits:
        # Load YOLO dataset
        for path in data_config[split]:
            images_dir = os.path.join(data_config["path"], path)

            # check folder format
            is_dir_yolo(images_dir.replace("images", "labels"))

            seg_Labels_dir = create_seg_labels_dir(images_dir)

            dataset = YOLODataset(
                img_path=images_dir,
                task="detect",
                data={"names": data_config["names"]},
                augment=False,
                # imgsz=640,  # used for dataloading only
                classes=None,
            )

            compute_save_segs(dataset=dataset, save_dir=seg_Labels_dir)


def convert_yolo_to_coco(
    dataset: YOLOConcatDataset | YOLODataset,
    output_dir: str,
    data_config: dict,
    split="val",
    clear_data: bool = False,
):
    # Define the categories for the COCO dataset
    categories = [{"id": k, "name": v} for k, v in data_config["names"].items()]

    # Define the COCO dataset dictionary
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    # mkdir
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    annot_dir = output_dir / "annotations"
    annot_dir.mkdir(exist_ok=True, parents=False)
    img_dir = output_dir / f"{split}"
    img_dir.mkdir(exist_ok=True, parents=False)
    annot_save_path = annot_dir / f"annotations_{split}.json"

    if clear_data:
        try:
            shutil.rmtree(img_dir)
            img_dir.mkdir(exist_ok=True, parents=False)
            logger.info(f"Deleting: {img_dir}")
            if os.path.exists(annot_save_path):
                os.remove(annot_save_path)
            logger.info(f"Deleting: {annot_save_path} ")
        except Exception as e:
            logger.info(e)

    # Loop through the images in the input directory
    for i, data in tqdm(enumerate(dataset), desc=f"converting yolo {split} to coco"):
        # Load the image and get its dimensions
        image_path = data["im_file"]
        height, width = data["ori_shape"]

        image_file = os.path.relpath(image_path, data_config["path"])
        image_file = "#".join([p.name for p in Path(image_file).parents])
        image_file = image_file + os.path.basename(image_path)

        shutil.copyfile(image_path, img_dir / image_file)

        # Add the image to the COCO dataset
        image_id = i
        image_dict = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_file,
        }
        coco_dataset["images"].append(image_dict)

        # Loop through the annotations and add them to the COCO dataset
        for i in range(data["bboxes"].shape[0]):
            x, y, w, h = data["bboxes"][i].tolist()
            x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
            x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
            ann_dict = {
                "id": len(coco_dataset["annotations"]),
                "image_id": image_id,
                "category_id": data["cls"][i].long().item(),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "segmentation": [
                    [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
                ],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0,
            }
            coco_dataset["annotations"].append(ann_dict)

    # Save the COCO dataset to a JSON file
    with open(annot_save_path, "w") as f:
        json.dump(coco_dataset, f, indent=2)


def get_upload_img_dir(
    coco_annotation: dict,
):
    directory = [
        os.path.dirname(metadata["file_name"]) for metadata in coco_annotation["images"]
    ]

    directory = set(directory)

    if len(directory) > 1:
        logger.info(
            f"There should be one upload directory per annotation project. There are {len(directory)}={directory}. Attempting to fix it through os.path.commonprefix. Not guaranteed to work."
        )
        return os.path.commonprefix(directory)

    if len(directory) == 0:
        raise NotImplementedError("There are no labels")

    return directory.pop()


def convert_ls_json_to_coco(
    input_file: str,
    ls_xml_config: str,
    out_file_name: str = None,
    parsed_config: dict = None,
) -> Dict:
    """Converts LS json annotations to coco format

    Args:
        input_file (str): path to LS json annotation file
        out_file_name (str, optional): if not None, it will save the converted annotations. Defaults to None.

    Returns:
        dict: annotations in coco format
    """

    from label_studio_converter import Converter

    # load converter
    config_str = None
    if parsed_config is not None:
        assert ls_xml_config is None, "ls_xml_config has to be None."
        config_str = parsed_config
    elif ls_xml_config is not None:
        assert parsed_config is None, "parsed_config has to be None."
        with io.open(ls_xml_config) as f:
            config_str = f.read()
    else:
        raise NotImplementedError("")

    handler = Converter(config=config_str, project_dir=None, download_resources=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        handler.convert_to_coco(
            input_data=input_file,
            output_dir=tmp_dir,
            output_image_dir=os.path.join(tmp_dir, "images"),
            is_dir=False,
        )
        # load and update image paths
        coco_json_path = os.path.join(tmp_dir, "result.json")
        coco_annotations = load_json(coco_json_path)

        if out_file_name is not None:
            with open(out_file_name, "w") as file:
                json.dump(coco_annotations, file, indent=2)

        return coco_annotations


def reformat_coco_file_paths(coco_path: str, save_path: str, img_dir: str) -> None:
    annotations = load_json(coco_path)
    images = annotations["images"]

    updated_images = []
    for image in tqdm(images, desc="updating coco metadata"):
        file_name = os.path.basename(unquote(image["file_name"]))
        image_path = os.path.join(img_dir, file_name)
        width, height = Image.open(image_path).size
        metadata = dict(width=width, height=height, file_name=file_name, id=image["id"])
        updated_images.append(metadata)

    annotations["images"] = updated_images

    save_json(annotations, save_path)


def load_coco_annotations(dest_dir_coco: str, image_dir=None) -> dict:
    """Loads existing coco annotations

    Args:
        dest_dir_coco (str, optional): directory with annotations in coco format.

    Returns:
        dict: the schema is {uploaded_image_dir:coco_annotation_path}
    """

    coco_paths = list(Path(dest_dir_coco).glob("*.json"))
    if image_dir is not None:
        upload_img_dirs = [image_dir] * len(coco_paths)
    else:
        upload_img_dirs = [
            get_upload_img_dir(
                coco_annotation=load_json(coco_path),
            )
            for coco_path in coco_paths
        ]

    return dict(zip(upload_img_dirs, coco_paths, strict=False))


class LabelstudioConverter:
    def __init__(self, config: DataConfig):
        self.config = config

    def get_ls_parsed_config(self, ls_json_path: str, ls_client=None):
        load_dotenv(self.config.dotenv_path)

        labelstudio_client = ls_client
        if ls_client is None:
            # Connect to the Label Studio API and check the connection
            LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
            API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
            labelstudio_client = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)

        with open(ls_json_path, "r") as f:
            ls_annotation = json.load(fp=f)
        ids = set([annot["project"] for annot in ls_annotation])
        assert len(ids) == 1, "annotations come from different project. Not allowed!"
        project_id = ids.pop()
        project = labelstudio_client.get_project(id=project_id)

        return project.parsed_label_config

    def to_coco(
        self,
        input_dir: str,
        dest_dir_coco: str,
        parse_ls_config: bool,
        ls_client: Client = None,
        ls_xml_config: str = None,
    ) -> dict:
        """Converts directory with LS json files to coco format.

        Args:
            input_dir (str, optional): directory with LS json annotation files. Defaults to JSON_DIR_PATH.
            dest_dir_coco (str, optional): destination directory. It should be different from input_dir. Defaults to COCO_DIR_PATH.

        Returns:
            dict: the schema is {uploaded_image_dir:coco_annotation_path}
        """

        upload_img_dirs, coco_paths = list(), list()

        for path in Path(input_dir).glob("*.json"):
            coco_path = os.path.join(dest_dir_coco, path.name)
            parsed_config = None

            if parse_ls_config:
                parsed_config = self.get_ls_parsed_config(
                    ls_json_path=path, ls_client=ls_client
                )
            annot = convert_ls_json_to_coco(
                path,
                out_file_name=coco_path,
                parsed_config=parsed_config,
                ls_xml_config=ls_xml_config,
            )
            upload_img_dirs.append(get_upload_img_dir(coco_annotation=annot))
            coco_paths.append(coco_path)

        return dict(zip(upload_img_dirs, coco_paths, strict=False))

    def to_yolo(self, coco_dict: Dict, output_dir: Path) -> None:
        """Convert JSON to YOLO format"""
        # Implementation here


class ImageProcessor:
    @staticmethod
    def save_tiles(
        df_tiles: pd.DataFrame, output_dir: Path, clear: bool = False
    ) -> None:
        """Saves tiles (or slices of images) as .jpg files

        Args:
            df_tiles (pd.DataFrame): provides tiles boundaries.
            out_img_dir (str): output directory to save tiles
            clear (bool, optional): states if output directory should be emptied. Defaults to False.
        """

        # clear out_img_dir
        if clear:
            logger.info(f"Deleting images in {output_dir}")
            shutil.rmtree(output_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # selecting non-duplicated
        df_tiles = df_tiles[
            ~df_tiles.duplicated(["x0", "x1", "y0", "y1", "images"])
        ].copy()

        for idx in tqdm(df_tiles.index, desc=f"Saving tiles to {output_dir}"):
            x0 = int(df_tiles.at[idx, "x0"])
            x1 = int(df_tiles.at[idx, "x1"])
            y0 = int(df_tiles.at[idx, "y0"])
            y1 = int(df_tiles.at[idx, "y1"])
            img_path = df_tiles.at[idx, "parent_images"]
            tile_name = df_tiles.at[idx, "images"]
            save_path = str(
                Path(os.path.join(output_dir, tile_name)).with_suffix(".jpg")
            )
            img = imread(img_path)
            tile = img[y0:y1, x0:x1, :]
            imsave(fname=save_path, arr=tile, check_contrast=False)

    @staticmethod
    def get_slices(
        coco_path: Path,
        img_dir: Path,
        config: DataConfig,
    ) -> pd.DataFrame:
        """Generate image slices from COCO annotations"""

        sliced_coco_dict, out_coco_path = slice_coco(
            coco_annotation_file_path=coco_path,
            image_dir=img_dir,
            output_coco_annotation_file_name=None,  # os.path.join(TEMP, "sliced_coco.json"),
            ignore_negative_samples=np.isclose(config.empty_ratio, 0),
            # output_dir="",
            slice_height=config.tilesize,
            slice_width=config.tilesize,
            overlap_height_ratio=config.overlap_ratio,
            overlap_width_ratio=config.overlap_ratio,
            min_area_ratio=config.min_area_ratio,
            verbose=config.verbose,
            out_ext=".jpg",
        )

        return sliced_coco_dict

    @staticmethod
    def sample_slices(
        coco_dict_slices: dict,
        img_dir: str,
        empty_ratio: float = 3.0,
        save_all: bool = False,
        out_csv_path: str = None,
        labels_to_discard: list = None,
        labels_to_keep: list = None,
        sample_only_empty: bool = False,
    ) -> pd.DataFrame:
        """Sample annotations from sliced coco annotations

        Args:
            coco_dict_slices (dict): sliced coco annotation from .utils.get_slices
            img_dir (str): image directory
            empty_ratio (float, optional): ratio negative samples (i.e. empty images) to positive samples. Defaults to 3. It loads 3 times more empty tiles than non-empty.
            out_csv_path (str, optional): if given, it saves the sampled annotations to the path. Defaults to None.
            labels_to_discard (list, optional): labels to discard. Defaults to None.
            labels_to_keep (list, optional): labels to keep. Defaults.
            sample_only_empty (bool, optional): states if only negative samples should be saved. It is useful for hard negative sample mining. Defaults to False.

        Raises:
            FileNotFoundError: raised when a parent image can't be found for a sliced_image (i.e. tile)

        Returns:
            pd.DataFrame: sampled annotations. Columns are 'x' (center), 'y' (center), 'w', 'h', 'x_min', 'y_min', 'x_max', 'y_max' etc.
        """

        assert empty_ratio >= 0.0, "It cannot be negative."
        assert (save_all + sample_only_empty) < 2, "Both cannot be true."
        if isinstance(labels_to_discard, Sequence) and isinstance(
            labels_to_keep, Sequence
        ):
            assert len(np.intersect1d(labels_to_discard, labels_to_keep)) == 0, (
                "Some target labels are said to be both discarded and kept. Check..."
            )

        def get_parent_image(file_name: str):
            ext = ".jpg"
            file_name = Path(file_name).stem

            parent_file = "_".join(file_name.split("_")[:-5])
            p = os.path.join(img_dir, parent_file + ext)

            if os.path.exists(p):
                return p
            else:
                raise FileNotFoundError(
                    f"Parent file note found for {file_name} in {img_dir} >> {parent_file}"
                )

        # build mapping for labels
        label_ids = [cat["id"] for cat in coco_dict_slices["categories"]]
        label_name = [cat["name"] for cat in coco_dict_slices["categories"]]
        label_map = dict(zip(label_ids, label_name, strict=False))

        # build dataFrame of image slices
        ids = list()
        x0s, x1s = list(), list()
        y0s, y1s = list(), list()
        file_paths = list()
        parent_file_paths = list()
        for metadata in coco_dict_slices["images"]:
            # img_path = os.path.join(img_dir,metadata['file_name'])
            file_paths.append(metadata["file_name"])
            file_name = os.path.basename(metadata["file_name"])
            x_0, y_0, x_1, y_1 = file_name.split(".")[0].split("_")[-4:]
            parent_image = get_parent_image(file_name)
            parent_file_paths.append(parent_image)
            x0s.append(int(x_0))
            x1s.append(int(x_1))
            y0s.append(int(y_0))
            y1s.append(int(y_1))
            ids.append(metadata["id"])
        df_limits = {
            "x0": x0s,
            "x1": x1s,
            "y0": y0s,
            "y1": y1s,
            "id": ids,
            "images": file_paths,
            "parent_images": parent_file_paths,
        }
        df_limits = pd.DataFrame.from_dict(df_limits, orient="columns")
        df_limits.set_index("id", inplace=True)

        # build dataframe of annotations
        x_mins, y_mins = list(), list()
        widths, heights = list(), list()
        ids_annot = list()
        label_ids = list()
        for annot in coco_dict_slices["annotations"]:
            ids_annot.append(annot["image_id"])
            x, y, w, h = annot["bbox"]
            label_ids.append(annot["category_id"])
            x_mins.append(x)
            y_mins.append(y)
            widths.append(w)
            heights.append(h)
        df_annot = {
            "x_min": x_mins,
            "y_min": y_mins,
            "width": widths,
            "height": heights,
            "id": ids_annot,
            "label_id": label_ids,
        }
        df_annot = pd.DataFrame.from_dict(df_annot, orient="columns")
        df_annot.set_index("id", inplace=True)
        df_annot["labels"] = df_annot["label_id"].map(label_map)
        for col in ["x_min", "y_min", "width", "height"]:
            df_annot.loc[:, col] = df_annot[col].apply(math.floor)

        # join dataframes
        df = df_limits.join(df_annot, how="outer")

        # get non-empty
        df_empty = df[df["x_min"].isna()].copy()
        df_empty.drop_duplicates(subset="images", inplace=True)

        # discard selected labels
        non_empty = list()
        if labels_to_discard is not None:
            df_discard = df[~df.labels.isin(labels_to_discard)].copy()
            df_discard = df_discard.loc[~df_discard["x_min"].isna(), :]
            non_empty.append(df_discard)
        if labels_to_keep is not None:
            df_keep = df[df.labels.isin(labels_to_keep)].copy()
            df_keep = df_keep.loc[~df_keep["x_min"].isna(), :]
            non_empty.append(df_keep)

        df_non_empty = pd.concat(non_empty, axis=0).reset_index(drop=True)

        # else:
        #     df_non_empty = df[~df.labels.isin(labels_to_discard)].copy()
        #     logger.info(
        #         "sample_data function: No label is discarded, they are all kept", end="\n"
        #     )

        if len(df_non_empty) < 1:
            logger.info(
                f"No labels found in {img_dir}. Pease consider checking arguments labels_to_discard or labels_to_keep."
            )

        # get number of images to sample
        non_empty_num = df_non_empty["images"].unique().shape[0]
        empty_num = math.floor(non_empty_num * empty_ratio)
        empty_num = min(empty_num, len(df_empty))
        frac = 1.0 if save_all else empty_num / len(df_empty)

        # get empty df and tiles
        if sample_only_empty:
            df = df_empty.sample(frac=empty_num / len(df_empty))
            df.reset_index(inplace=True)
            # create x_center and y_center
            df["x"] = np.nan
            df["y"] = np.nan
            df["width"] = np.nan
            df["height"] = np.nan
        else:
            df_empty = df_empty.sample(frac=frac, random_state=41, replace=False)
            logger.info(
                f"Sampling {len(df_empty)} empty images, and {non_empty_num} non-empty images."
            )

            # concat dfs
            df = pd.concat([df_empty, df_non_empty], axis=0)
            df.reset_index(inplace=True)

            # create x_center and y_center
            df["x"] = df["x_min"] + df["width"] * 0.5
            df["y"] = df["y_min"] + df["height"] * 0.5

        # save df
        if out_csv_path is not None:
            df.to_csv(out_csv_path, sep=",", index=False)

        return df

    @staticmethod
    def get_gsd(
        image_path: str,
        image: Image.Image | None = None,
        sensor_height: float = None,
        flight_height: int = 180,
    ):
        ##-- Sensor heights
        sensor_heights = dict(ZenmuseP1=24)

        ##-- Extract exif
        exif = GPSUtils.get_exif(file_name=image_path, image=image)

        if sensor_height is None:
            sensor_height = sensor_heights.get(exif["Model"])
            if sensor_height is None:
                raise ValueError("Sensor height not found. Please provide it.")

        ##-- Compute gsd
        focal_length = exif["FocalLength"] * 0.1  # in cm
        image_height = exif["ExifImageHeight"]  # in px
        sensor_height = sensor_height * 0.1  # in cm
        flight_height = flight_height * 1e2  # in cm

        gsd = flight_height * sensor_height / (focal_length * image_height)

        return round(gsd, 3)

    @staticmethod
    def generate_pixel_coordinates(
        x, y, lat_center, lon_center, W, H, gsd=0.026
    ) -> tuple[float, float]:
        """computes (x,y) pixel gps coordinates

        Args:
            x (int): x center
            y (int): y center
            lat_center (float): latitude of center of image
            lon_center (float): longitude of center of image
            W (int): image width
            H (int): image height
            gsd (float, optional): _description_. Defaults to 0.026.

        Returns:
            tuple: latitude, longitude
        """
        # Convert center to UTM
        easting_center, northing_center, zone_num, zone_let = utm.from_latlon(
            lat_center, lon_center
        )

        # Calculate offsets
        delta_x = (x - W / 2) * gsd
        delta_y = (H / 2 - y) * gsd  # Invert y-axis

        # Compute UTM
        easting = easting_center + delta_x
        northing = northing_center + delta_y

        # Convert back to lat/lon
        lat, lon = utm.to_latlon(easting, northing, zone_num, zone_let)

        return lat, lon


class GPSUtils:
    @staticmethod
    def get_exif(file_name: str, image: Image = None) -> dict | None:
        from PIL.ExifTags import TAGS

        if image is None:
            with Image.open(file_name) as img:
                exif_data = img._getexif()
        else:
            assert isinstance(image, Image.Image), "Provide PIL Image"
            exif_data = image._getexif()

        if exif_data is None:
            return None

        extracted_exif = dict()
        for k, v in exif_data.items():
            extracted_exif[TAGS.get(k)] = v

        return extracted_exif

    @staticmethod
    def get_gps_info(labeled_exif: dict) -> dict | None:
        # https://exiftool.org/TagNames/GPS.html
        from PIL.ExifTags import GPSTAGS

        gps_info = labeled_exif.get("GPSInfo", None)

        if gps_info is None:
            return None

        info = {GPSTAGS.get(key, key): value for key, value in gps_info.items()}

        info["GPSAltitude"] = info["GPSAltitude"].__repr__()

        # convert bytes types
        for k, v in info.items():
            if isinstance(v, bytes):
                info[k] = list(v)

        return info

    @staticmethod
    def to_decimal(gps_coord: str) -> tuple:
        if gps_coord is None:
            return (None, None, None)

        lat, long, alt = geopy.Point.from_string(gps_coord)
        coords = lat, long, alt * 1e3
        return coords

    @staticmethod
    def get_gps_coord(
        file_name: str,
        image: Image = None,
        altitude: str = None,
        return_as_decimal: bool = False,
    ) -> tuple | None:
        extracted_exif = GPSUtils.get_exif(file_name=file_name, image=image)

        if extracted_exif is None:
            return None

        gps_info = GPSUtils.get_gps_info(extracted_exif)

        if gps_info is None:
            return None

        if gps_info.get("GPSAltitudeRef", None):
            altitude_map = {
                0: "Above Sea Level",
                1: "Below Sea Level",
                2: "Positive Sea Level (sea-level ref)",
                3: "Negative Sea Level (sea-level ref)",
            }

            # map GPSAltitudeRef
            try:
                gps_info["GPSAltitudeRef"] = altitude_map[gps_info["GPSAltitudeRef"]]
            except:
                gps_info["GPSAltitudeRef"] = altitude_map[gps_info["GPSAltitudeRef"][0]]

        # rewite latitude
        gps_coords = dict()
        for coord in ["GPSLatitude", "GPSLongitude"]:
            degrees, minutes, seconds = gps_info[coord]
            ref = gps_info[coord + "Ref"]
            gps_coords[coord] = f"{degrees} {minutes}m {seconds}s {ref}"

        coords = gps_coords["GPSLatitude"] + " " + gps_coords["GPSLongitude"]

        if altitude is None:
            alt = f"{gps_info.get('GPSAltitude', None)}m"
        else:
            alt = altitude

        if alt:
            coords = (
                gps_coords["GPSLatitude"] + " " + gps_coords["GPSLongitude"] + " " + alt
            )
        else:
            coords = gps_coords["GPSLatitude"] + " " + gps_coords["GPSLongitude"]
        if return_as_decimal:
            lat, long, alt = geopy.Point.from_string(coords)
            coords = lat, long, alt * 1e3

        return coords, gps_info


def compute_detection_gps(
    x_center,
    y_center,
    image: Image.Image,
    image_gps_loc: str,
    flight_height: int = 180,
    sensor_height: int = 24,
    gsd: float = None,  # cm/px
) -> str:
    # None
    if image_gps_loc is None:
        logger.info(
            "GPS coordinates of detections could not be computed because ``image_gps_loc=None``. Returning None"
        )
        return None

    assert isinstance(image, Image.Image), "Provide PIL Image"

    # compute detection
    W, H = image.size

    lat_center, lon_center, alt = GPSUtils.to_decimal(image_gps_loc)
    alt = alt * 1e-3  # converting to km
    if gsd is None:
        gsd = ImageProcessor.get_gsd(
            image=image,
            image_path=None,
            sensor_height=sensor_height,
            flight_height=flight_height,
        )

    gsd *= 1e-2  # convert to m/px

    px_lat, px_long = ImageProcessor.generate_pixel_coordinates(
        x=x_center,
        y=y_center,
        lat_center=lat_center,
        lon_center=lon_center,
        W=W,
        H=H,
        gsd=gsd,
    )

    gps_loc = str(geopy.Point(latitude=px_lat, longitude=px_long, altitude=alt))

    return gps_loc
