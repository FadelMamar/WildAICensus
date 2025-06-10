import json
import logging
import os
import shutil
import traceback
from pathlib import Path
from typing import Dict, Sequence
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain, product
import math
import geopy
from urllib.parse import unquote
from label_studio_tools.core.utils.io import get_local_path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.pool import ThreadPool
from functools import partial
import fiftyone as fo
from random import shuffle
from .annotation_utils import (
    ImageProcessor,
    GPSUtils,
    LabelstudioConverter,
    load_coco_annotations,
    resize_bbox,
)
from .base import Tile, Detection

from .config import DataConfig, LabelConfig, EvaluationConfig, TilingConfig
from .io import load_yaml, DataHandler, get_images_from_dirs
from .processor import FeatureExtractor
from ..ml.models import Detector
from ..ml.workers import ObjectDetectionSystem
from ..ml.interface import InferenceEngine


logger = logging.getLogger(__name__)


class TileBuilder:
    def __init__(self, config: TilingConfig):
        self.config = config
        self._metadata = None
        self.tile_metadata = None

    @staticmethod
    def get_coordinates(
        image_width: int,
        tile_w: int,
        image_height: int,
        tile_h: int,
        overlaping_factor: float,
    ):
        # x limits
        lim = math.ceil((image_width - tile_w) / ((1 - overlaping_factor) * tile_w))
        x_right = [
            math.floor(tile_w + i * (1 - overlaping_factor) * tile_w)
            for i in range(lim)
        ]
        x_coords = [(x - tile_w, x) for x in x_right]
        if len(x_coords) > 0:
            left, right = x_coords[-1]
            x_coords[-1] = (left, image_width)  # extending to remaining pixels

        # y limits
        lim = math.ceil((image_height - tile_h) / ((1 - overlaping_factor) * tile_h))
        y_bottom = [
            math.floor(tile_h + i * (1 - overlaping_factor) * tile_h)
            for i in range(lim)
        ]
        y_coords = [(y - tile_h, y) for y in y_bottom]
        if len(y_coords) > 0:
            top, bottom = y_coords[-1]
            y_coords[-1] = (top, image_height)  # extending to remaining pixels

        # tiles coordinates
        if len(y_coords) == 0:
            y_coords = [
                (0, image_height),
            ]

        if len(x_coords) == 0:
            x_coords = [
                (0, image_width),
            ]

        coordinates = product(x_coords, y_coords)
        return list(coordinates)

    def get_patches(self, image: np.ndarray, coords: list):
        patches = list()

        # store patches
        for (x1, x2), (y1, y2) in coords:
            patches.append(image[y1:y2, x1:x2])

        return patches

    def save_list_images(self, patches: list, basename: str, dest_folder: str) -> None:
        """Save mini-batch into image files
        Args:
            batch (list): mini-batch
            basename (str) : parent image name, with extension
            dest_folder (str): destination folder path
        """

        base_wo_extension, extension = basename.split(".")[0], basename.split(".")[1]
        for i, b in enumerate(range(len(patches))):
            full_path = "_".join([base_wo_extension, str(i) + "."]) + extension
            save_path = os.path.join(dest_folder, full_path)
            cv2.imwrite(save_path, patches[b].astype("uint8"))

    def _run_single_image(self, img_path: str):
        # open image
        pil_image = Image.open(img_path)
        image_array = np.asarray(pil_image.convert("RGB"))
        img_name = os.path.basename(img_path)
        image_width = image_array.shape[1]
        image_height = image_array.shape[0]

        # Cropping out image-level overlap
        height_overlap = math.ceil(self.config.rmheight * image_array.shape[0])
        width_overlap = math.ceil(self.config.rmwidth * image_array.shape[1])

        if height_overlap * width_overlap > 0:
            image_array = image_array[
                height_overlap:-height_overlap, width_overlap:-width_overlap
            ]
            print(
                f"Removing {2 * width_overlap} pixels to the width; and {2 * height_overlap} pixels to the height."
            )
        elif (height_overlap == 0) and (width_overlap != 0):
            image_array = image_array[:, width_overlap:-width_overlap]
        elif width_overlap == 0 and (height_overlap != 0):
            image_array = image_array[height_overlap:-height_overlap, :]

        # Computes tile width and height using the given ratios
        assert (self.config.ratiowidth <= 1.0) and (self.config.ratioheight <= 1.0), (
            "The ratios should be at most 1.0"
        )
        width = image_array.shape[1]
        height = image_array.shape[0]
        if self.config.ratiowidth > 0.0:
            width = math.ceil(image_array.shape[1] * self.config.ratiowidth)
        if self.config.ratioheight > 0.0:
            height = math.ceil(image_array.shape[0] * self.config.ratioheight)

        # checking overlapfactor provided
        assert self.config.overlapfactor < 1, "It should be less than 1."

        # get tile coordinates
        coords = self.get_coordinates(
            image_width,
            tile_w=width,
            image_height=image_height,
            tile_h=height,
            overlaping_factor=self.config.overlapfactor,
        )

        # get tiles gps coordinates
        image_gps = GPSUtils.get_gps_coord(
            file_name=None, return_as_decimal=True, image=pil_image
        )
        if image_gps is not None:
            (lat, long, alt), _ = image_gps
            alt = alt / 1000  # conver to meters

            tile_gps_coords = []
            gsd = ImageProcessor.get_gsd(
                image=pil_image,
                image_path=None,
                sensor_height=self.config.sensor_height,
                flight_height=self.config.flight_height,
            )
            for (x_left, x_right), (y_top, y_bottom) in coords:
                x = (x_left + x_right) / 2
                y = (y_top + y_bottom) / 2
                gps = ImageProcessor.generate_pixel_coordinates(
                    x=x,
                    y=y,
                    W=image_width,
                    H=image_height,
                    lat_center=lat,
                    lon_center=long,
                    gsd=gsd,
                )
                gps = geopy.Point(gps[0], gps[1], alt)
                tile_gps_coords.append(str(gps))
        else:
            tile_gps_coords = [None for _ in range(len(coords))]

        # close pil image
        pil_image.close()

        # save patches
        if not self.config.save_coords_only:
            patches = self.get_patches(image_array, coords=coords)
            self.save_list_images(patches, img_name, self.config.dest)

        # record metadata
        tile_metadata = {}
        tile_metadata[str(img_path)] = dict(
            px_coordinates=coords,
            gps_coordinates=tile_gps_coords,
        )

        return tile_metadata

    def run(
        self, load_existing_metadata: bool = False, max_workers: int = 1
    ) -> dict[str, list]:
        if load_existing_metadata:
            json_path = self.config.metadata_save_path
            if json_path is None:
                json_path = Path(self.config.dest) / "metadata.json"
            try:
                with open(json_path, "r") as f:
                    self._metadata = json.load(f)

                return self._expand_metada()

            except Exception as e:
                print(e)
                print("Aborting attempt to load")

        dest = Path(self.config.dest)

        if not dest.exists():
            dest.mkdir(parents=True, exist_ok=True)

        images_paths = chain.from_iterable(
            [Path(self.config.root).glob(p) for p in self.config.patterns]
        )
        images_paths = list(set(images_paths))
        tile_metadata = dict()

        # Sequential mode
        if max_workers < 2:
            for img_path in tqdm(images_paths, desc="Exporting patches"):
                try:
                    metadata = self._run_single_image(img_path)
                    tile_metadata.update(metadata)
                except Exception as e:
                    logger.error(e)

        # TODO: add error handling
        # Parallel mode
        else:
            with ThreadPool(max_workers) as executor:
                for metadata in tqdm(
                    executor.map(self._run_single_image, images_paths),
                    desc="Exporting patches",
                ):
                    tile_metadata.update(metadata)

        # saving metadata
        json_path = self.config.metadata_save_path
        if json_path is None:
            json_path = Path(self.config.dest) / "metadata.json"
            # try:
            with open(json_path, "w") as f:
                json.dump(tile_metadata, f, indent=1)
            # except Exception as e:
            #     print(e)
            #     with open(Path(self.config.dest) / "metadata.json", "w") as f:
            #         json.dump(tile_metadata, f, indent=1)

        self._metadata = tile_metadata

        self._expand_metada()

        return self.tile_metadata

    def _expand_metada(
        self,
    ) -> dict:
        assert self._metadata is not None

        expanded_metadata = dict()

        for parent_image_path, metadata in self._metadata.items():
            stem = Path(parent_image_path).stem

            tile_coords = metadata["px_coordinates"]
            tile_gps = metadata["gps_coordinates"]
            for i, (gps, coords) in enumerate(zip(tile_coords, tile_gps)):
                expanded_metadata[f"{stem}_{i}"] = dict(
                    coordinates=tile_coords[i],
                    gps=tile_gps[i],
                    parent_image=str(parent_image_path),
                )

        self.tile_metadata = expanded_metadata
        return expanded_metadata


class LabelingDataset:
    def __init__(self, tiles: list[Tile], data: pd.DataFrame = None):
        self.tiles: list[Tile] = tiles
        self.data: pd.DataFrame = data
        self._is_built: bool = data is not None

        if self.data is not None:
            self.data = self.data.convert_dtypes()

        assert self.tiles is not None

    def add_tile(self, tile: Tile) -> None:
        self.tiles.append(tile)

    def add_predictions(self, engine: InferenceEngine, build: bool = True) -> None:
        self.tiles = engine.inference(
            tiles=self.tiles, images_paths=None, return_tiles=True
        )
        if build:
            self.build(force_rebuild=True)
        return None

    def import_data(self, path: str) -> None:
        self.data = pd.read_csv(
            path,
        )
        return None

    def build(self, force_rebuild: bool = False):
        if force_rebuild:
            pass
        elif self._is_built:
            logger.info(
                "Disabling the build...Dataset is already built. Use force_rebuild=True"
            )
            return

        data = [tile.detections_to_df() for tile in self.tiles if tile is not None]
        if len(data) < 1:
            return None

        self.data = pd.concat(data, axis=0).reset_index(drop=True).convert_dtypes()
        self._is_built = True
        return None

    def update_detection_gps(
        self, sensor_height: float, flight_height: float, gsd: float
    ):
        if gsd is None:
            logger.warning(
                "gsd=None, it might break the pipeline if the ``FocalLength`` is not available in the Exif of the image."
            )

        for tile in self.tiles:
            try:
                tile.update_detection_gps(
                    sensor_height=sensor_height, flight_height=flight_height, gsd=gsd
                )
            except Exception as e:
                logger.error(e)

        return None

    def offset_detections(self, build: bool = False):
        for tile in self.tiles:
            tile.offset_detections()

        if build:
            self.build()
        return

    def export_detections_gps(self, save_path: str = None) -> pd.DataFrame:
        df_export = self.data[["class_name", "gps_loc", "file_name"]].copy()

        df_export[["Latitude", "Longitude", "Elevation"]] = (
            df_export["gps_loc"].apply(GPSUtils.to_decimal).apply(pd.Series)
        )
        if save_path:
            df_export.to_csv(save_path, index=False)

        return df_export

    def save_data_csv(self, save_path: str):
        self.data.to_csv(save_path, index=False)
        return None

    def __len__(
        self,
    ) -> int:
        if self.data is None:
            return 0

        return len(self.data)

    def __getitem__(self, index) -> dict:
        cols = list(self.data.columns)
        # # cols.remove("file_name")
        cols.remove("score")
        # image = self.data.at[index, "file_name"]
        # image = Image.open(image)
        data = self.data.loc[index, cols].to_dict()

        return data

    @classmethod
    def from_paths(cls, paths: Sequence[str]):
        tiles = [Tile(image_path=p) for p in paths]
        o = cls(tiles=tiles, data=None)
        o.build()
        o.data["is_annot"] = None
        return o

    @classmethod
    def from_dirs(cls, images_dirs: Sequence[str]):
        paths = get_images_from_dirs(images_dirs=images_dirs)
        return cls.from_paths(paths)

    @classmethod
    def from_yolo(
        cls,
        images_dirs: Sequence[str] = None,
        paths: Sequence[str] = None,
        load_empty: bool = True,
        label_map: dict = None,
        max_workers: int = 1,
    ):
        assert (images_dirs is None) + (paths is None) == 1, (
            "Exactly one of 'paths' or 'images_dirs' should be given."
        )

        if paths is None:
            assert isinstance(images_dirs, Sequence)
            paths = get_images_from_dirs(images_dirs=images_dirs)
        else:
            assert isinstance(paths, Sequence)

        # load groundtruth
        df_labels, label_format = DataHandler.load_yolo_groundtruth(
            images_dir=None,
            images_paths=paths,
            load_empty=load_empty,
            max_workers=max_workers,
            label_map=label_map,
        )

        # load tiles
        tiles = [Tile(image_path=p) for p in df_labels["file_name"].unique()]

        df_labels["is_annot"] = True

        return cls(data=df_labels, tiles=tiles)

    @classmethod
    def from_ls(
        cls,
        labelstudio_client,
        project_id: int,
        config: TilingConfig,
        top_n=0,
        tile_metadata: dict = None,
        load_existing_metadata: bool = False,
        max_workers: int = 1,
        ls_download_resources: bool = False,
    ):
        project = labelstudio_client.projects.get(id=project_id)

        assert config.root is not None, "Provide path to untiled directory."
        # data_dir = labelstudio_client.import_storage.local.get(project_id).path
        # logger.info(f"Using root directory: {data_dir}")

        if tile_metadata is None:
            tile_metadata = TileBuilder(config=config).run(
                load_existing_metadata=load_existing_metadata
            )

        def load_unique_task(task) -> Tile | None:
            img_url = unquote(task.data["image"])

            try:
                image_path = get_local_path(
                    img_url,
                    download_resources=ls_download_resources,
                    hostname=os.getenv("LABEL_STUDIO_URL"),
                )

                # get tile gps_coords and offsets
                value = tile_metadata.get(Path(image_path).stem, None)
                tile_gps_loc = None
                x1 = y1 = None
                if value is None:
                    logger.warning(
                        f" No GPS coordinates was found when reading metadata of {img_url}. -> skipping"
                    )
                    return None

                tile_gps_loc = value["gps"]
                (x1, x2), (y1, y2) = value["coordinates"]
                parent_image = value["parent_image"]

                # build tile
                detection_objects = Detection.from_ls(task.annotations, image_path)

                tile = Tile(
                    annotations=detection_objects,
                    image_data=None,
                    image_path=image_path,
                    x_offset=x1,
                    y_offset=y1,
                    parent_image=parent_image,
                    tile_gps_loc=tile_gps_loc,
                )

                # update detections (annotations or predictions) gps loc using tile_gps_loc
                tile.update_detection_gps(
                    sensor_height=config.sensor_height,
                    flight_height=config.flight_height,
                    gsd=config.gsd,
                )
                return tile

            except Exception:
                logger.warning(f"Failed for: {img_url} -> skipping")
                traceback.print_exc()
                return None

        # get tasks in project
        tasks = labelstudio_client.tasks.list(
            project=project.id,
        )

        tiles = []
        # create
        if top_n > 0:
            max_workers = 1
        logger.info("Loading annotated dataset from Label Studio")

        # Single thread
        if max_workers < 2:
            for i, task in enumerate(tasks):
                if top_n > 0 and i >= top_n:
                    break
                tile = load_unique_task(task)
                if tile is not None:
                    tiles.append(tile)

            # build dataset
            dataset = cls(tiles=tiles)
            dataset.build()
            return dataset

        # Multi threads
        with ThreadPool(max_workers) as executor:
            for i, task in enumerate(executor.map(load_unique_task, tasks)):
                tile = load_unique_task(task)
                if tile is not None:
                    tiles.append(tile)

        # build dataset
        dataset = cls(tiles=tiles)
        dataset.build()

        return dataset

    # TODO: Debug
    def to_fiftyone(self, dataset_name: str, persistent: bool = False) -> fo.Dataset:
        try:
            # Try to load existing dataset
            dataset = fo.load_dataset(dataset_name)
            logger.info(f"Loaded existing dataset: {dataset_name}")
        except ValueError:
            # Create new dataset if it doesn't exist
            dataset = fo.Dataset(dataset_name, persistent=persistent)
            logger.info(f"Created new dataset: {dataset_name}")

        samples = []

        for img_path, df_detections in tqdm(
            self.data.groupby("file_name"), desc="Creating-fifytone-dataset"
        ):
            if not os.path.exists(img_path):
                logger.warning(f"Warning: Image not found at {img_path}.Skipping")
                continue

            sample = fo.Sample(filepath=img_path)

            if df_detections.dropna().empty:
                logger.info(f"Image at {img_path} is a negative sample")
                sample["is_positive"] = False
                samples.append(sample)
                continue

            # conf = bboxes['scores']
            class_name = df_detections["class"].tolist()

            # get bbox [x, y, w, h] in normalized coords [0,1]
            bboxes = df_detections[["x_min", "y_min"]].copy()
            bboxes.loc[:, "x_min"] /= df_detections.loc[:, "width"]
            bboxes.loc[:, "y_min"] /= df_detections.loc[:, "height"]
            bboxes["w"] = 0.0
            bboxes["h"] = 0.0
            bboxes.loc[:, "w"] = (
                df_detections.loc[:, "x_max"] - df_detections.loc[:, "x_min"]
            ) / df_detections.loc[:, "width"]
            bboxes.loc[:, "h"] = (
                df_detections.loc[:, "y_max"] - df_detections.loc[:, "y_min"]
            ) / df_detections.loc[:, "height"]
            bboxes = bboxes[["x_min", "y_min", "w", "h"]].to_numpy().tolist()

            # Convert detections to FiftyOne format
            fo_detections = []
            for i, box in enumerate(bboxes):
                fo_detection = fo.Detection(
                    label=class_name[i],
                    bounding_box=box,
                    # confidence=det.get('confidence', None)
                )
                fo_detections.append(fo_detection)

            sample["gt"] = fo.Detections(detections=fo_detections)
            sample["is_positive"] = True
            samples.append(sample)

        dataset.add_samples(samples)

        return dataset


class LabelHandler:
    def __init__(self, config: LabelConfig):
        self.config = config

        self._label_map = None

        if (self.config.discard is not None) and (self.config.keep is not None):
            intersec = np.intersect1d(self.config.keep, self.config.discard)
            assert len(intersec) == 0, (
                f"{intersec} are required to be discarded and kept. Error..."
            )

    def load_map(
        self,
    ) -> Dict:
        """Load and filter label mapping"""

        # load label mapping
        with open(self.config.label_map, "r") as file:
            label_map = json.load(file)

        names = [p["name"] for p in label_map]

        if self.config.discard is not None:
            names = [p for p in names if p not in self.config.discard]

        if self.config.keep is not None:
            names = [p for p in label_map if p in self.config.keep]

        self._label_map = dict(zip(range(len(names)), names, strict=False))

        return label_map

    def update_config(self, yaml_path: Path) -> None:
        """Updates yolo data config yaml file "names" and "nc" fields."""

        # load yaml
        yolo_config = load_yaml(yaml_path)

        # updaate yaml and save
        yolo_config.update({"names": self._label_map, "nc": len(self._label_map)})


class YOLODatasetBuilder:
    def __init__(
        self,
        data_config: DataConfig,
    ):
        self.config = data_config

        self._validate_config()

    def _validate_config(self):
        """Ensure configuration parameters are valid"""
        if self.config.slice_width <= 0 or self.config.slice_height <= 0:
            raise ValueError("Slice dimensions must be positive")

        #  Checking inconsistency in arguments
        if (self.config.clear_output + self.config.load_coco_annotations) == 2:
            raise ValueError(
                "Warning : both clear_yolo_dir and load_coco_annotations are enabled! "
                "it is likely to not work as expected."
            )

    def _run_single_coco_dir(
        self,
        image_dir: str,
        coco_path: str,
        name_id_map: dict,
        labels_to_discard: list,
        labels_to_keep: list,
    ):
        # slice annotations
        coco_dict_slices = ImageProcessor.get_slices(
            coco_path=coco_path, img_dir=image_dir, config=self.config
        )
        # sample tiles
        df_tiles = ImageProcessor.sample_slices(
            coco_dict_slices=coco_dict_slices,
            empty_ratio=self.config.empty_ratio,
            out_csv_path=None,
            img_dir=image_dir,
            save_all=self.config.save_all,
            labels_to_discard=labels_to_discard,
            labels_to_keep=labels_to_keep,
            sample_only_empty=self.config.save_only_empty,
        )

        # detector_training mode
        if self.config.is_single_cls:
            df_tiles["label_id"] = 0
        else:
            df_tiles["label_id"] = df_tiles["labels"].map(name_id_map)
            mask = ~df_tiles["label_id"].isna()
            df_tiles.loc[mask, "label_id"] = df_tiles.loc[mask, "label_id"].apply(int)

        # save labels in yolo format
        self.save_annotations(
            df_annotation=df_tiles.dropna(axis=0, how="any"),
            output_dir=self.config.dest_path_labels,
        )

        # save tiles
        ImageProcessor.save_tiles(
            df_tiles=df_tiles,
            output_dir=self.config.dest_path_images,
            clear=self.config.clear_output,
        )

        return None

    def build(
        self,
        map_imgdir_cocopath: dict,
        label_handler: LabelHandler,
        max_workers: int = 1,
    ) -> None:
        """Main pipeline entry point"""

        # load label map and update yolo data_cfg_yaml file
        name_id_map = {}
        if not self.config.is_single_cls:
            label_map = label_handler.load_map()
            label_handler.update_config(self.config.yolo_data_config_yaml)
            name_id_map = {val: key for key, val in label_map.items()}

        # TODO: add error handling
        # Parallel mode
        if max_workers > 1 and len(map_imgdir_cocopath) > 1:
            chunksize = len(map_imgdir_cocopath) // max_workers
            func = partial(
                self._run_single_coco_dir,
                name_id_map=name_id_map,
                labels_to_discard=label_handler.config.discard,
                labels_to_keep=label_handler.config.keep,
            )

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for _ in tqdm(
                    executor.map(
                        func, map_imgdir_cocopath.items(), chunksize=chunksize
                    ),
                    desc="Building yolo dataset",
                ):
                    continue

        # Sequential mode
        else:
            for image_dir, cocopath in tqdm(
                map_imgdir_cocopath.items(), desc="Building yolo dataset"
            ):
                try:
                    self._run_single_coco_dir(
                        image_dir=image_dir,
                        coco_path=cocopath,
                        name_id_map=name_id_map,
                        labels_to_discard=label_handler.config.discard,
                        labels_to_keep=label_handler.config.keep,
                    )

                except Exception:
                    print("--" * 25, end="\n")
                    traceback.print_exc()
                    print("--" * 25)
                    print(
                        f"Failed to build yolo dataset for for {image_dir} -- {cocopath}\n\n"
                    )

        return None

    def save_annotations(self, df_annotation: pd.DataFrame, output_dir: Path) -> None:
        """Save annotations in YOLO format"""

        cols = ["label_id", "x", "y", "width", "height"]
        for col in cols:
            assert df_annotation[col].isna().sum() < 1, (
                "there are NaN values. Check out."
            )

        # change type
        for col in cols[1:]:
            df_annotation.loc[:, col] = df_annotation[col].apply(float)
        df_annotation.loc[:, "label_id"] = df_annotation["label_id"].apply(int)
        df_annotation = df_annotation.astype({"label_id": "int32"})

        # normalize values
        df_annotation.loc[:, "x"] = df_annotation["x"].apply(
            lambda x: x / self.config.slice_width
        )
        df_annotation.loc[:, "y"] = df_annotation["y"].apply(
            lambda y: y / self.config.slice_height
        )
        df_annotation.loc[:, "width"] = df_annotation["width"].apply(
            lambda x: x / self.config.slice_width
        )
        df_annotation.loc[:, "height"] = df_annotation["height"].apply(
            lambda y: y / self.config.slice_height
        )

        # check value range
        assert df_annotation[cols[1:]].all().max() <= 1.0, "max value <= 1"
        assert df_annotation[cols[1:]].all().min() >= 0.0, "min value >=0"

        for image_name, df in tqdm(
            df_annotation.groupby("images"), desc="Saving yolo labels"
        ):
            txt_file = image_name.split(".")[0] + ".txt"
            df[cols].drop_duplicates().to_csv(
                os.path.join(output_dir, txt_file), sep=" ", index=False, header=False
            )


class ClassificationDatasetBuilder:
    def __init__(
        self,
        eval_config: EvaluationConfig,
    ):
        from .evaluation import PerformanceEvaluator

        self.config = eval_config
        # self.detector: InferenceEngine = None
        self.source_dirs = None
        self.output_dir = None
        self.perf_eval = PerformanceEvaluator(config=self.config)
        self.bbox_resize_factor = None
        self.feature_extractor = None

        self.tn_label = "true_negatives"
        self.tp_label = "true_positives"

    def set_dirs(self, source_dirs: Sequence[str], output_dir: str):
        assert isinstance(source_dirs, Sequence), (
            "Please provide a Sequence de directory, e.g. List or Tuple"
        )
        for d in source_dirs:
            assert os.path.exists(d) and Path(d).is_dir(), f"Directory {d} not found."
        self.source_dirs = source_dirs
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    def run(
        self,
        strategies: list[str] = ["gt", "hn"],
        detector: InferenceEngine = None,
        feature_extractor: FeatureExtractor = None,
        bbox_resize_factor: int = 1,
        save_true_negatives: bool = False,
        tn_kwargs=dict(w=50, h=50, number=3),
        tp_kwargs=dict(w=50, h=50),
        fp_kwargs=dict(w=50, h=50),
        hn_kwargs=dict(w=50, h=50),
        max_workers: int = 1,
    ):
        assert len(set(strategies) & set(["gt", "fp", "hn"])) > 0, (
            f"None of gt, fp, or hn was provided strategy. Received {strategies}"
        )
        assert isinstance(detector, InferenceEngine), (
            f"Expected detector to be InferenceEngine. Received {type(detector)}"
        )

        self.bbox_resize_factor = bbox_resize_factor
        self.feature_extractor = feature_extractor

        # load predictions
        dataset = None
        if "fp" in strategies or "hn" in strategies:
            tiles = [Tile(image_path=p) for p in get_images_from_dirs(self.source_dirs)]
            dataset = LabelingDataset(tiles=tiles)
            dataset.add_predictions(engine=detector, build=True)

        for strategy in list(set(strategies)):
            if strategy == "gt":
                df_labels, _ = self.load_groundtruth(
                    images_dirs=self.source_dirs,
                    images_paths=None,
                    load_empty=save_true_negatives,
                    max_workers=max_workers,
                )
                self.save_groundtruth(
                    df_labels=df_labels,
                    save_true_negatives=save_true_negatives,
                    tn_kwargs=tn_kwargs,
                    tp_kwargs=tp_kwargs,
                )

            elif strategy == "fp":
                self._save_fp(
                    dataset=dataset, bbox_resize_factor=bbox_resize_factor, **fp_kwargs
                )

            elif strategy == "hn":
                self._save_hn(
                    dataset=dataset, bbox_resize_factor=bbox_resize_factor, **hn_kwargs
                )

            else:
                raise NotImplementedError(f"strategy:{strategy} is not defined.")

    def _save_one_image(
        self,
        image: np.ndarray,
        label_name: str | int,
        file_name: str,
        tag: str,
        ext: str = ".jpg",
    ):
        # skipping images 80% black
        frac = (image == 0.0).sum() / image.sum()
        if frac > 0.8 and self.feature_extractor:
            logger.info(f"Skipping {os.path.basename(file_name)}. It's all black.")
            return None

        img_dir = Path(self.output_dir) / str(label_name)
        img_dir.mkdir(exist_ok=True, parents=False)
        save_path = img_dir / f"{Path(file_name).stem}#{tag}"
        save_path = save_path.with_suffix(ext)

        if ext != ".npy":
            cv2.imwrite(save_path, image)
        else:
            np.save(save_path, image)

        return None

    def _save_batch(self, batch: list[dict]):
        if len(batch) < 1:
            return None

        if self.feature_extractor:
            images = [data["image"] for data in batch]
            batch_features = self.feature_extractor.run(images)

        for i, data in enumerate(batch):
            if self.feature_extractor:
                data["image"] = batch_features[i]
                self._save_one_image(ext=".npy", **data)
            else:
                self._save_one_image(ext=".jpg", **data)

    def _save_tn(
        self,
        file_name: str,
        bbox_resize_factor: int = 1,
        w: int = 50,
        h: int = 50,
        number: int = 2,
    ):
        image = Image.open(file_name).convert("RGB")
        image = np.asarray(image)

        img_height, img_width = image.shape[:2]

        label_name = self.tn_label

        xs = np.random.randint(
            low=w * bbox_resize_factor,
            high=img_width - w * bbox_resize_factor,
            size=number,
        )
        ys = np.random.randint(
            low=h * bbox_resize_factor,
            high=img_height - h * bbox_resize_factor,
            size=number,
        )
        batch = []
        pairs = list(product(xs, ys))
        shuffle(pairs)

        for x, y in pairs[:number]:
            # rescale w,h
            w_ = (np.random.rand() + 0.5) * w
            h_ = (np.random.rand() + 0.5) * h

            x1 = x - w_ / 2
            x2 = x + w_ / 2
            y1 = y - h_ / 2
            y2 = y + h_ / 2

            x1, x2, y1, y2 = resize_bbox(
                bbox_resize_factor, x1, x2, y1, y2, img_width, img_height
            )

            # record
            data = dict(
                image=image[y1:y2, x1:x2],
                label_name=label_name,
                file_name=file_name,
                tag=f"#{y1}_{y2}_{x1}_{x2}",
            )
            batch.append(data)

        # save data
        self._save_batch(batch)

        return None

    def _save_tp(
        self,
        df_gt: pd.DataFrame,
        file_name: str,
        bbox_resize_factor: int = 1,
        w: int = None,
        h: int = None,
    ):
        image = Image.open(file_name).convert("RGB")
        image = np.asarray(image)

        batch = []
        count = 0
        for i, row in df_gt.iterrows():
            x1 = int(row["x_min"])
            y1 = int(row["y_min"])
            x2 = int(row["x_max"])
            y2 = int(row["y_max"])
            img_width = row["width"]
            img_height = row["height"]

            if w and h:
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)
                x1 = x - w / 2
                x2 = x + w / 2
                y1 = y - h / 2
                y2 = y + h / 2

            x1, x2, y1, y2 = resize_bbox(
                bbox_resize_factor, x1, x2, y1, y2, img_width, img_height
            )

            # record
            data = dict(
                image=image[y1:y2, x1:x2],
                label_name=self.tp_label,
                file_name=file_name,
                tag=f"#{y1}_{y2}_{x1}_{x2}",
            )
            batch.append(data)
            count += 1

        # save data
        self._save_batch(batch)

        return None

    def _save_hn(
        self,
        dataset: LabelingDataset,
        bbox_resize_factor: int = 1,
        w: int = None,
        h: int = None,
    ):
        logger.info("Computing Hard negatives...")

        is_tp = (
            lambda p: Path(str(p).replace("images", "labels"))
            .with_suffix(".txt")
            .exists()
        )

        count = 0
        for tile in tqdm(dataset.tiles, desc="Saving Hard negatives"):
            if is_tp(tile.image_path):
                continue  # skip true positives

            image = Image.open(tile.image_path).convert("RGB")
            img_width, img_height = image.size
            image = np.asarray(image)

            batch = []

            for det in tile.predictions:
                if det.is_empty:
                    continue

                if w and h:
                    x = det.x
                    y = det.y
                    x1 = x - w / 2
                    x2 = x + w / 2
                    y1 = y - h / 2
                    y2 = y + h / 2
                else:
                    x1 = det.x_min
                    y1 = det.y_min
                    x2 = det.x_max
                    y2 = det.y_max

                x1, x2, y1, y2 = resize_bbox(
                    bbox_resize_factor, x1, x2, y1, y2, img_width, img_height
                )

                # record
                data = dict(
                    image=image[y1:y2, x1:x2],
                    label_name=self.tn_label,
                    file_name=tile.image_path,
                    tag=f"#{y1}_{y2}_{x1}_{x2}",
                )
                batch.append(data)
                count += 1

            # save data
            self._save_batch(batch)

        logger.info(f"{count} Hard negatives have been saved.")
        return None

    def save_groundtruth(
        self,
        df_labels: pd.DataFrame,
        save_true_positives: bool = True,
        save_true_negatives: bool = False,
        tn_kwargs: dict = {},
        tp_kwargs: dict = {},
    ):
        cols = ["x_min", "y_min", "x_max", "y_max", "width", "height"]

        # save labels
        for file_name, df_gt in tqdm(
            df_labels.groupby("file_name"), desc="Saving groundtruth"
        ):
            # save positive samples
            if save_true_positives:
                self._save_tp(
                    df_gt.loc[:, cols].dropna(axis=0, how="any"),
                    file_name,
                    self.bbox_resize_factor,
                    **tp_kwargs,
                )

            # save empty samples
            if save_true_negatives and df_gt.loc[:, cols].isna().sum().sum() > 0:
                self._save_tn(
                    file_name,
                    self.bbox_resize_factor,
                    **tn_kwargs,
                )
        return None

    def load_groundtruth(
        self,
        images_dirs: list[str] = None,
        images_paths: list[str] = None,
        load_empty: bool = False,
        max_workers: int = 1,
    ) -> tuple[pd.DataFrame, str]:
        paths = images_paths
        if paths is None:
            assert images_dirs is not None, (
                "Both images_dirs and images_paths are None. Provide exactly one."
            )
            iters = [Path(p).glob("*") for p in images_dirs]
            paths = chain.from_iterable(iters)

        labels, _format = DataHandler.load_yolo_groundtruth(
            images_dir=None,
            images_paths=paths,
            load_empty=load_empty,
            max_workers=max_workers,
        )

        return labels, _format

    def _save_fp(
        self,
        dataset: LabelingDataset,
        bbox_resize_factor: int = 1,
        w: int = None,
        h: int = None,
    ):
        """Run batch detection and save cropped ROIs"""

        logger.info("Computing False Positives...")

        # raise NotImplementedError("Debugging to do...")

        df_metrics = self.perf_eval.run(
            dataset=dataset,
            pred_results_dir=self.output_dir,
            save_tag="cls",
            load_results=self.config.load_results,
        )

        if df_metrics.empty:
            return None

        mask_fp = df_metrics["pred_FP"] == True
        for file_name, df_det in tqdm(
            df_metrics.loc[mask_fp, :].groupby("file_name"), desc="Saving FPs and FNs"
        ):
            image = Image.open(file_name).convert("RGB")
            img_width, img_height = image.size
            image = np.asarray(image)

            batch = []
            count = 0
            for i, row in df_det.iterrows():
                try:
                    x1 = int(row["pred_x_min"])
                    y1 = int(row["pred_y_min"])
                    x2 = int(row["pred_x_max"])
                    y2 = int(row["pred_y_max"])
                    label_name = "false_positives"
                except:
                    continue

                if w and h:
                    x = int((x1 + x2) / 2)
                    y = int((y1 + y2) / 2)
                    x1 = x - w / 2
                    x2 = x + w / 2
                    y1 = y - h / 2
                    y2 = y + h / 2

                x1, x2, y1, y2 = resize_bbox(
                    bbox_resize_factor, x1, x2, y1, y2, img_width, img_height
                )

                # record
                data = dict(
                    image=image[y1:y2, x1:x2],
                    label_name=label_name,
                    file_name=file_name,
                    tag=f"#{y1}_{y2}_{x1}_{x2}",
                )
                batch.append(data)
                count += 1

            # save data
            self._save_batch(batch)

        return None


# TODO: debug
class DataPreparation:
    def __init__(self, dataset_config: DataConfig, label_config: LabelConfig):
        self.dataset_config = dataset_config
        self.label_config = label_config

        self._initialize_components()

    def _initialize_components(self):
        self.label_handler = LabelHandler(self.label_config)
        self.converter = LabelstudioConverter(self.dataset_config)
        self.dataset_builder = YOLODatasetBuilder(self.dataset_config)

    def _clean_workspace(
        self,
    ):
        # clear directories
        if self.dataset_config.clear_output:
            for p in [
                self.dataset_config.dest_path_images,
                self.dataset_config.dest_path_labels,
                self.dataset_config.coco_json_dir,
            ]:
                shutil.rmtree(p)
                Path(p).mkdir(parents=True, exist_ok=True)
                logger.info(f"Deleting all content in: {p}")

    def run(self, ls_xml_config=None, ls_client=None, image_dir: str = None) -> None:
        """Execute full preparation pipeline"""

        # 1. Clean workspace
        self._clean_workspace()

        # 2. Convert source label studio format  to COCO
        if self.dataset_config.load_coco_annotations:
            map_imgdir_cocopath = load_coco_annotations(
                dest_dir_coco=self.dataset_config.coco_json_dir, image_dir=image_dir
            )
        else:
            map_imgdir_cocopath = self.converter.to_coco(
                input_dir=self.dataset_config.ls_json_dir,
                dest_dir_coco=self.dataset_config.coco_json_dir,
                parse_ls_config=self.dataset_config.parse_ls_config,
                ls_client=ls_client,
                ls_xml_config=ls_xml_config,
            )

        # 3. Load label map from ``label_handler.config.label_map``
        self.label_handler.load_map()

        # 3. Generate dataset
        self.dataset_builder.build(map_imgdir_cocopath, self.label_handler)
