from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import math
import geopy
import torch
from torchvision.transforms import PILToTensor
from PIL import Image
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

from .annotation_utils import compute_detection_gps, GPSUtils


@dataclass
class Detection:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    label: int
    class_name: str
    score: float = None
    gps_loc: str = None
    image_gps_loc: str = None
    parent_image: str = None

    @classmethod
    def empty(cls, parent_image: str = None):
        return cls(
            x_min=np.nan,
            x_max=np.nan,
            y_min=np.nan,
            y_max=np.nan,
            label=np.nan,
            class_name=None,
            parent_image=parent_image,
        )

    @classmethod
    def from_coco(
        cls,
        coco: dict,
        parent_image: str,
        image_gps_loc: str = None,
        gps_loc: str = None,
    ):
        bbox = coco["bbox"]
        label = coco["category_id"]
        class_ = coco["category_name"]
        score = coco.get("score", None)

        det = cls(
            x_min=int(bbox[0]),
            y_min=int(bbox[1]),
            x_max=int(bbox[0] + bbox[2]),
            y_max=int(bbox[1] + bbox[3]),
            class_name=class_,
            label=label,
            score=score,
            image_gps_loc=image_gps_loc,
            gps_loc=gps_loc,
            parent_image=parent_image,
        )

        return det

    @classmethod
    def from_ls(cls, detections: list, image_path: str):
        det_objects = []
        for detection in detections:
            for det in detection["result"]:
                image_height = det["original_height"]
                image_width = det["original_width"]
                value = det["value"]
                class_name = value["rectanglelabels"]  # size 1
                x_min = value["x"] * image_width / 100
                y_min = value["y"] * image_height / 100
                w = value["width"] * image_width / 100
                h = value["height"] * image_height / 100

                assert len(class_name) == 1, "Error. Check out code or Labeling format."
                class_name = class_name[0]

                det = cls(
                    x_min=int(x_min),
                    y_min=int(y_min),
                    x_max=int(x_min + w),
                    y_max=int(y_min + h),
                    class_name=class_name,
                    label=None,
                    score=None,
                    image_gps_loc=None,
                    gps_loc=None,
                    parent_image=image_path,
                )

                det_objects.append(det)

        # if empty, add empty detection
        if len(det_objects) == 0:
            det_objects.append(Detection.empty(parent_image=image_path))

        return det_objects

    def to_absolute_coords(self, x_offset: int, y_offset: int) -> None:
        """Convert relative coordinates to absolute image coordinates."""
        self.x_min += x_offset
        self.x_max += x_offset
        self.y_min += y_offset
        self.y_max += y_offset

    @property
    def is_empty(self):
        vals = [self.x, self.y, self.w, self.h]
        return any([(np.isnan(v) or v is None) for v in vals])

    def to_dict(
        self,
    ):
        out = vars(self)

        out["w"] = self.w
        out["h"] = self.h
        out["x"] = self.x
        out["y"] = self.y
        out["area"] = self.area

        return out

    def to_ls(
        self, from_name, to_name, label_type, img_height: int, img_width: int
    ) -> dict:
        # formatting the prediction to work with Label studio
        score = self.score
        if not isinstance(score, float):
            score = 0.0
        template = {
            "from_name": from_name,
            "to_name": to_name,
            "type": label_type,
            "original_width": img_width,
            "original_height": img_height,
            "image_rotation": 0,
            "value": {
                label_type: [
                    self.class_name,
                ],
                "x": self.x_min / img_width * 100,
                "y": self.y_min / img_height * 100,
                "width": self.w / img_width * 100,
                "height": self.h / img_height * 100,
                "rotation": 0,
            },
            "score": score,
        }
        return template

    def get_base_image(self) -> Image.Image:
        assert self.parent_image is not None, "Parent image is not defined"
        return Image.open(self.parent_image)

    @property
    def area(
        self,
    ):
        return self.w * self.h

    @property
    def x(
        self,
    ):
        x = (self.x_min + self.x_max) / 2
        if not np.isnan(x):
            return math.floor(x)
        return x

    @property
    def y(
        self,
    ):
        y = (self.y_min + self.y_max) / 2
        if not np.isnan(y):
            return math.floor(y)
        return y

    @property
    def w(
        self,
    ):
        w = self.x_max - self.x_min
        if not np.isnan(w):
            return int(w)
        return w

    @property
    def h(
        self,
    ):
        h = self.y_max - self.y_min
        if not np.isnan(h):
            return int(h)
        return h

    @property
    def gps_as_decimals(
        self,
    ):
        assert isinstance(self.gps_loc, str)

        point = geopy.Point.from_string(self.gps_loc)

        lat = point.latitude
        long = point.longitude
        alt = point.altitude * 1e3  # converting to meters

        return lat, long, alt


@dataclass
class Tile:
    """Class representing an image tile."""

    image_path: str
    image_data: Image.Image = None
    width: int = None
    height: int = None
    x_offset: int = None
    y_offset: int = None
    parent_image: str = None
    date: str = None
    parent_image_date: str = None
    tile_gps_loc: str = None
    predictions: List[Detection] = None
    annotations: List[Detection] = None
    _pred_is_original: bool = False
    _annot_is_original: bool = False

    def __post_init__(self):
        if self.parent_image:
            try:
                self.parent_image_date = Image.open(self.parent_image)._getexif()[36867]
            except:
                pass

        if self.image_path:
            try:
                self.date = Image.open(self.image_path)._getexif()[36867]
            except:
                pass

        if self.image_data is None:
            self.width, self.height = Image.open(self.image_path).size

        else:
            self.width, self.height = self.image_data.size

        self._extract_gps_coords()

        return None

    def load_image_data(self) -> Image.Image:
        if self.image_data is not None:
            return self.image_data
        else:
            return Image.open(self.image_path)

    def _extract_gps_coords(
        self,
    ) -> None:
        # assert self.image_path is not None, "Provide image_path field when defining a tile"
        image = None
        if self.image_path is None:
            image = self.image_data

        coords = GPSUtils.get_gps_coord(
            file_name=self.image_path,
            image=image,
            altitude=None,
            return_as_decimal=False,
        )
        if coords is not None:
            gps, _ = coords
            self.tile_gps_loc = gps

        logger.debug("gps extraction of tile failed")

        return None

    def offset_detections(
        self,
    ):
        if self.x_offset is not None and self.y_offset is not None:
            if self._pred_is_original:
                logger.info(
                    "Skipping - Predictions have already been mapped to the reference coordinates."
                )
            if self.predictions and (not self._pred_is_original):
                for det in self.predictions:
                    det.to_absolute_coords(self.x_offset, self.y_offset)
                self._pred_is_original = True

            if self.annotations and (not self._annot_is_original):
                if self._annot_is_original:
                    logger.info(
                        "Skipping - Annotations have already been mapped to the reference coordinates."
                    )
                for det in self.annotations:
                    det.to_absolute_coords(self.x_offset, self.y_offset)
                self._annot_is_original = True
        else:
            logger.info("Failed...self.x_offset is None or self.y_offset is not None.")

    def update_detection_gps(
        self,
        sensor_height: float,
        flight_height: float,
        gsd: float,
    ):
        assert isinstance(self.tile_gps_loc, str), (
            f"Expected self.tile_gps_loc to be 'str'. Found '{type(self.tile_gps_loc)}' "
        )

        image = self.image_data
        if image is None:
            image = Image.open(self.image_path)

        array = []
        if self.annotations:
            array = array + self.annotations

        if self.predictions:
            array = array + self.predictions

        for det in array:
            if det.is_empty:
                continue

            det.image_gps_loc = self.tile_gps_loc

            if det.image_gps_loc is not None:
                try:
                    det.gps_loc = compute_detection_gps(
                        x_center=det.x,
                        y_center=det.y,
                        image=image,
                        image_gps_loc=det.image_gps_loc,
                        flight_height=flight_height,
                        sensor_height=sensor_height,
                        gsd=gsd,
                    )
                except Exception as e:
                    # print(e)
                    logger.error(f"Failed to compute GPS location of detections. {e}")
                    det.gps_loc = None

    def detections_to_df(
        self,
    ) -> pd.DataFrame:
        assert self.image_path is not None, "provide the path to the tile."

        out = []
        # add_tag = lambda x,tag:{f"{tag}_{k}":v for k,v in x.items()}

        def add_tag(out: dict, is_annot: bool):
            out["is_annot"] = is_annot
            return out

        if self.annotations:
            out = out + [
                add_tag(det.to_dict(), is_annot=True) for det in self.annotations
            ]

        if self.predictions:
            out = out + [
                add_tag(det.to_dict(), is_annot=False) for det in self.predictions
            ]

        df = pd.DataFrame.from_dict(out, orient="columns")

        if df.empty:
            df.at[0, "image_width"] = self.width

        df["image_width"] = self.width
        df["image_height"] = self.height
        df["parent_image"] = self.image_path
        df["original_date"] = self.parent_image_date

        # YOLO format
        if len(out) > 0:
            df["w"] = df["w"] / self.width
            df["h"] = df["h"] / self.height
            df["x"] = df["x"] / self.width
            df["y"] = df["y"] / self.height

            # check detections
            self.check_detections(df)

        df.rename(columns={"parent_image": "file_name"}, inplace=True)

        return df

    def set_predictions(self, data: List[Detection]) -> None:
        assert isinstance(data, list), f"Expected 'list' but received {type(list)}"

        if len(data) > 0:
            self.predictions = data

        else:
            self.predictions = [
                Detection.empty(parent_image=self.image_path),
            ]

        return None

    def set_annotations(self, data: List[Detection]) -> None:
        assert isinstance(data, list), f"Expected 'list' but received {type(list)}"

        if len(data) > 0:
            self.annotations = data

        else:
            self.annotations = [
                Detection.empty(parent_image=self.image_path),
            ]

        return None

    def check_detections(self, df: pd.DataFrame) -> None:
        df = df[["x_min", "x_max", "y_min", "y_max"]].dropna().copy()

        if df.empty:
            return None

        assert df.to_numpy().min() >= 0
        assert df[["x_min", "x_max"]].to_numpy().max() <= self.width
        assert df[["y_min", "y_max"]].to_numpy().max() <= self.height

        return None

    def as_batch(self, tile_size: int, stride: int) -> tuple[torch.Tensor, dict]:
        if self.image_data is not None:
            image = self.image_data
        else:
            assert self.image_path is not None, (
                "'image_path' should be set if 'image_data' is None!"
            )
            image = Image.open(self.image_path).convert("RGB")

        def get_tiles(image: torch.Tensor):
            if image.dim() == 2:
                image = image.unsqueeze(0)  # Add channel dimension
                squeeze_output = True
            else:
                squeeze_output = False

            C, H, W = image.shape

            # Calculate number of tiles in each dimension
            num_tiles_h = (H - tile_size) // stride + 1
            num_tiles_w = (W - tile_size) // stride + 1

            # Use unfold to create tiles
            # First unfold along height dimension
            unfolded_h = image.unfold(
                1, tile_size, stride
            )  # Shape: (C, num_tiles_h, W, tile_size)

            # Then unfold along width dimension
            tiles = unfolded_h.unfold(
                2, tile_size, stride
            )  # Shape: (C, num_tiles_h, num_tiles_w, tile_size, tile_size)

            # Reshape to get individual tiles
            tiles = tiles.contiguous().view(
                C, num_tiles_h * num_tiles_w, tile_size, tile_size
            )
            tiles = tiles.permute(
                1, 0, 2, 3
            )  # Shape: (num_tiles, C, tile_size, tile_size)

            if squeeze_output:
                tiles = tiles.squeeze(1)

            return tiles, num_tiles_h, num_tiles_w

        image = PILToTensor()(image)

        tiles, num_tiles_h, num_tiles_w = get_tiles(image)

        C, H, W = image.shape
        x_indices = torch.arange(W).reshape(1, -1).expand(H, W)
        y_indices = torch.arange(H).reshape(-1, 1).expand(H, W)
        x_indices, _, _ = get_tiles(x_indices)
        y_indices, _, _ = get_tiles(y_indices)
        x_min = y_indices.min(1, True)[0].min(2)[0].squeeze().cpu().numpy()
        y_min = y_indices.min(1, True)[0].min(2)[0].squeeze().cpu().numpy()

        offset_info = {
            "y_offset": y_min.tolist(),
            "x_offset": x_min.tolist(),
            "y_end": (y_min + tile_size).tolist(),
            "x_end": (x_min + tile_size).tolist(),
            "file_name": [str(self.image_path)] * len(x_min),
        }

        return tiles, offset_info
