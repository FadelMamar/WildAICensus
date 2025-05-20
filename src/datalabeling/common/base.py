from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import math
import geopy
from PIL import Image
import pandas as pd

from .annotation_utils import compute_detection_gps


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

        return det_objects

    def to_absolute_coords(self, x_offset: int, y_offset: int) -> None:
        """Convert relative coordinates to absolute image coordinates."""
        self.x_min += x_offset
        self.x_max += x_offset
        self.y_min += y_offset
        self.y_max += y_offset

    @property
    def is_empty(self):
        return any([self.x is None, self.y is None, self.w is None, self.h is None])

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
        return math.floor((self.x_min + self.x_max) / 2)

    @property
    def y(
        self,
    ):
        return math.floor((self.y_min + self.y_max) / 2)

    @property
    def w(
        self,
    ):
        return int(self.x_max - self.x_min)

    @property
    def h(
        self,
    ):
        return int(self.y_max - self.y_min)

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
    tile_gps_loc: str = None
    detections: List[Detection] = None

    def offset_detections(
        self,
    ):
        if self.x_offset is not None and self.y_offset is not None:
            for det in self.detections:
                det.to_absolute_coords(self.x_offset, self.y_offset)

    def update_detection_gps(
        self,
        sensor_height: float = 24.0,
        flight_height: float = 180.0,
        gsd=None,
    ):
        image = self.image_data
        if image is None:
            image = Image.open(self.image_path)

        for det in self.detections:
            det.image_gps_loc = self.tile_gps_loc

            if det.image_gps_loc is not None:
                det.gps_loc = compute_detection_gps(
                    x_center=det.x,
                    y_center=det.y,
                    image=image,
                    image_gps_loc=det.image_gps_loc,
                    flight_height=flight_height,
                    sensor_height=sensor_height,
                    gsd=gsd,
                )

    def detections_to_df(
        self,
    ) -> pd.DataFrame:
        self._set_with_height()

        for det in self.detections:
            assert self.image_path is not None, "provide the path to the tile."
            det.parent_image = self.image_path

        out = [det.to_dict() for det in self.detections]
        df = pd.DataFrame.from_dict(out, orient="columns")

        df["image_width"] = self.width
        df["image_height"] = self.height

        # YOLO format
        if len(self.detections) > 0:
            df["w"] = df["w"] / self.width
            df["h"] = df["h"] / self.height
            df["x"] = df["x"] / self.width
            df["y"] = df["y"] / self.height
        else:
            df["parent_image"] = self.image_path

        df.rename(columns={"parent_image": "file_name"}, inplace=True)

        return df

    def _set_with_height(
        self,
    ):
        image = self.image_data
        if image is None:
            image = Image.open(self.image_path)
        if self.width is None or self.height is None:
            self.width, self.height = image.size
