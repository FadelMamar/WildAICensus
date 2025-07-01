from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import math
import geopy
from jinja2.nodes import If
import torch
from torchvision.transforms import PILToTensor
from torchvision.ops import nms
from torchmetrics.functional.detection import complete_intersection_over_union
from PIL import Image
import pandas as pd
import numpy as np
import logging
import uuid

logger = logging.getLogger(__name__)

from .annotation_utils import compute_detection_gps, GPSUtils, ImageProcessor
from .config import SENSOR_HEIGHTS, FlightSpecs


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.
    Args:
        bbox1 (List[float]): Bounding box in [x_min, y_min, x_max, y_max] format.
        bbox2 (List[float]): Bounding box in [x_min, y_min, x_max, y_max] format.
    Returns:
        float: IoU value between the two bounding boxes.
    """

    bbox1 = torch.tensor([bbox1])
    bbox2 = torch.tensor([bbox2])

    iou = complete_intersection_over_union(
        preds=bbox1, target=bbox2, aggregate=False
    ).item()

    return iou


@dataclass
class GeographicBounds:
    """Geographic bounding box for image footprint in UTM coordinates"""

    north: float  # Max latitude
    south: float  # Min latitude
    east: float  # Max longitude
    west: float  # Min longitude

    @property
    def area(self) -> float:
        """Calculate area in square degrees covered by the bounding box."""
        return (self.east - self.west) * (self.north - self.south)

    def box(self, box_format="xyxy") -> List[float]:
        """Return the bounding box as a list in the specified format.
        Args:
            box_format (str): Format of the box. Only 'xyxy' is supported.
        Returns:
            List[float]: Bounding box coordinates.
        """
        if box_format == "xyxy":
            return [self.west, self.south, self.east, self.north]
        else:
            raise ValueError("only 'xyxy' supproted.")

    def overlap_ratio(self, other: "GeographicBounds") -> float:
        """Calculate overlap ratio (IoU) with another bounds using torchmetrics IntersectionOverUnion.
        Args:
            other (GeographicBounds): Another geographic bounds object.
        Returns:
            float: Overlap ratio (IoU) between the two bounds.
        """

        # Prepare boxes in [x_min, y_min, x_max, y_max] format
        box_self = self.box(box_format="xyxy")
        box_other = other.box(box_format="xyxy")

        return compute_iou(box_self, box_other)


@dataclass
class Detection:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    label: int
    class_name: str
    id: Optional[str] = None
    score: Optional[float] = np.nan
    gps_loc: Optional[str] = None
    image_gps_loc: str = None
    parent_image: Optional[str] = None
    image_id: Optional[str] = None
    timestamp: str = None
    distance_to_centroid: float = None

    geographic_footprint: Optional[GeographicBounds] = None

    def __post_init__(self):
        """Post-initialization to set unique ID and compute distance to centroid."""
        if self.id is None:
            self.id = str(uuid.uuid4())

        self._get_distance_to_centroid()

    def _get_distance_to_centroid(
        self,
    ) -> None:
        """Compute the distance from the detection to the centroid of the parent image."""
        if self.parent_image is None:
            return None

        with Image.open(self.parent_image) as image:
            width, height = image.size
            self.distance_to_centroid = math.sqrt(
                (self.x - width / 2) ** 2 + (self.y - height / 2) ** 2
            )

        return None

    def iou(self, other: "Detection") -> float:
        """Compute Intersection over Union (IoU) between this and another detection's bounding box."""
        return compute_iou(self.bbox(box_format="xyxy"), other.bbox(box_format="xyxy"))

    def geo_iou(self, other: "Detection") -> float:
        """Compute Intersection over Union (IoU) between this and another detection's geographic footprint."""
        return self.geographic_footprint.overlap_ratio(other.geographic_footprint)

    @property
    def geo_box(self):
        return self.geographic_footprint.box(box_format="xyxy")

    @classmethod
    def empty(cls, parent_image: str = None):
        """Create an empty detection object for a given parent image."""
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
        """Create a Detection object from a COCO-format dictionary.
        Args:
            coco (dict): COCO-format annotation.
            parent_image (str): Path to the parent image.
            image_gps_loc (str, optional): GPS location of the image.
            gps_loc (str, optional): GPS location of the detection.
        Returns:
            Detection: The created Detection object.
        """
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
        """Create a list of Detection objects from Label Studio format annotations.
        Args:
            detections (list): List of Label Studio detection dicts.
            image_path (str): Path to the image.
        Returns:
            list: List of Detection objects.
        """
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
        """Convert relative coordinates to absolute image coordinates by applying offsets.
        Args:
            x_offset (int): Offset to add to x coordinates.
            y_offset (int): Offset to add to y coordinates.
        """
        self.x_min += x_offset
        self.x_max += x_offset
        self.y_min += y_offset
        self.y_max += y_offset

    def update_values_from_tile(self, tile: "Tile"):  # TODO: add more values
        self.parent_image = tile.image_path
        self.image_gps_loc = tile.tile_gps_loc
        self.image_id = tile.id
        self.set_geographic_footprint_from_gps(gsd=tile.gsd, image_width=tile.width, image_height=tile.height)

        # if np.isnan(self.x_min) or np.isnan(self.y_min) or np.isnan(self.x_max) or np.isnan(self.y_max):
        if self.is_empty:
            logger.debug(f"Skipping empty detection with NaN values: {self.to_dict()}")
        else:
            self.update_detection_gps(gsd=tile.gsd,image=tile.load_image_data(), 
                                  image_gps_loc=tile.tile_gps_loc,
                                flight_height=tile.flight_specs.flight_height,
                                sensor_height=tile.flight_specs.sensor_height)
    
    def update_detection_gps(self, gsd: float,image:Image.Image, image_gps_loc:str,flight_height:float,sensor_height:float) -> None:
        self.gps_loc = compute_detection_gps(
                            x_center=self.x,
                            y_center=self.y,
                            image=image,
                            image_gps_loc=image_gps_loc,
                            flight_height=flight_height,
                            sensor_height=sensor_height,
                            gsd=gsd,
                        )

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
        if self.parent_image is None:
            raise ValueError("Parent image is not defined")

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

    def bbox(self, formating: str = "xyxy"):
        if formating == "xyxy":
            return [self.x_min, self.y_min, self.x_max, self.y_max]
        else:
            raise ValueError("only 'xyxy' supproted.")

    def clamp_bbox(self, x_range: tuple, y_range: tuple):
        for r in [x_range, y_range]:
            assert len(r) == 2, f"r={r}"
            assert r[1] >= r[0], f"r={r}"

        clamp = lambda x, r: max(min(x, r[1]), r[0])

        self.x_min = clamp(self.x_min, x_range)
        self.y_min = clamp(self.y_min, y_range)

        self.x_max = clamp(self.x_max, x_range)
        self.y_max = clamp(self.y_max, y_range)

        pass

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
        if self.gps_loc is None:
            return None, None, None

        point = geopy.Point.from_string(self.gps_loc)

        lat = point.latitude
        long = point.longitude
        alt = point.altitude * 1e3  # converting to meters

        return lat, long, alt

    def set_geographic_footprint_from_gps(
        self, gsd: float, image_width: int, image_height: int
    ):
        """
        Update the detection's geographic_footprint using its gps_loc field, similar to Tile._extract_geographic_footprint.
        width, height: dimensions of the image
        gsd: ground sample distance (cm/px)
        """
        if self.gps_loc is None:
            logger.debug("No gps coordinate found in the detection")
            return
        
        if self.is_empty:
            return None

        try:
            latitude, longitude, altitude = self.gps_as_decimals
        except Exception as e:
            logger.error(f"Failed to parse gps_loc: {self.gps_loc}, error: {e}")
            return

        xs = np.array([self.x_min, self.x_max])
        ys = np.array([self.y_min, self.y_max])
        xs_utm, ys_utm = ImageProcessor.generate_pixel_coordinates(
            x=xs,
            y=ys,
            lat_center=latitude,
            lon_center=longitude,
            W=image_width,
            H=image_height,
            gsd=gsd,
            return_as_utm=True,
        )
        self.geographic_footprint = GeographicBounds(
            north=max(ys_utm),
            south=min(ys_utm),
            east=max(xs_utm),
            west=min(xs_utm),
        )
        return None


@dataclass
class Tile:
    """Class representing an image tile."""

    image_path: str
    image_data: Image.Image = None

    id: Optional[str] = None

    width: int = None
    height: int = None

    x_offset: int = None
    y_offset: int = None

    parent_image: Optional[str] = None
    date: str = None
    parent_image_date: str = None

    tile_gps_loc: str = None
    latitude: float = None
    longitude: float = None
    altitude: float = None
    flight_specs: FlightSpecs = None

    geographic_footprint: Optional[GeographicBounds] = None
    gsd: float = None  # cm/px

    predictions: List[Detection] = None
    annotations: List[Detection] = None

    _pred_is_original: bool = False
    _annot_is_original: bool = False

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

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
        exif = self._extract_exif()

        if self.flight_specs is None:
            logger.debug("Flight specs are not provided.")
            return
        elif isinstance(self.flight_specs, FlightSpecs):
            pass
        else:
            raise ValueError(
                f"Flight specs is either None or not a 'FlightSpecs' object. Found {type(self.flight_specs)}"
            )

        sensor_height = self.flight_specs.sensor_height
        if sensor_height is None:
            sensor_height = SENSOR_HEIGHTS.get(exif["Model"])
            if sensor_height is None:
                logger.debug("Sensor height not found. Please provide it.")

        # self.gsd = self.flight_specs.gsd
        if self.flight_specs is not None:
            self.gsd = ImageProcessor.get_gsd(
                image_path=self.image_path,
                image=self.image_data,
                sensor_height=sensor_height,
                flight_height=self.flight_specs.flight_height,
                focal_length=self.flight_specs.focal_length,
            )

        self._extract_geographic_footprint()

        return None

    def load_image_data(self) -> Image.Image:
        if self.image_data is not None:
            return self.image_data
        else:
            return Image.open(self.image_path)

    @property
    def geo_box(self):
        return self.geographic_footprint.box(box_format="xyxy")

    def geo_iou(self, other: "Tile") -> float:
        return self.geographic_footprint.overlap_ratio(other.geographic_footprint)

    def _extract_exif(self):
        exif = GPSUtils.get_exif(file_name=self.image_path, image=self.image_data)
        return exif

    def _extract_geographic_footprint(self):
        if self.tile_gps_loc is None:
            logger.debug(
                "No gps coordinate found in the tile. Geographic footprint will not be set."
            )
            return
        xs = np.array([0, self.width])
        ys = np.array([0, self.height])

        xs_utm, ys_utm = ImageProcessor.generate_pixel_coordinates(
            x=xs,
            y=ys,
            lat_center=self.latitude,
            lon_center=self.longitude,
            W=self.width,
            H=self.height,
            gsd=self.gsd,
            return_as_utm=True,
        )

        self.geographic_footprint = GeographicBounds(
            north=max(ys_utm),
            south=min(ys_utm),
            east=max(xs_utm),
            west=min(xs_utm),
        )

        return None

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
            return_as_decimal=True,
        )
        if coords is not None:
            self.latitude, self.longitude, self.altitude = coords[0]
            self.tile_gps_loc = str(
                geopy.Point(self.latitude, self.longitude, self.altitude / 1e3)
            )
        else:
            logger.debug(f"Failed to extract GPS coordinates from {self.image_path}.")

        return None

    def set_offsets(self, x_offset: int, y_offset: int):
        self.y_offset = y_offset
        self.x_offset = x_offset

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

    def _nms(self, threshold: float = 0.5):
        if len(self.predictions) < 2:
            return self.predictions

        bboxs = torch.Tensor([det.bbox() for det in self.predictions])
        scores = torch.Tensor([det.score for det in self.predictions])

        # get indices of examples to keep
        indx = nms(boxes=bboxs, scores=scores, iou_threshold=threshold)

        return [self.predictions[i] for i in indx.tolist()]

    def filter_detections(
        self,
        method: str = "nms",
        threshold: float = 0.5,
        clamp: bool = True,
        confidence_threshold: float = 0.0,
    ):
        assert method == "nms", "only nms is supported"

        if len(self.predictions) < 1:
            return

        if confidence_threshold > 0.0:
            self.predictions = [
                det for det in self.predictions if det.score >= confidence_threshold
            ]

        if clamp:
            for det in self.predictions:
                det.clamp_bbox(x_range=(0, self.width), y_range=(0, self.height))

        self.predictions = self._nms(threshold)

        return None

    def update_detection_gps(
        self,
    ):
        if (self.tile_gps_loc is None) or (self.flight_specs is None):
            logger.info(f"No gps coordinate found in tile: {self.image_path}")
            return

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
                        flight_height=self.flight_specs.flight_height,
                        sensor_height=self.flight_specs.sensor_height,
                        gsd=self.gsd,
                    )
                except Exception as e:
                    logger.error(f"Failed to compute GPS location of detections. {e}")
                    det.gps_loc = None

            det.set_geographic_footprint_from_gps(
                gsd=self.gsd, image_width=self.width, image_height=self.height
            )

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
            df.at[0, "image_width"] = self.width  # add a row

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
            for det in data:
                det.update_values_from_tile(self)
            self.predictions = data

        else:
            self.predictions = [
                Detection.empty(parent_image=self.image_path),
            ]

        # if self.tile_gps_loc is None:
        #    self._extract_gps_coords()
        #    self._extract_geographic_footprint()

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
        assert df[["x_min", "x_max"]].to_numpy().max() <= self.width, (
            f"{df[['x_min', 'x_max']].to_numpy().max()} <= {self.width}"
        )
        assert df[["y_min", "y_max"]].to_numpy().max() <= self.height, (
            f"{df[['y_min', 'y_max']].to_numpy().max()} <= {self.height}"
        )

        return None

    def _get_patches(self, image: torch.Tensor, patch_size: int, stride: int):
        """
        Extract patches from an image tensor using unfolding.

        Args:
            image (torch.Tensor): Image tensor to extract patches from.

        Returns:
            torch.Tensor: Tensor of image patches.
        """
        if image.dim() == 2:
            image = image.unsqueeze(0)  # Add channel dimension
            squeeze_output = True
        else:
            squeeze_output = False

        C, H, W = image.shape

        # Use unfold to create tiles
        # First unfold along height dimension
        unfolded_h = image.unfold(1, patch_size, stride)

        # Then unfold along width dimension
        tiles = unfolded_h.unfold(2, patch_size, stride)

        # Reshape to get individual tiles
        tiles = tiles.contiguous().view(C, -1, patch_size, patch_size)
        tiles = tiles.permute(1, 0, 2, 3)

        if squeeze_output:
            tiles = tiles.squeeze(1)

        return tiles

    def _get_patches_and_offset_info(
        self, patch_size: int, stride: int
    ) -> tuple[torch.Tensor, dict]:
        """
        Extract patches from a tile and compute offset information.

        Args:
            tile (Tile): Tile object containing image data.
            patch_size (int): Size of each patch.

        Returns:
            tuple: (batch of RGB patches, offset information dictionary)
        """

        image = self.load_image_data()
        image = image.convert("RGB")
        image = PILToTensor()(image)

        if self.width <= patch_size or self.height <= patch_size:
            logger.debug("image is too small for patch extraction")
            offset_info = {
                "y_offset": [
                    0,
                ],
                "x_offset": [
                    0,
                ],
                "y_end": [
                    self.height,
                ],
                "x_end": [
                    self.width,
                ],
                "file_name": str(self.image_path),
            }
            return image, offset_info

        tiles = self._get_patches(image, patch_size, stride)

        C, H, W = image.shape
        x_indices = torch.arange(W).reshape(1, -1).expand(H, W)
        y_indices = torch.arange(H).reshape(-1, 1).expand(H, W)
        x_indices = self._get_patches(x_indices, patch_size, stride)
        y_indices = self._get_patches(y_indices, patch_size, stride)
        x_min = y_indices.min(1, True)[0].min(2)[0].squeeze().cpu().numpy()
        y_min = y_indices.min(1, True)[0].min(2)[0].squeeze().cpu().numpy()

        offset_info = {
            "y_offset": y_min.tolist(),
            "x_offset": x_min.tolist(),
            "y_end": (y_min + patch_size).tolist(),
            "x_end": (x_min + patch_size).tolist(),
            "file_name": str(self.image_path),
        }

        return tiles, offset_info
