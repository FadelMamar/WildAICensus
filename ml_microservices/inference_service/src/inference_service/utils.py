from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Sequence
import os, json
import logging
import torch
from pathlib import Path
import mlflow
from ultralytics.engine.results import Results as UltralyticsResults
from torchvision.transforms import PILToTensor
from torchvision.ops import nms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset


logger = logging.getLogger(__name__)


def load_registered_model(
    alias,
    name,
    load_unwrapped: bool = True,
):
    client = mlflow.MlflowClient()

    version = client.get_model_version_by_alias(name=name, alias=alias).version
    modelversion = f"{name}:{version}"
    modelURI = f"models:/{name}/{version}"

    dwnd_location = Path(os.environ.get("WEIGHTS_PATH", "./model_weights")) / f"{name}"
    dwnd_location = dwnd_location / str(version)
    if dwnd_location.exists():
        model = mlflow.pyfunc.load_model(str(dwnd_location))
    else:
        dwnd_location.mkdir(parents=True, exist_ok=True)
        model = mlflow.pyfunc.load_model(modelURI, dst_path=str(dwnd_location))

    metadata = dict(version=modelversion)
    metadata.update(model.metadata.metadata)

    if load_unwrapped:
        model = model.unwrap_python_model().model

    return model, metadata


@dataclass
class Detection:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    label: int
    class_name: str
    score: float = None


@dataclass
class Tile:
    """Class representing an image tile."""

    image_path: str
    image_data: Image.Image
    width: int = None
    height: int = None
    x_offset: int = None
    y_offset: int = None

    def __post_init__(self):
        if self.image_data is None:
            self.width, self.height = Image.open(self.image_path).size

        else:
            self.width, self.height = self.image_data.size

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


class Detector(object):
    def __init__(
        self,
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mlflow_model_name = os.environ.get("MODEL_NAME", "labeler")
        self.mlflow_model_alias = os.environ.get("MODEL_ALIAS", "demo")

        self.tracking_url = os.environ.get(
            "MLFLOW_TRACKING_URI", "http://mlflow_service:5000"
        )

        self.nms_iou = float(os.environ.get("NMS_IOU", 0.5))
        self.label_map = json.loads(os.environ.get("LABEL_MAP", "{}"))
        self.tilesize = None
        self.batch_size = None
        self.overlap_ratio = float(os.environ.get("OVERLAP_RATIO", 0.2))

        self.model = None
        self.modelURI = None
        self.model_metadata = None

        self._set_model()

    def _set_model(self):
        mlflow.set_tracking_uri(self.tracking_url)

        self.model, self.model_metadata = load_registered_model(
            alias=self.mlflow_model_alias,
            name=self.mlflow_model_name,
            load_unwrapped=True,
        )
        print(
            "Loading model from:",
            self.mlflow_model_name,
            "model_metadata",
            self.model_metadata,
        )

        self.batch_size = self.model_metadata.get("batch") or int(
            os.environ.get("BATCH_SIZE", 1)
        )
        self.tilesize = self.model_metadata.get("tilesize") or int(
            os.environ.get("TILESIZE", 800)
        )

        # warmup
        logger.info(
            f"Running warmup with batch_size={self.batch_size} and tilesize={self.tilesize}"
        )
        self.model(
            torch.zeros((self.batch_size, 3, self.tilesize, self.tilesize)),
            verbose=False,
        )

    def _load_tile_as_patches(self, tile: Tile):
        stride = int((1 - self.overlap_ratio) * self.tilesize)
        batch_of_patches, offset_info = tile.as_batch(
            tile_size=self.tilesize, stride=stride
        )

        logger.debug(f"Tiled image shape: {batch_of_patches.shape}")

        if batch_of_patches.max() > 1.0:
            batch_of_patches = batch_of_patches / 255.0

        dataset = TensorDataset(batch_of_patches)

        return dataset, offset_info

    def predict(self, images: List[Image.Image]) -> List[dict]:
        out = []
        for image in images:
            assert isinstance(image, Image.Image), (
                "All images should be PIL Image objects."
            )
            o = self._predict(tile=Tile(image_path=None, image_data=image))

            out.append(o)

        return out

    def _predict(
        self,
        tile: Tile,
    ):
        dataset, offset_info = self._load_tile_as_patches(tile)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        def trim_result(results: list[UltralyticsResults]) -> list:
            trimmed = []
            for res in results:
                if res.obb is not None:
                    boxes = res.obb
                else:
                    boxes = res.boxes
                trimmed.append(boxes)
            return trimmed

        results = []
        with torch.no_grad():
            for (batch,) in loader:
                num_images = batch.shape[0]
                if num_images != self.batch_size:
                    # if batch size is less than expected, pad with zeros
                    padding = torch.zeros(
                        (self.batch_size - batch.shape[0], *batch.shape[1:])
                    )
                    batch = torch.cat([batch, padding], dim=0)
                res = self.model(batch, verbose=False)
                res = res[:num_images]
                results = results + trim_result(res)

        detections = self.postprocess(
            results=results, tile=tile, offset_info=offset_info
        )

        return detections

    def postprocess(
        self, results: list[UltralyticsResults], tile: Tile, offset_info: dict
    ):
        # ultralytics results to coco
        detections = self._result_to_coco(
            results,
            offset_info=offset_info,
            tile_width=tile.width,
            tile_height=tile.height,
        )

        return detections

    def _result_to_coco(
        self,
        result: list,
        tile_width: int,
        tile_height: int,
        offset_info: dict,
    ) -> list[dict]:
        bboxs = []
        conf = []
        label = []

        for i, boxes in enumerate(result):
            if boxes.xyxy.cpu().numel() == 0:
                continue

            # mapping to untiled coordinates
            bbox = boxes.xyxy.cpu().clone()

            assert torch.min(bboxs) >= 0.0, (
                f"Found negative bounding box coordinates: {bboxs}"
            )
            assert torch.max(bboxs[:, [0, 2]]) <= tile_width, (
                f"Bounding box x-coordinates exceed tile width: {bboxs[:, [0, 2]]} > {tile_width}"
            )
            assert torch.max(bboxs[:, [1, 3]]) <= tile_height, (
                f"Bounding box y-coordinates exceed tile height: {bboxs[:, [1, 3]]} > {tile_height}"
            )

            bbox[:, [0, 2]] = bbox[:, [0, 2]] + offset_info["x_offset"][i]
            bbox[:, [1, 3]] = bbox[:, [1, 3]] + offset_info["y_offset"][i]

            bboxs.append(bbox)
            conf.append(boxes.conf.cpu())
            label.append(boxes.cls.cpu())

        if len(bboxs) == 0:
            return []

        bboxs = torch.vstack(bboxs)
        conf = torch.hstack(conf)
        label = torch.hstack(label).long()

        assert torch.min(bboxs) >= 0.0
        assert torch.max(bboxs[:, [0, 2]]) <= tile_width
        assert torch.max(bboxs[:, [1, 3]]) <= tile_height

        # Non-max suppression
        indx = nms(boxes=bboxs, scores=conf, iou_threshold=self.nms_iou)

        # xyxy -> coco xywh
        bboxs[:, 2] = bboxs[:, 2] - bboxs[:, 0]
        bboxs[:, 3] = bboxs[:, 3] - bboxs[:, 1]

        # retaining selected boxes
        bboxs = bboxs.tolist()
        conf = conf.tolist()
        label = label.tolist()

        # to coco format
        coco = [
            dict(
                bbox=bboxs[i],
                category_id=label[i],
                category_name=self.label_map.get(label[i], None),
                score=conf[i],
                file_name=offset_info["file_name"][i],
            )
            for i in indx.tolist()
        ]

        return coco
