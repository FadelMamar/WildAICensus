import logging
import traceback
from pathlib import Path

import geopy
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from torch.utils.data import DataLoader, TensorDataset
import requests, base64

# from label_studio_ml.utils import (get_env, get_local_path)
from tqdm import tqdm
from ultralytics import YOLO

from ..common.annotation_utils import GPSUtils, ImageProcessor
from ..common.config import Detection, PredictionConfig

logger = logging.getLogger(__name__)


class Detector(object):
    def __init__(
        self, detection_model: UltralyticsDetectionModel, config: PredictionConfig
    ):
        self.config = config
        self.detection_model = detection_model

    def set_detection_model(self, detection_model, path_to_weights=None):
        if detection_model:
            self.detection_model = detection_model

        else:
            logger.info(f"Computing device: {self.config.device}")

            self.detection_model = UltralyticsDetectionModel(
                model=YOLO(path_to_weights, task="detect"),
                confidence_threshold=self.config.confidence_threshold,
                image_size=self.config.imgsz,
                device=self.config.device,
            )

    # TODO: batch predictions with slicing
    def predict(
        self,
        image: Image.Image = None,
        inference_service_url: str = None,
        image_path: str = None,
        sahi_prostprocess: float = "NMS",
        override_tilesize: int = None,
        postprocess_match_threshold: float = 0.5,
        timeout: int = 3 * 60,
        nms_iou: float = None,
        verbose: int = 0,
    ) -> list[Detection]:
        # predict using inference service
        if isinstance(inference_service_url, str):
            detections, image_gps = Detector.predict_url(
                image_path=image_path,
                inference_service_url=inference_service_url,
                timeout=timeout,
            )
            detections = self._format_detections(
                detections=detections,
                image_path=image_path,
                image_gps_loc=image_gps,
                image=Image.open(image_path),
            )

        # predict using local model
        if image is None:
            assert image_path is not None, "Provide the image path."
            image = Image.open(image_path)
        else:
            assert isinstance(image, Image.Image)

        if self.config.use_sliding_window:
            tilesize = override_tilesize or self.config.tilesize
            result = get_sliced_prediction(
                image,
                self.detection_model,
                slice_height=tilesize,
                slice_width=tilesize,
                overlap_height_ratio=self.config.overlap_ratio,
                overlap_width_ratio=self.config.overlap_ratio,
                postprocess_type=sahi_prostprocess,
                postprocess_match_metric="IOU",
                verbose=verbose,
                postprocess_match_threshold=postprocess_match_threshold or nms_iou,
            )
            detections = result.to_coco_annotations()
        else:
            result = get_prediction(
                image=image,
                detection_model=self.detection_model,
                shift_amount=[0, 0],
                full_shape=None,
                postprocess=None,
                verbose=verbose,
            )
            detections = result.to_coco_annotations()

        # image gps coordinate
        gps_info = GPSUtils.get_gps_coord(
            file_name=image_path, image=image, return_as_decimal=False
        )
        if isinstance(gps_info, tuple):
            gps_coords = gps_info[0]
        else:
            gps_coords = gps_info

        detections = self._format_detections(
            detections=detections,
            image_path=image_path,
            image_gps_loc=gps_coords,
            image=image,
        )

        return detections

    def _format_detections(
        self,
        detections: list[Detection],
        image_path: str,
        image_gps_loc: str,
        image: Image.Image,
    ):
        # format detections
        detections = [
            Detection.from_coco(
                pred, parent_image=image_path, image_gps_loc=image_gps_loc, gps_loc=None
            )
            for pred in detections
        ]

        # add detections gps
        for det in detections:
            det.gps_loc = self.compute_detection_gps(
                x_center=det.x,
                y_center=det.y,
                image=image,
                image_gps_loc=det.image_gps_loc,
            )
        return detections

    @staticmethod
    def predict_url(
        image_path: str,
        inference_service_url: str = "http://127.0.0.1:4141/predict",
        timeout=3 * 60,
    ):
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "image": img_b64,
            "sahi_prostprocess": "NMS",
            "override_tilesize": None,  # tilesize to use for
            "postprocess_match_threshold": 0.5,
            "nms_iou": None,
        }

        resp = requests.post(
            inference_service_url,
            json=payload,
            timeout=timeout,
        ).json()

        detections = resp["detections"]
        image_gps = resp["image_gps"]

        return detections, image_gps

    # TODO: to debug and optimize
    def sliced_prediction(
        self,
        image_path: str,
        image: Image = None,
        patchsize: int = 640,
        stride: int = 128,
    ):
        if image is None:
            assert image_path is not None, "Provide the image path."
            image = Image.open(image_path)

        image = image.convert("RGB")
        to_tensor = T.ToTensor()
        image = to_tensor(image).unsqueeze(0)
        image = image[:, ::-1, :, :]

        # unfold gives shape [1, C*ph*pw, L] where L = number of patches
        patches_flat = F.unfold(
            image, kernel_size=(patchsize, patchsize), stride=(stride, stride)
        )
        batch_of_patches = patches_flat.transpose(1, 2).reshape(
            -1, 3, patchsize, patchsize
        )

        # number of patches along width, height:
        H, W = image.shape[2:]
        n_w = (W - patchsize) // stride + 1
        n_h = (H - patchsize) // stride + 1

        # for patch index k in [0..L-1]:
        row_idx = lambda k: k // n_h
        col_idx = lambda k: k % n_w

        # top‐left pixel of this patch in original:
        y0 = lambda i: row_idx(i) * stride
        x0 = lambda j: col_idx(j) * stride

        dataset = TensorDataset(batch_of_patches)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        results = []
        indexes = []
        offset = 0
        with torch.no_grad():
            for (batch,) in loader:
                res = self.detection_model.model(batch)
                results.append(res)
                indexes = indexes + list(range(offset, offset + batch.shape[0]))
                offset += batch.shape[0]

        # TODO: debug top-left pixels in the original image
        y0_x0 = [(y0(i), x0(i)) for i in indexes]

        return results, y0_x0

    def predict_directory(
        self,
        path_to_dir: str = None,
        images_paths: list[str] = None,
        as_dataframe: bool = True,
        save_path: str = None,
    ) -> dict[str, list[Detection]] | pd.DataFrame:
        """Computes predictions on a directory

        Args:
            path_to_dir (str): path to directory with images. Defaults to None
            images_list (list): paths of images to run the detection on
            as_dataframe (bool): returns results as pd.DataFrame
            save_path (str) : converts to dataframe and then save

        Returns:
            dict: a directory with the schema {image_path:prediction_coco_format}
        """

        assert (path_to_dir is None) + (images_paths is None) < 2, (
            "Both should not be given."
        )
        results = {}
        paths = images_paths or list(Path(path_to_dir).iterdir())
        for image_path in tqdm(paths, desc="Computing predictions..."):
            try:
                pred = self.predict(
                    image=None,
                    image_path=image_path,
                )
            except Exception as e:
                logger.error(e)
                logger.error(f"Failed for {image_path}")
                continue

            results.update({str(image_path): pred})

        if len(results) < 1:
            logger.info("0 detections.")

        # returns as df or save
        if as_dataframe or save_path:
            results = self._format_results_as_dataframe(results)

            if save_path is not None:
                try:
                    results.to_json(save_path, orient="records", indent=2)
                except Exception:
                    logger.info("!!!Failed to save results as json!!!\n")
                    traceback.print_exc()

        return results

    def compute_detection_gps(
        self,
        x_center,
        y_center,
        image: Image.Image,
        image_gps_loc: str,
        flight_height: int = 180,
        sensor_height: int = 24,
        gsd: float = None,
    ):
        # None
        if image_gps_loc is None:
            return None

        assert isinstance(image, Image.Image), "Provide PIL Image"

        # compute detection
        W, H = image.size

        lat_center, lon_center, alt = GPSUtils.to_decimal(image_gps_loc)

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

    def _format_results_as_dataframe(
        self, results: dict[str, list[Detection]]
    ) -> pd.DataFrame:
        if len(results) < 1:
            return pd.DataFrame()

        unravel_dict = []
        for img_path, detections in results.items():
            for det in detections:
                unravel_dict.append(dict(file_name=img_path, **det.to_dict()))

        dfs = pd.DataFrame.from_dict(unravel_dict)

        dfs["bbox_w"] = dfs["x_max"] - dfs["x_min"]
        dfs["bbox_h"] = dfs["y_max"] - dfs["y_min"]

        # converting gps coords to decimal
        dfs[["img_Latitude", "img_Longitude", "img_Elevation"]] = (
            dfs["image_gps_loc"].apply(GPSUtils.to_decimal).apply(pd.Series)
        )
        dfs[["Latitude", "Longitude", "Elevation"]] = (
            dfs["gps_loc"].apply(GPSUtils.to_decimal).apply(pd.Series)
        )

        return dfs
