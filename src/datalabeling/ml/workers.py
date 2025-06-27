# -*- coding: utf-8 -*-
"""
Created on Fri May 30 13:00:13 2025

@author: FADELCO
"""

import threading
import queue
from multiprocessing import Queue
import logging
from typing import Any, Dict, List, Optional, Tuple, Sequence
from itertools import chain
import time
from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset
from torchvision.transforms import PILToTensor
from PIL import Image
import os
import json
import base64
import requests
import traceback
from ultralytics.engine.results import Results as UltralyticsResults
from tqdm import tqdm
from torchvision.ops import nms
import asyncio
import aiohttp
import uuid
import datetime
import sqlite3

from .models import Detector, UltralyticsDetector, GroundingDinoDetector
from ..common.config import PredictionConfig
from ..common.base import Tile, Detection
from ..common.annotation_utils import GPSUtils, compute_detection_gps
from ..common.processor import DetectionsPostprocessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class LoadingDataset(Dataset):
    def __init__(
        self,
        data: Sequence[torch.Tensor],
        # offset_info: Sequence[dict],
        # metadata: Sequence[dict],
    ):
        super().__init__()

        data = list(data)
        self.data = torch.cat(data, dim=0)
        # self.offset_info = list(offset_info)
        # self.metadata = list(metadata)
        self.indices_map = dict()  # index in dataset -> offset_info

        index = 0
        for i, tensor in enumerate(data):
            for _ in range(tensor.shape[0]):
                self.indices_map[index] = i
                index += 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], torch.Tensor([self.indices_map[index]]).int()


class SharedBuffers:
    """
    Shared memory buffers for inter-thread communication

    GUIDELINES:
    - Adjust queue sizes based on your system's memory and processing speed
    - Consider using collections.deque with maxlen for memory-bounded buffers
    - Add more buffers if needed for your specific use case
    """

    def __init__(self, max_size: int = 64, timeout: int = 120):
        super(SharedBuffers, self)  # .__init__(name="SharedBuffers")

        self.raw_data_buffer = Queue(maxsize=max_size)
        self.detection_results_buffer = Queue(maxsize=max_size)
        self.final_results_buffer = Queue(maxsize=max_size)

        # Thread synchronization
        self._shutdown_event = threading.Event()
        self.is_closed = self._shutdown_event.is_set()
        self.logger = logging.getLogger("SharedBuffers")

        self.queue_kwargs = dict(block=True, timeout=timeout)

        self.counts = dict(data=0, detections=0, filtered_detections=0)

    def put(
        self, data: Any = None, detections: Any = None, filtered_detections: Any = None
    ):
        kwargs = self.queue_kwargs
        try:
            if data:
                self._put_data(data, **kwargs)
                self.counts["data"] += 1

            if detections:
                self._put_detections(detections, **kwargs)
                self.counts["detections"] += 1

            if filtered_detections:
                self._put_results(filtered_detections, **kwargs)
                self.counts["filtered_detections"] += 1

        except queue.Full:
            # self.logger.error("Queue is Full and timeout was exceeded.")
            raise ValueError("Queue is Full and timeout was exceeded.")

        except:
            traceback.print_exc()
            raise ValueError()

    def get(
        self,
        data: bool = False,
        detections: bool = False,
        filtered_detections: bool = False,
    ):
        kwargs = self.queue_kwargs

        assert (data + detections + filtered_detections) == 1, (
            "Specify only one item to get"
        )

        try:
            if data:
                self.counts["data"] = max(self.counts["data"] - 1, 0)
                return self._get_data(**kwargs)

            if detections:
                self.counts["detections"] = max(self.counts["detections"] - 1, 0)
                return self._get_detections(**kwargs)

            if filtered_detections:
                self.counts["filtered_detections"] = max(
                    self.counts["filtered_detections"] - 1, 0
                )
                return self._get_results(**kwargs)

        except queue.Empty:
            self.logger.info("Queue is Empty or has been consumed.")
            return "DONE"

        except Exception:
            self.logger.error("Un-catched error!")
            traceback.print_exc()
            # raise ValueError()
            return "DONE"

    @property
    def is_full_data_queue(
        self,
    ):
        return self.raw_data_buffer.full()

    def _put_data(self, data: dict, **kwargs):
        self.raw_data_buffer.put(data, **kwargs)

    def _get_data(self, **kwargs):
        return self.raw_data_buffer.get(**kwargs)

    def _put_detections(self, detections, **kwargs):
        self.detection_results_buffer.put(detections, **kwargs)

    def _get_detections(self, **kwargs):
        return self.detection_results_buffer.get(**kwargs)

    def _put_results(self, results, **kwargs):
        self.final_results_buffer.put(results, **kwargs)

    def _get_results(self, **kwargs):
        return self.final_results_buffer.get(**kwargs)

    def shutdown(self):
        """Signal all DataLoading thread to shutdown gracefully"""
        self._shutdown_event.set()


class DataLoadingThread(threading.Thread):
    """
    Thread responsible for loading and preparing data
    """

    def __init__(
        self,
        shared_buffers: SharedBuffers,
        data_source: Sequence[str],
        batchsize: int = 2,
        tile_size: int = 800,
        overlap_ratio: float = 0.2,
    ):
        super(DataLoadingThread, self).__init__(name="DataLoadingThread")

        self.shared_buffers = shared_buffers
        self.logger = logging.getLogger(self.name)

        assert isinstance(data_source, Sequence)
        self._source_iterator = iter(data_source)

        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.stride = int((1 - self.overlap_ratio) * self.tile_size)
        self.count = 0
        self.batchsize = batchsize

    # TODO: input type checking
    def _checks(
        self,
    ):
        pass

    def _get_patches(self, image: torch.Tensor):
        if image.dim() == 2:
            image = image.unsqueeze(0)  # Add channel dimension
            squeeze_output = True
        else:
            squeeze_output = False

        C, H, W = image.shape

        # Use unfold to create tiles
        # First unfold along height dimension
        unfolded_h = image.unfold(1, self.tile_size, self.stride)

        # Then unfold along width dimension
        tiles = unfolded_h.unfold(2, self.tile_size, self.stride)

        # Reshape to get individual tiles
        tiles = tiles.contiguous().view(C, -1, self.tile_size, self.tile_size)
        tiles = tiles.permute(1, 0, 2, 3)

        if squeeze_output:
            tiles = tiles.squeeze(1)

        return tiles

    def _load_data(
        self,
    ) -> Tile | str:
        """
        Load and preprocess data from your source

        Returns:
            Tile or str: Tile object containing image data or "DONE" if no more data
        """

        try:
            image_path = next(self._source_iterator)
            tile = Tile(image_path=image_path)
            self.logger.debug(f"Loading: sample {self.count} has been loaded.")

            self.count += 1

            return tile

        except StopIteration:
            return "DONE"

        except Exception:
            self.logger.error("Un-catched error!")
            traceback.print_exc()
            return "DONE"

    def _get_patches_from_tile(
        self, tile: Tile, patch_size: int
    ) -> tuple[torch.Tensor, dict]:
        """Extract patches from the tile

        Args:
            tile (Tile): tile object containing image data
            patch_size (int): patch size

        Returns:
            tuple[torch.Tensor, dict]: batch of RGB patches, offset information
        """

        image = tile.load_image_data()
        image = image.convert("RGB")
        image = PILToTensor()(image)

        if tile.width <= patch_size or tile.height <= patch_size:
            self.logger.debug("image is too small for patch extraction")
            offset_info = {
                "y_offset": [
                    0,
                ],
                "x_offset": [
                    0,
                ],
                "y_end": [
                    tile.height,
                ],
                "x_end": [
                    tile.width,
                ],
                "file_name": str(tile.image_path),
            }
            return image, offset_info

        tiles = self._get_patches(image)

        C, H, W = image.shape
        x_indices = torch.arange(W).reshape(1, -1).expand(H, W)
        y_indices = torch.arange(H).reshape(-1, 1).expand(H, W)
        x_indices = self._get_patches(x_indices)
        y_indices = self._get_patches(y_indices)
        x_min = y_indices.min(1, True)[0].min(2)[0].squeeze().cpu().numpy()
        y_min = y_indices.min(1, True)[0].min(2)[0].squeeze().cpu().numpy()

        offset_info = {
            "y_offset": y_min.tolist(),
            "x_offset": x_min.tolist(),
            "y_end": (y_min + patch_size).tolist(),
            "x_end": (x_min + patch_size).tolist(),
            "file_name": str(tile.image_path),
        }

        return tiles, offset_info

    def preprocess_data(self, tile: Tile) -> Tuple[TensorDataset, Dict]:
        """
        Preprocess raw data before detection
        - Extract patches from the tile
        - Normalize: image = image / 255.0
        - Color conversion: BGR -> RGB
        """
        # PLACEHOLDER - REPLACE WITH YOUR PREPROCESSING
        self.logger.debug(f"Preprocessing: sample {self.count} has been loaded.")

        # load as RGB and extract patches
        batch_of_patches, offset_info = self._get_patches_from_tile(
            tile=tile, patch_size=self.tile_size
        )

        # print(offset_info,"\n\n")

        if batch_of_patches.max() > 1.0:
            batch_of_patches = batch_of_patches / 255.0

        if len(batch_of_patches) == 3:
            batch_of_patches.unsqueeze_(0)

        # data = TensorDataset(batch_of_patches)

        return batch_of_patches, offset_info

    def _load_once(self):
        tile = self._load_data()

        if tile == "DONE":
            self.shared_buffers.put(data="DONE")
            self.logger.info(f"No more data to load. Loaded {self.count} samples.")
            return "DONE"

        # Preprocess data
        batch_of_patches, offset_info = self.preprocess_data(tile=tile)
        data_package = dict(
            metadata={
                "tile": tile,
                "idx": self.count,
            },
            data=batch_of_patches,
            offset_info=offset_info,
        )

        # push data
        self.shared_buffers.put(data=data_package)
        return "OK"

    def run(self):
        """Main thread execution loop"""
        self.logger.info("Starting data loading thread")

        while True:
            for _ in range(self.batchsize):
                if self._load_once() == "DONE":
                    return
                elif self._load_once() == "OK":
                    continue
                else:
                    return


class DetectionThread(threading.Thread):
    """
    Thread responsible for running object detection
    """

    def __init__(
        self,
        shared_buffers: SharedBuffers,
        config: PredictionConfig,
    ):
        super().__init__(name="DetectionThread")
        self.shared_buffers = shared_buffers
        self.model: Detector = None
        self.logger = logging.getLogger(self.name)

        self.session = None
        self.count = 0
        self.config = config
        self.max_wait_time = 0.1  # seconds for batch collection

    def set_model(
        self,
        model: Detector,
    ):
        if self.config.inference_service_url:
            assert model is None, "model should be None if using inference service"
            self.logger.info(f"Using deployment @ {self.config.inference_service_url}")
            return None

        assert isinstance(model, Detector), "Provide a valid Detector model"
        self.model = model
        self.model.warmup(imgsz=(self.config.tilesize, self.config.tilesize))

        self.logger.info("Model loaded successfully")

    def _pad_if_needed(self, batch: torch.Tensor, out_shape: tuple) -> torch.Tensor:
        # if batch size is less than expected, pad with zeros

        assert len(out_shape) == len(batch.shape)
        assert len(out_shape) == 4

        condition = any([a < b for a, b in zip(batch.shape, out_shape)])

        if condition:
            b, c, h, w = batch.shape
            padded = torch.zeros(out_shape)
            padded[:b, :c, :h, :w] = batch.clone()
            batch = padded

        return batch

    def _predict(self, batch: torch.Tensor):
        num_images = batch.shape[0]
        batch = self._pad_if_needed(
            batch, out_shape=(self.config.batch_size, *batch.shape[1:])
        )
        if self.config.inference_service_url:
            res = Detector.predict_url(
                image=batch,
                inference_service_url=self.config.inference_service_url,
                nms_iou=self.config.nms_iou,
                confidence_threshold=self.config.confidence_threshold,
            )

        else:
            res = self.model.predict(
                batch,
            )

        res = res[:num_images]

        return res

    def run_detection(
        self,
        data: LoadingDataset,
    ) -> dict:
        """
        Run object detection on input data
        """

        batchsize = (
            self.config.batch_size
            if self.config.inference_service_url
            else self.config.batch_size
        )

        loader = DataLoader(data, batch_size=batchsize, shuffle=False)

        if self.config.verbose:
            loader = tqdm(loader, desc="sliced_inference")

        results = []
        indices = []
        with torch.no_grad():
            for batch, index in loader:
                res = self._predict(batch)
                results.extend(res)
                indices.extend(index.cpu().flatten().tolist())

        # collect results per image
        # ever image is designed by an index
        results = {
            i: [res for j, res in enumerate(results) if i == indices[j]]
            for i in list(set(indices))
        }
        return results

    def collect_batch(self) -> LoadingDataset:
        """Collect tensors into a batch using hybrid time/size strategy"""
        batch = []
        offsets = []
        metadata = []
        start_time = time.time()

        while len(batch) < self.config.batch_size * 2:
            # Calculate remaining wait time
            elapsed = time.time() - start_time
            remaining_time = self.max_wait_time - elapsed

            if remaining_time <= 0:
                break

            # try:
            # Get data from buffer
            data_package = self.shared_buffers.get(data=True)

            if data_package == "DONE":
                self.logger.info("No more data to process. DONE.")
                break
                # return "DONE", None, None

            # if data_package == "empty":
            #     self.logger.info("Buffer is empty.")
            #     return "empty", None, None

            batch.append(data_package.pop("data"))
            offsets.append(data_package.pop("offset_info"))
            metadata.append(data_package.pop("metadata"))

        if len(batch) < 1:
            return "DONE", None, None

        dataset = LoadingDataset(
            data=batch,
        )

        return dataset, offsets, metadata

    def run(self):
        """Main thread execution loop"""
        self.logger.info("Starting detection thread")

        # Load model
        assert self.model or self.config.inference_service_url, (
            "Provide detection model or url to inference service i.e. YOLO"
        )

        while True:
            # get data
            data, offsets, metadata = self.collect_batch()

            if data == "DONE":
                self.shared_buffers.put(detections="DONE")
                break

            try:
                # Run detection
                t1 = time.perf_counter()
                detection_results = self.run_detection(data)
                t_end = (time.perf_counter() - t1) / len(data)

                self.logger.debug(f"Mean Detection time: {t_end:.3f}s")

            except Exception as e:
                traceback.print_exc()
                self.logger.error("Graceful shutdown.")
                self.shared_buffers.put(detections="DONE")
                break
                # raise ValueError()

            # Put in queue

            for i, results in detection_results.items():
                results_package = {
                    "detections": results,
                    "offset_info": offsets[i],
                    "metadata": metadata[i],
                    "detection_time": t_end,
                }
                self.shared_buffers.put(detections=results_package)


class PostProcessingThread(threading.Thread):
    """
    Thread responsible for post-processing detection results
    """

    def __init__(
        self,
        shared_buffers: SharedBuffers,
        config: PredictionConfig,
        label_map: dict = None,
    ):
        super().__init__(name="PostProcessingThread")
        self.shared_buffers = shared_buffers
        self.outputs = list()
        self.logger = logging.getLogger(self.name)
        self.config = config
        self.label_map = label_map or dict()
        self.count = 0
        self.roi_processor = None

    def set_processor(self, roi_processor: DetectionsPostprocessor):
        self.roi_processor = roi_processor

    def postprocess(
        self, detections: List[Detection], tile: Tile, offset_info: dict
    ) -> List[Detection]:
        # offset detections
        for i, pred in enumerate(detections):
            pred.parent_image = tile.image_path
            pred.image_gps_loc = tile.tile_gps_loc
            pred.gps_loc = None
            pred.to_absolute_coords(
                x_offset=offset_info["x_offset"][i], y_offset=offset_info["y_offset"][i]
            )

        tile.set_predictions(detections)
        tile.filter_detections(
            method="nms",
            threshold=self.config.nms_iou,
            clamp=True,
            confidence_threshold=self.config.confidence_threshold,
        )
        tile.update_detection_gps(
            sensor_height=self.config.sensor_height,
            flight_height=self.config.flight_height,
            gsd=self.config.gsd,
        )

        # post process roi
        if self.roi_processor:
            detections = self.roi_processor.run(
                detections,
                image=tile.load_image_data(),
                box_size=self.config.cls_imgsz,
                verbose=self.config.verbose,
            )

        self.count += 1

        return detections

    def _run_once(
        self,
    ):
        # Get detection results from buffer
        results_package = self.shared_buffers.get(detections=True)

        if results_package == "DONE":
            self.shared_buffers.put(filtered_detections="DONE")
            self.logger.info("No more data to process")
            return "DONE"

        # Apply post-processing
        raw_detections = results_package["detections"]
        tile = results_package["metadata"]["tile"]
        offset_info = results_package["offset_info"]

        try:
            t1 = time.perf_counter()
            filtered_detections = self.postprocess(
                raw_detections, tile=tile, offset_info=offset_info
            )
            t2 = time.perf_counter() - t1
            self.logger.debug(f"Postprocessing took: {t2:.3f}s")
            results_package["final_detections"] = filtered_detections
            results_package["postprocess_time"] = t2
            self.outputs.append(results_package)
            self.shared_buffers.put(filtered_detections=results_package)
        except:
            traceback.print_exc()
            self.shared_buffers.put(filtered_detections="DONE")
            self.logger.info("Graceful shutdown of thread.")
            return "DONE"

        return None

    def run(self):
        """Main thread execution loop"""
        self.logger.info("Starting post-processing thread")

        while True:
            status = self._run_once()
            if status == "DONE":
                break


class DetectionUploader(threading.Thread):
    """
    Thread responsible for uploading processed detections to the database.
    """

    def __init__(
        self,
        shared_buffers,
        sqlite_path: str,
    ):
        super().__init__(name="DetectionUploader")
        self.shared_buffers = shared_buffers
        self.logger = logging.getLogger(self.name)

        # Initialize DB connection
        self.conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        with self.conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS wildlife_detections (
                detection_id UUID PRIMARY KEY,
                species TEXT,
                latitude REAL,
                longitude REAL,
                altitude REAL,
                confidence REAL,
                image_gps TEXT,
                source_image TEXT,
                timestamp TEXT,
            );
            """)
            self.conn.commit()

    def _detection_to_dict(self, det: Detection):
        assert isinstance(det, Detection)
        lat, long, alt = det.gps_as_decimals
        return {
            "detection_id": uuid.uuid4(),
            "species": det.class_name,
            "latitude": round(lat, 8),
            "longitude": round(long, 8),
            "altitude": alt,
            "confidence": round(det.score, 3),
            "image_gps": det.image_gps_loc,
            "source_image": str(det.parent_image),
        }

    def _upload_detections(self, detections):
        rows = []
        for det in detections:
            det_dict = self._detection_to_dict(det)

            rows.append(
                (
                    det_dict["detection_id"],
                    det_dict["species"],
                    det_dict["latitude"],
                    det_dict["longitude"],
                    det_dict["confidence"],
                    det_dict["image_gps"],
                    det_dict["source_image"],
                    str(datetime.datetime.utcnow()),
                )
            )

        self.cursor.execute(
            """
                INSERT INTO wildlife_detections (
                    detection_id, species, latitude, longitude, confidence,
                    image_gps, source_image, timestamp
                ) VALUES %s
            """,
            rows,
        )
        self.conn.commit()

    def run(self):
        self.logger.info("Starting detection upload thread.")

        try:
            while True:
                data_package = self.shared_buffers.get(filtered_detections=True)

                if data_package == "DONE":
                    self.logger.info("Upload thread received shutdown signal.")
                    break

                detections = data_package["final_detections"]

                t1 = time.perf_counter()
                self._upload_detections(detections)
                t2 = time.perf_counter() - t1

                self.logger.info(
                    f"Uploaded {len(detections)} detections in {t2:.2f} seconds."
                )
        except Exception as e:
            self.logger.exception("Error during upload:")
        finally:
            self.conn.close()
            self.logger.info("Database connection closed.")


class ObjectDetectionSystem:
    """
    Main system coordinator
    """

    def __init__(
        self,
        config: PredictionConfig,
        buffer_size=32,
        timeout=15,
        detection_label_map: dict = None,
    ):
        # Initialize shared buffers
        self.shared_buffers = SharedBuffers(max_size=buffer_size, timeout=timeout)

        # Initialize threads
        self.data_thread: DataLoadingThread = None
        self.detection_thread: DetectionThread = None
        self.postprocess_thread: PostProcessingThread = None
        self.detection_uploader: DetectionUploader = None
        self.label_map = detection_label_map
        self.config = config
        self._detection_model = None
        self._detection_task = None
        self._detection_model_path = None
        self._roi_processor = None

        self.logger = logging.getLogger("ObjectDetectionSystem")

    def set_processor(self, roi_processor: DetectionsPostprocessor):
        self._roi_processor = roi_processor

    def set_model(self, model: Detector):
        self._detection_model = model

    def _set_handlers(
        self,
    ):
        self.detection_thread.set_model(
            model=self._detection_model,
        )

        self.postprocess_thread.set_processor(roi_processor=self._roi_processor)

    @property
    def outputs(
        self,
    ):
        if not self.postprocess_thread.is_alive():
            return self.postprocess_thread.outputs

        return None

    def _process_pipeline(self):
        """Start all threads"""
        self.logger.info("Starting Object Detection System")

        self._set_handlers()

        # Start threads in order
        self.data_thread.start()
        self.detection_thread.start()
        self.postprocess_thread.start()
        self.detection_uploader.start()

        self.logger.info("All threads started")

        self._join_workers()

    def _join_workers(self):
        """Stop all threads gracefully"""
        self.logger.info("Stopping Object Detection System")

        # Signal shutdown
        self.shared_buffers.shutdown()

        # Wait for threads to finish
        self.data_thread.join()
        self.detection_thread.join()
        self.postprocess_thread.join()
        self.detection_uploader.join()

        self.logger.info("All threads stopped")
        return None

    def run(
        self, images_paths: Sequence[str], img_loading_batch: int = 4
    ) -> List[List[Detection]]:
        """
        Run the system for a specified duration or until stopped
        """
        images_paths = list(images_paths)

        # Initialize threads
        self.data_thread = DataLoadingThread(
            self.shared_buffers,
            data_source=images_paths,
            tile_size=self.config.tilesize,
            batchsize=img_loading_batch,
            overlap_ratio=self.config.overlap_ratio,
        )

        self.detection_thread = DetectionThread(self.shared_buffers, config=self.config)

        self.postprocess_thread = PostProcessingThread(
            self.shared_buffers,
            config=self.config,
            label_map=self.label_map,
        )

        self.detection_uploader = DetectionUploader()

        self._process_pipeline()

        out = self.outputs

        detections = [o["final_detections"] for o in out]

        return detections
