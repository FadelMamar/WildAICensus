# -*- coding: utf-8 -*-
"""
Created on Fri May 30 13:00:13 2025

@author: FADELCO
"""

import threading
import queue
from queue import Queue
import logging
from typing import Any, Dict, List, Optional, Tuple, Sequence
from itertools import chain
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import PILToTensor

import traceback
from tqdm import tqdm
from torchvision.ops import nms
import asyncio
import aiohttp
import uuid
import datetime
import sqlite3


from .models import Detector
from ..common.config import PredictionConfig
from ..common.base import Tile, Detection
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
        """
        Initialize the LoadingDataset with a sequence of tensors.

        Args:
            data (Sequence[torch.Tensor]): Sequence of tensors to be concatenated into the dataset.
        """
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
        """
        Return the number of samples in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, index):
        """
        Retrieve a sample and its corresponding index from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (sample tensor, index tensor)
        """
        return self.data[index], torch.Tensor([self.indices_map[index]]).int()


class SharedBuffers:
    """
    Shared memory buffers for inter-thread communication

    GUIDELINES:
    - Adjust queue sizes based on your system's memory and processing speed
    - Consider using collections.deque with maxlen for memory-bounded buffers
    - Add more buffers if needed for your specific use case
    """

    def __init__(self, max_size: int = 64, timeout: int = 15):
        """
        Initialize shared memory buffers for inter-thread communication.

        Args:
            max_size (int): Maximum size for each queue.
            timeout (int): Timeout for queue operations in seconds.
        """
        super(SharedBuffers, self)  # .__init__(name="SharedBuffers")

        self.raw_data_buffer = Queue(maxsize=max_size)
        self.detection_results_buffer = Queue(maxsize=max_size)
        self.final_results_buffer = Queue(maxsize=max_size)

        # Thread synchronization
        self._shutdown_event = threading.Event()
        self.is_shutdown = self._shutdown_event.is_set()
        self.logger = logging.getLogger("SharedBuffers")

        self.queue_kwargs = dict(block=True, timeout=timeout)

        self.counts = dict(data=0, detections=0, filtered_detections=0)

    def put(
        self, data: Any = None, detections: Any = None, filtered_detections: Any = None
    ):
        """
        Put data, detections, or filtered detections into the appropriate buffer.

        Args:
            data (Any): Raw data to put in the buffer.
            detections (Any): Detection results to put in the buffer.
            filtered_detections (Any): Post-processed detection results to put in the buffer.
        """
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

        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"msg:{e}")

    def get(
        self,
        data: bool = False,
        detections: bool = False,
        filtered_detections: bool = False,
    ):
        """
        Retrieve an item from the specified buffer.

        Args:
            data (bool): If True, get raw data.
            detections (bool): If True, get detection results.
            filtered_detections (bool): If True, get filtered detection results.

        Returns:
            Any: The requested item from the buffer, or "DONE" if empty.
        """
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
            request = (
                "data"
                if data
                else "detections"
                if detections
                else "filtered_detections"
            )
            msg = f"{request} Queue is Empty or has been consumed."
            self.logger.info(msg)
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
        """
        Check if the raw data buffer is full.

        Returns:
            bool: True if the raw data buffer is full, False otherwise.
        """
        return self.raw_data_buffer.full()

    def _put_data(self, data: dict, **kwargs):
        """
        Put raw data into the raw data buffer.

        Args:
            data (dict): Data to put in the buffer.
        """
        self.raw_data_buffer.put(data, **kwargs)

    def _get_data(self, **kwargs):
        """
        Get raw data from the raw data buffer.

        Returns:
            Any: Data from the buffer.
        """
        return self.raw_data_buffer.get(**kwargs)

    def _put_detections(self, detections, **kwargs):
        """
        Put detection results into the detection results buffer.

        Args:
            detections: Detection results to put in the buffer.
        """
        self.detection_results_buffer.put(detections, **kwargs)

    def _get_detections(self, **kwargs):
        """
        Get detection results from the detection results buffer.

        Returns:
            Any: Detection results from the buffer.
        """
        return self.detection_results_buffer.get(**kwargs)

    def _put_results(self, results, **kwargs):
        """
        Put filtered detection results into the final results buffer.

        Args:
            results: Filtered detection results to put in the buffer.
        """
        self.final_results_buffer.put(results, **kwargs)

    def _get_results(self, **kwargs):
        """
        Get filtered detection results from the final results buffer.

        Returns:
            Any: Filtered detection results from the buffer.
        """
        return self.final_results_buffer.get(**kwargs)

    def shutdown(self):
        """
        Signal all DataLoading threads to shutdown gracefully.
        """
        self._shutdown_event.set()
        self.is_shutdown = True


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
        """
        Initialize the DataLoadingThread.

        Args:
            shared_buffers (SharedBuffers): Shared buffer object for communication.
            data_source (Sequence[str]): Sequence of image paths to load.
            batchsize (int): Number of samples to load per batch.
            tile_size (int): Size of each tile to extract from images.
            overlap_ratio (float): Overlap ratio for tiling.
        """
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
        """
        Placeholder for input type checking or other checks.
        """
        pass

    def _get_patches(self, image: torch.Tensor):
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
        Load and preprocess data from the source iterator.

        Returns:
            Tile or str: Tile object containing image data or "DONE" if no more data.
        """

        try:
            image_path = next(self._source_iterator)
            tile = Tile(image_path=image_path)
            self.logger.debug(f"Loading: sample {self.count}")

            self.count += 1

            return tile

        except StopIteration:
            return "DONE"

        except Exception:
            self.logger.error("Un-catched error!")
            traceback.print_exc()
            return "ERROR"

    def _get_patches_from_tile(
        self, tile: Tile, patch_size: int
    ) -> tuple[torch.Tensor, dict]:
        """
        Extract patches from a tile and compute offset information.

        Args:
            tile (Tile): Tile object containing image data.
            patch_size (int): Size of each patch.

        Returns:
            tuple: (batch of RGB patches, offset information dictionary)
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

    def preprocess_data(self, tile: Tile) -> tuple[torch.Tensor, dict]:
        """
        Preprocess raw data before detection by extracting and normalizing patches.

        Args:
            tile (Tile): Tile object to preprocess.

        Returns:
            tuple: (batch of patches, offset information)
        """
        self.logger.debug(f"Preprocessing: sample {self.count} has been loaded.")

        # load as RGB and extract patches
        batch_of_patches, offset_info = self._get_patches_from_tile(
            tile=tile, patch_size=self.tile_size
        )

        if batch_of_patches.max() > 1.0:
            batch_of_patches = batch_of_patches / 255.0

        if len(batch_of_patches) == 3:
            batch_of_patches.unsqueeze_(0)

        return batch_of_patches, offset_info

    def _load_once(self) -> str:
        """
        Load and preprocess a single data sample, then put it in the shared buffer.

        Returns:
            str: "DONE" if no more data, otherwise "OK".
        """
        tile = self._load_data()

        if tile == "DONE":
            self.shared_buffers.put(data="DONE")
            self.logger.debug(f"No more data to load. Loaded {self.count} samples.")
            return "DONE"
        elif tile == "ERROR":
            self.shared_buffers.put(data="ERROR")
            self.logger.error("Error loading data")
            return "ERROR"

        # Preprocess data
        try:
            batch_of_patches, offset_info = self.preprocess_data(tile=tile)
            data_package = dict(
                metadata={
                    "tile": tile,
                    "idx": self.count,
                },
                data=batch_of_patches,
                offset_info=offset_info,
                number_patches=batch_of_patches.shape[0],
            )
            # push data
            self.shared_buffers.put(data=data_package)
            return "OK"
        except:
            traceback.print_exc()
            return "ERROR"

    def run(self):
        """
        Main thread execution loop for loading data batches.
        """
        self.logger.info("Starting...")

        while True:
            for _ in range(self.batchsize):
                status = self._load_once()
                if status == "DONE":
                    self.logger.info("DONE")
                    return
                elif status == "OK":
                    continue
                elif status == "ERROR":
                    return
                else:
                    self.logger.error("Unknown error")
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
        """
        Initialize the DetectionThread.

        Args:
            shared_buffers (SharedBuffers): Shared buffer object for communication.
            config (PredictionConfig): Prediction configuration.
        """
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
        """
        Set the detection model for the thread.

        Args:
            model (Detector): Detection model to use.
        """
        if self.config.inference_service_url:
            if model is not None:
                raise ValueError("model should be None if using inference service")
            self.logger.info(f"Using deployment @ {self.config.inference_service_url}")
            return

        if not isinstance(model, Detector):
            raise ValueError("Provide a valid Detector model")
        self.model = model
        self.model.warmup(imgsz=(self.config.tilesize, self.config.tilesize))

        self.logger.info("Model loaded successfully")

    def _pad_if_needed(self, batch: torch.Tensor, out_shape: tuple) -> torch.Tensor:
        """
        Pad the batch tensor with zeros if its shape is less than the expected output shape.

        Args:
            batch (torch.Tensor): Input batch tensor.
            out_shape (tuple): Desired output shape.

        Returns:
            torch.Tensor: Padded batch tensor.
        """
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
        """
        Run prediction on a batch of images using the detection model or inference service.

        Args:
            batch (torch.Tensor): Batch of images to predict on.

        Returns:
            list: Prediction results for each image in the batch.
        """
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
            if isinstance(self.model, Detector):
                res = self.model.predict(batch)
            else:
                raise ValueError("Detection model is not set.")

        res = res[:num_images]

        return res

    def run_detection(
        self,
        data,
    ) -> dict:
        """
        Run object detection on input data and collect results per image.

        Args:
            data (LoadingDataset): Dataset to run detection on.

        Returns:
            dict: Mapping from image index to detection results.
        """

        loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=False)

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

    # TODO: increase number of images in dataloader
    def collect_batch(self):
        """
        Collect tensors into a batch using a hybrid time/size strategy.

        Returns:
            tuple: (dataset, offsets, metadata) or ("DONE", None, None) if no more data.
        """
        batch = []
        offsets = []
        metadata = []
        start_time = time.time()

        while len(batch) < 1:
            # Calculate remaining wait time
            elapsed = time.time() - start_time
            remaining_time = self.max_wait_time - elapsed

            if remaining_time <= 0:
                break

            # try:
            # Get data from buffer
            data_package = self.shared_buffers.get(data=True)

            if data_package == "DONE":
                break

            elif data_package == "ERROR":
                self.logger.error("Error collecting batch")
                return "ERROR", None, None

            if not isinstance(data_package, dict):
                self.logger.error("data_package is not a dict")
                return "ERROR", None, None

            batch.append(data_package.get("data"))
            offsets.append(data_package.get("offset_info"))
            metadata.append(data_package.get("metadata"))

        if len(batch) == 0:
            return "DONE", None, None

        dataset = LoadingDataset(
            data=batch,
        )

        return dataset, offsets, metadata

    def end_thread(self, msg):
        """
        End the thread gracefully by putting a shutdown message in the shared buffer.

        Args:
            msg (str): Shutdown message to put in the buffer.
        """
        self.shared_buffers.put(detections=msg)
        self.logger.info(f"Thread ended with message: {msg}")

    def run(self):
        """
        Main thread execution loop for running detection on batches.
        """
        self.logger.info("Starting...")

        # Load model
        assert self.model or self.config.inference_service_url, (
            "Provide detection model or url to inference service"
        )

        while True:
            try:
                # get data
                data, offsets, metadata = self.collect_batch()

                if data == "DONE":
                    self.end_thread("DONE")
                    return
                elif data == "ERROR":
                    self.end_thread("ERROR")
                    return

                # Run detection
                t1 = time.perf_counter()
                detection_results = self.run_detection(data)
                dt = (time.perf_counter() - t1) / len(data)
                self.logger.info(f"Detection time: {dt:.3f}s")

                for i, results in detection_results.items():
                    results_package = {
                        "detections": results,
                        "offset_info": offsets[i],
                        "metadata": metadata[i],
                        "detection_time": dt,
                    }
                self.shared_buffers.put(detections=results_package)

            except Exception as e:
                traceback.print_exc()
                self.end_thread("ERROR")
                return


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
        """
        Initialize the PostProcessingThread.

        Args:
            shared_buffers (SharedBuffers): Shared buffer object for communication.
            config (PredictionConfig): Prediction configuration.
            label_map (dict, optional): Label map for detections.
        """
        super().__init__(name="PostProcessingThread")
        self.shared_buffers = shared_buffers
        self.outputs = None
        self.logger = logging.getLogger(self.name)
        self.config = config
        self.label_map = label_map or dict()
        self.count = 0
        self.roi_processor = None

    def end_thread(self, msg):
        """
        End the thread gracefully by putting a shutdown message in the shared buffer.

        Args:
            msg (str): Shutdown message to put in the buffer.
        """
        self.shared_buffers.put(filtered_detections=msg)
        self.logger.info(f"Thread ended with message: {msg}")

    def set_processor(self, roi_processor: DetectionsPostprocessor):
        """
        Set the ROI processor for post-processing detections.

        Args:
            roi_processor (DetectionsPostprocessor): ROI processor to use.
        """
        self.roi_processor = roi_processor

    def postprocess(
        self, detections: List[Detection], tile: Tile, offset_info: dict
    ) -> List[Detection]:
        """
        Post-process detection results, update tile information, and apply ROI processor if available.

        Args:
            detections (List[Detection]): List of detection results.
            tile (Tile): Tile object associated with detections.
            offset_info (dict): Offset information for detections.

        Returns:
            List[Detection]: Post-processed detections.
        """
        # offset detections
        for i, pred in enumerate(detections):
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
        tile.update_detection_gps()

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
        """
        Run a single post-processing step on detection results from the buffer.

        Returns:
            str or None: "DONE" if no more data, otherwise None.
        """
        # Get detection results from buffer
        results_package = self.shared_buffers.get(detections=True)

        # self.logger.warning(results_package)

        if results_package == "DONE":
            return "DONE"

        elif results_package == "ERROR":
            return "ERROR"

        if not isinstance(results_package, dict):
            self.logger.error("results_package is not a dict")
            return "ERROR"

        # initialize output buffer
        if self.outputs is None:
            self.outputs = []

        try:
            t1 = time.perf_counter()

            # Apply post-processing
            raw_detections = results_package["detections"]
            tile = results_package["metadata"]["tile"]
            offset_info = results_package["offset_info"]

            filtered_detections = self.postprocess(
                raw_detections, tile=tile, offset_info=offset_info
            )

            self.logger.info(
                f"{len(filtered_detections)} detections after postprocessing"
            )

            t2 = time.perf_counter() - t1
            self.logger.debug(f"Postprocessing took: {t2:.3f}s")
            results_package["final_detections"] = filtered_detections
            results_package["postprocess_time"] = t2
            self.outputs.append(results_package)
            self.shared_buffers.put(filtered_detections=results_package)
            return "OK"

        except:
            traceback.print_exc()
            self.end_thread("ERROR")
            return

    def run(self):
        self.logger.info("Starting...")
        while True:
            try:
                status = self._run_once()
                self.logger.debug(f"PostProcessingThread status: {status}")
                if status == "OK":
                    continue
                else:
                    self.end_thread(status)
                    return
            except Exception as e:
                traceback.print_exc()
                self.end_thread("ERROR")
                return


# TODO
class DetectionUploader(threading.Thread):
    """
    Thread responsible for uploading processed detections to the database.
    """

    def __init__(
        self,
        shared_buffers,
        sqlite_path: str,
    ):
        """
        Initialize the DetectionUploader thread for uploading detections to a database.

        Args:
            shared_buffers: Shared buffer object for communication.
            sqlite_path (str): Path to the SQLite database file.
        """
        super().__init__(name="DetectionUploader")
        self.shared_buffers = shared_buffers
        self.logger = logging.getLogger(self.name)

        # Initialize DB connection
        self.conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        """
        Create the wildlife_detections table in the database if it does not exist.
        """
        cur = self.conn.cursor()
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
            timestamp TEXT
        );
        """)
        self.conn.commit()

    def _detection_to_dict(self, det: Detection):
        """
        Convert a Detection object to a dictionary suitable for database insertion.

        Args:
            det (Detection): Detection object to convert.

        Returns:
            dict: Dictionary representation of the detection.
        """
        assert isinstance(det, Detection)
        lat, long, alt = (0.0, 0.0, 0.0)
        if hasattr(det, "gps_as_decimals"):
            lat, long, alt = det.gps_as_decimals
        return {
            "detection_id": uuid.uuid4(),
            "species": getattr(det, "class_name", None),
            "latitude": round(lat, 8),
            "longitude": round(long, 8),
            "altitude": alt,
            "confidence": round(getattr(det, "score", 0.0), 3),
            "image_gps": getattr(det, "image_gps_loc", None),
            "source_image": str(getattr(det, "parent_image", "")),
        }

    def _upload_detections(self, detections):
        """
        Upload a list of detections to the database.

        Args:
            detections (list): List of Detection objects to upload.
        """
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
        """
        Main thread execution loop for uploading detections to the database.
        """
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
        """
        Initialize the ObjectDetectionSystem coordinator.

        Args:
            config (PredictionConfig): Prediction configuration.
            buffer_size (int): Buffer size for shared buffers.
            timeout (int): Timeout for shared buffers.
            detection_label_map (dict, optional): Label map for detections.
        """
        # Initialize shared buffers
        self.shared_buffers = SharedBuffers(max_size=buffer_size, timeout=timeout)

        # Initialize threads
        self.data_thread: DataLoadingThread = None
        self.detection_thread: DetectionThread = None
        self.postprocess_thread: PostProcessingThread = None
        # self.detection_uploader: DetectionUploader = None
        self.label_map = detection_label_map
        self.config = config
        self._detection_model = None
        self._detection_task = None
        self._detection_model_path = None
        self._roi_processor = None

        self.logger = logging.getLogger("ObjectDetectionSystem")

    def set_processor(self, roi_processor: DetectionsPostprocessor):
        """
        Set the ROI processor for the detection system.

        Args:
            roi_processor (DetectionsPostprocessor): ROI processor to use.
        """
        self._roi_processor = roi_processor

    def set_model(self, model: Detector):
        """
        Set the detection model for the detection system.

        Args:
            model (Detector): Detection model to use.
        """
        self._detection_model = model

    def _set_handlers(
        self,
    ):
        """
        Set up handlers for detection and post-processing threads.
        """
        self.detection_thread.set_model(
            model=self._detection_model,
        )

        self.postprocess_thread.set_processor(roi_processor=self._roi_processor)

    @property
    def outputs(
        self,
    ):
        """
        Get the outputs from the post-processing thread if it is not alive.

        Returns:
            list or None: List of outputs or None if the thread is still running.
        """
        if not self.postprocess_thread.is_alive():
            return self.postprocess_thread.outputs

        return None

    def _is_alive(self):
        """
        Check if the object detection system is alive.
        """
        running_threads = {
            "data_thread": self.data_thread.is_alive(),
            "detection_thread": self.detection_thread.is_alive(),
            "postprocess_thread": self.postprocess_thread.is_alive(),
        }
        return running_threads

    def _process_pipeline(self):
        """
        Start all threads in the object detection system.
        """
        self.logger.info("Starting Object Detection System")

        self._set_handlers()

        # self.data_thread.daemon = True
        # self.detection_thread.daemon = True
        # self.postprocess_thread.daemon = True

        # Start threads in order
        self.data_thread.start()
        self.detection_thread.start()
        self.postprocess_thread.start()
        # self.detection_uploader.start()

        self._join_workers()

    def _join_workers(self):
        """
        Stop all threads gracefully and wait for them to finish.
        """

        # Signal shutdown
        # self.shared_buffers.shutdown()

        # Wait for threads to finish
        self.data_thread.join()
        self.detection_thread.join()
        self.postprocess_thread.join()
        # self.detection_uploader.join()

        # self.logger.info("All threads stopped.")
        return None

    def run(
        self, images_paths: Sequence[str], img_loading_batch: int = 4
    ) -> List[List[Detection]]:
        """
        Run the object detection system on a list of image paths.

        Args:
            images_paths (Sequence[str]): List of image paths to process.
            img_loading_batch (int): Batch size for image loading.

        Returns:
            List[List[Detection]]: List of detection results for each image.
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

        # self.detection_uploader = DetectionUploader()

        self._process_pipeline()

        out = self.outputs

        if sum(self._is_alive().values()):
            print(self._is_alive())

        if out is None:
            print("Pipeline failed. Postprocessing did not start...")
            return []

        detections = [o["final_detections"] for o in out]

        return detections
