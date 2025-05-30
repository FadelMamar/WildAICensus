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
import time
from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset
from torchvision.transforms import PILToTensor
from PIL import Image
import traceback
from ultralytics.engine.results import Results as UltralyticsResults
from tqdm import tqdm
from torchvision.ops import nms

from ..common.config import PredictionConfig
from ..common.base import Tile, Detection
from ..common.annotation_utils import GPSUtils, compute_detection_gps
from ..common.processor import DetectionsPostprocessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class SharedBuffers:
    """
    Shared memory buffers for inter-thread communication

    GUIDELINES:
    - Adjust queue sizes based on your system's memory and processing speed
    - Consider using collections.deque with maxlen for memory-bounded buffers
    - Add more buffers if needed for your specific use case
    """

    def __init__(self, max_size: int = 64, timeout: int = 15):
        super(SharedBuffers, self)  # .__init__(name="SharedBuffers")
        # Buffer between Data Loading -> Detection
        self.raw_data_buffer = Queue(maxsize=max_size)

        # Buffer between Detection -> Post-processing
        self.detection_results_buffer = Queue(maxsize=max_size)

        # Optional: Final results buffer for output
        self.final_results_buffer = Queue(maxsize=max_size)

        # Thread synchronization
        self._shutdown_event = threading.Event()

        self.logger = logging.getLogger("SharedBuffers")

        self.queue_kwargs = dict(block=True, timeout=timeout)

        self.is_closed = self._shutdown_event.is_set()

    def put(
        self, data: Any = None, detections: Any = None, filtered_detections: Any = None
    ):
        kwargs = self.queue_kwargs
        try:
            if data:
                self._put_data(data, **kwargs)

            if detections:
                self._put_detections(detections, **kwargs)

            if filtered_detections:
                self._put_results(filtered_detections, **kwargs)

        except queue.Full:
            self.logger.error("Queue is Full and timeout was exceeded.")
            raise ValueError("Increase timeout?")

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
                return self._get_data(**kwargs)

            if detections:
                return self._get_detections(**kwargs)

            if filtered_detections:
                return self._get_results(**kwargs)

        except queue.Empty:
            self.logger.info("Queue is Empty or has been consumed.")
            return "empty"

        except Exception:
            self.logger.error("Un-catched error!")
            traceback.print_exc()
            raise ValueError()

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

    def __init__(self, shared_buffers: SharedBuffers, data_source: Sequence[str]):
        super(DataLoadingThread, self).__init__(name="DataLoadingThread")

        self.shared_buffers = shared_buffers
        self.logger = logging.getLogger(self.name)

        assert isinstance(data_source, Sequence)
        self._source_iterator = iter(data_source)

        self.tile_size = 800
        self.overlap_ratio = 0.2
        self.stride = int((1 - self.overlap_ratio) * self.tile_size)
        self.count = 0

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

    def load_data(
        self,
    ) -> Tile | str:
        """
        Load and preprocess data from your source

        Returns:
            Tuple of (processed_data, metadata) or None if no more data

        TODO: Implement your data loading logic here
        Examples:
        - For video: return (frame, {"frame_id": count, "timestamp": time.time()})
        - For images: return (image, {"filename": path, "shape": image.shape})
        """
        # PLACEHOLDER - REPLACE WITH YOUR IMPLEMENTATION

        try:
            image_path = next(self._source_iterator)
            tile = Tile(image_path=image_path)
            self.logger.debug(f"Loading: sample {self.count} has been loaded.")

            self.count += 1

            return tile

        except StopIteration:
            return "empty"

        except Exception:
            traceback.print_exc()
            raise ValueError()

    def get_patches_from_tile(
        self, tile: Tile, tile_size: int, stride: int
    ) -> tuple[torch.Tensor, dict]:
        image = tile.load_image_data()
        image = image.convert("RGB")
        image = PILToTensor()(image)

        if tile.width < tile_size or tile.height < tile_size:
            self.logger.debug("image is too small for patch extraction")
            offset_info = {
                "y_offset": [0],
                "x_offset": [0],
                "y_end": tile.height,
                "x_end": tile.width,
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
            "y_end": (y_min + tile_size).tolist(),
            "x_end": (x_min + tile_size).tolist(),
            "file_name": str(tile.image_path),
        }

        return tiles, offset_info

    def preprocess_data(self, tile: Tile) -> Tuple[TensorDataset, Dict]:
        """
        Preprocess raw data before detection

        TODO: Implement preprocessing steps
        Examples:
        - Resize: cv2.resize(image, (model_width, model_height))
        - Normalize: image.astype(np.float32) / 255.0
        - Color conversion: cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        """
        # PLACEHOLDER - REPLACE WITH YOUR PREPROCESSING
        self.logger.debug(f"Preprocessing: sample {self.count} has been loaded.")

        stride = int((1 - self.overlap_ratio) * self.tile_size)

        # load as RGB and extract patches
        batch_of_patches, offset_info = self.get_patches_from_tile(
            tile=tile, tile_size=self.tile_size, stride=stride
        )

        if batch_of_patches.max() > 1.0:
            batch_of_patches = batch_of_patches / 255.0

        if len(batch_of_patches) == 3:
            batch_of_patches.unsqueeze_(0)

        data = TensorDataset(batch_of_patches)

        return data, offset_info

    def run(self):
        """Main thread execution loop"""
        self.logger.info("Starting data loading thread")

        # try:
        while not self.shared_buffers.is_closed:
            # Load data
            tile = self.load_data()
            if tile == "empty":
                self.logger.info("No more data to load")
                break

            # Preprocess data
            data, offset_info = self.preprocess_data(tile=tile)
            # print(data)

            # Package data with metadata
            data_package = {}
            data_package.update(
                {
                    "tile": tile,
                    "data": data,
                    "idx": self.count,
                    "offset_info": offset_info,
                },
            )

            # while self.shared_buffers.is_full_data_queue:
            self.shared_buffers.put(data=data_package)

        # except Exception as e:
        #     self.logger.error(f"Error in data loading thread: {e}")
        #     traceback.print_exc()


class DetectionThread(threading.Thread):
    """
    Thread responsible for running object detection
    """

    def __init__(self, shared_buffers: SharedBuffers, config: PredictionConfig):
        super().__init__(name="DetectionThread")
        self.shared_buffers = shared_buffers
        self.model = None
        self.logger = logging.getLogger(self.name)

        self.count = 0
        self.config = config

    def set_model(self, model: YOLO, path_weights: str, task="detect"):
        if model:
            self.model = model
        else:
            self.model = YOLO(path_weights, task=task)
        self.logger.info("Model loaded successfully")

    def _trim_result(self, results: list[UltralyticsResults]) -> list:
        trimmed = []
        for res in results:
            if res.obb is not None:
                boxes = res.obb
            else:
                boxes = res.boxes
            trimmed.append(boxes)
        return trimmed

    def _pad_if_needed(self, batch: torch.Tensor, out_shape: tuple):
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

    def run_detection(
        self,
        data: Any,
    ) -> Dict:
        """
        Run object detection on input data
        """
        # PLACEHOLDER - REPLACE WITH YOUR DETECTION CODE

        loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=False)

        if self.config.verbose:
            loader = tqdm(loader, desc="sliced_inference")

        results = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = self._pad_if_needed(
                    batch, out_shape=(self.config.batch_size, *batch.shape[1:])
                )
                res = self.model(
                    batch,
                    verbose=False,
                    imgsz=self.config.tilesize,
                    conf=self.config.confidence_threshold,
                    iou=self.config.nms_iou,
                )
                b = batch.shape[0]
                res = res[:b]
                results = results + self._trim_result(res)

        self.count += 1

        return results

    def run(self):
        """Main thread execution loop"""
        self.logger.info("Starting detection thread")

        # Load model
        assert self.model is not None, "Provide base detection model i.e. YOLO"

        # try:
        while True:
            # Get data from buffer
            data_package = self.shared_buffers.get(data=True)
            if data_package == "empty":
                break
            # Run detection
            data = data_package.pop("data")
            t1 = time.perf_counter()
            detection_results = self.run_detection(data)
            t_end = time.perf_counter() - t1

            # Package results with original metadata
            results_package = {
                "detections": detection_results,
                "detection_time": t_end,
            }
            results_package.update(data_package)

            # Put results in buffer
            self.shared_buffers.put(detections=results_package)

        # except Exception as e:
        #     self.logger.error(f"Error in detection thread: {e}")
        #     traceback.print_exc()


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
        self, results: list[UltralyticsResults], tile: Tile, offset_info: dict
    ) -> List[Detection]:
        # ultralytics results to coco
        detections = self._result_to_coco(
            results,
            offset_info=offset_info,
            tile_width=tile.width,
            tile_height=tile.height,
        )

        # Get gps coordinates
        gps_coords = tile.tile_gps_loc
        if gps_coords is None:
            # image gps coordinate
            gps_info = GPSUtils.get_gps_coord(
                file_name=tile.image_path, image=None, return_as_decimal=False
            )
            if isinstance(gps_info, tuple):
                gps_coords = gps_info[0]
            else:
                gps_coords = gps_info

        # format detections
        detections = [
            Detection.from_coco(
                pred,
                parent_image=tile.image_path,
                image_gps_loc=tile.tile_gps_loc,
                gps_loc=None,
            )
            for pred in detections
        ]

        # add detections gps
        for det in detections:
            if det.image_gps_loc is None:
                continue
            with Image.open(tile.image_path) as image:
                det.gps_loc = compute_detection_gps(
                    x_center=det.x,
                    y_center=det.y,
                    image=image,
                    image_gps_loc=det.image_gps_loc,
                    flight_height=self.config.flight_height,
                    sensor_height=self.config.sensor_height,
                    gsd=self.config.gsd,
                )

        # post process roi
        if self.roi_processor:
            detections = self.roi_processor.run(
                detections, image=tile.load_image_data(), box_size=self.config.cls_imgsz
            )

        self.count += 1

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

        assert torch.min(bboxs) >= 0
        assert torch.max(bboxs[:, [0, 2]]) <= tile_width
        assert torch.max(bboxs[:, [1, 3]]) <= tile_height

        # Non-max suppression
        indx = nms(boxes=bboxs, scores=conf, iou_threshold=self.config.nms_iou)

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

    def run(self):
        """Main thread execution loop"""
        self.logger.info("Starting post-processing thread")

        # try:
        while True:
            # Get detection results from buffer
            results_package = self.shared_buffers.get(detections=True)
            if results_package == "empty":
                break

            # Apply post-processing
            detections = results_package["detections"]
            tile = results_package["tile"]
            offset_info = results_package["offset_info"]

            t1 = time.perf_counter()
            detections = self.postprocess(
                detections, tile=tile, offset_info=offset_info
            )
            t2 = time.perf_counter() - t1
            # Filter by confidence
            # detections = self.filter_detections(detections,
            #                                   self.output_config.get("conf_threshold", 0.5))

            # Update results package
            results_package["final_detections"] = detections
            results_package["postprocess_time"] = t2

            # self.shared_buffers.put(filtered_detections=results_package)
            self.outputs.append(results_package)

        # except Exception as e:
        #     self.logger.error(f"Error in post-processing thread: {e}")
        #     traceback.print_exc()


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
        self.data_thread = None
        self.detection_thread = DetectionThread(self.shared_buffers, config=config)
        self.postprocess_thread = PostProcessingThread(
            self.shared_buffers,
            config=config,
            label_map=detection_label_map,
        )

        self.logger = logging.getLogger("ObjectDetectionSystem")

    def set_processor(self, roi_processor: DetectionsPostprocessor):
        self.postprocess_thread.set_processor(roi_processor=roi_processor)

    def set_model(self, model: YOLO, path_weights: str, task: str = "detect"):
        self.detection_thread.set_model(
            model=model, path_weights=path_weights, task=task
        )

    @property
    def outputs(
        self,
    ):
        if not self.postprocess_thread.is_alive():
            return self.postprocess_thread.outputs
        else:
            self.logger.info(
                "Postprocessing thread is still alive. Make sure it has ended first "
            )
            return None

    def _process_pipeline(self):
        """Start all threads"""
        self.logger.info("Starting Object Detection System")

        # Start threads in order
        self.data_thread.start()
        self.detection_thread.start()
        self.postprocess_thread.start()
        self.logger.info("All threads started")

        # wait for threads to finish
        self._join_workers()

    def _join_workers(self):
        """Stop all threads gracefully"""
        # self.logger.info("Stopping Object Detection System")

        # Signal shutdown
        self.shared_buffers.shutdown()

        # Wait for threads to finish
        self.data_thread.join(timeout=5.0)
        self.detection_thread.join(timeout=35.0)
        self.postprocess_thread.join(timeout=5.0)

        self.logger.info("All threads stopped")
        return None

    def run(self, images_paths: Sequence[str]):
        """
        Run the system for a specified duration or until stopped
        """

        # Initialize dataloader
        self.data_thread = DataLoadingThread(
            self.shared_buffers, data_source=images_paths
        )

        self._process_pipeline()


if __name__ == "__main__":
    config = PredictionConfig(
        imgsz=800,
        tilesize=800,
        batch_size=4,
        overlap_ratio=0.2,
        confidence_threshold=0.2,
        inference_service_url=None,
        flight_height=180,
        sensor_height=24,
        gsd=None,
        nms_iou=0.5,
        verbose=True,
        # min_area=100,
        # max_area=None,
        cls_imgsz=128,
        # device="cuda:0",
    )

    image_path = r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images\0d1ba3c424ad4414ac37dbd0c93460ea_1_51_0_1024_640_1664.jpg"
    # image_path = r"D:\savmap_dataset_v2\raw\tmp\0a3ed15cfab4453795564140e8fde8ba.JPG"

    images = [image_path] * 5

    # load Roi processor
    path = r"D:\datalabeling\base_models_weights\roi_classifier.ckpt"  # r"./runs-classifier/best-v2.ckpt"
    model = ImageClassifier.load_from_checkpoint(
        path, cls_is_features=True, map_location=config.device
    )
    roi_classifier = get_processor("classifier")(
        model,
        label_map={0: "gt", 1: "tn"},
        device=config.device,
        feature_extractor=get_processor("feature_extractor")(),
        imgsz=config.cls_imgsz,
    )
    roi_processor = DetectionsPostprocessor(
        keep_classes=["gt"],  # from roi classifier
    )
    roi_processor.set_classifier(roi_classifier)

    detection_system = ObjectDetectionSystem(
        data_source=images,
        config=config,
        buffer_size=32,
        timeout=15,
        detection_label_map={},
        roi_processor=roi_processor,
    )

    detection_system.run()

    outputs = detection_system.outputs

    pass
