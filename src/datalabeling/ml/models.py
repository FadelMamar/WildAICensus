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
from typing import Sequence
import lightning as L
import yaml
from animaloc.eval import HerdNetEvaluator, PointsMetrics
from animaloc.eval.lmds import HerdNetLMDS

from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC
from torchvision import models

from ..common.annotation_utils import GPSUtils, ImageProcessor
from ..common.config import Detection, PredictionConfig



logger = logging.getLogger(__name__)

class HerdnetTrainer(L.LightningModule):
    def __init__(
        self,
        data_config_yaml: str,
        lr: float,
        model: torch.nn.Module,
        weight_decay: float,
        work_dir: str,
        eval_radius: int = 20,
        classification_threshold: float = 0.25,
        epochs: int = None,
        lrf: float = 1e-1,
    ):
        super().__init__()

        self.save_hyperparameters(
            "lr",
            "weight_decay",
            "data_config_yaml",
            "eval_radius",
            "lrf",
            "epochs",
            ignore=["model", "work_dir", "classification_threshold"],
        )

        self.work_dir = work_dir
        self.classification_threshold = classification_threshold

        # Get number of classes
        with open(data_config_yaml, "r") as file:
            data_config = yaml.load(file, Loader=yaml.FullLoader)
            # including a class for background
            self.num_classes = data_config["nc"] + 1

        self.class_mapping = {str(k + 1): v for k, v in data_config["names"].items()}

        self.model = model

        # metrics
        self.metrics_val = PointsMetrics(
            radius=eval_radius, num_classes=self.num_classes
        )
        self.metrics_test = PointsMetrics(
            radius=eval_radius, num_classes=self.num_classes
        )

        self.metrics = {"val": self.metrics_val, "test": self.metrics_test}

        self.stitcher = None

        self.herdnet_evaluator = HerdNetEvaluator(
            model=self.model,
            dataloader=DataLoader(dataset=[None, None], batch_size=1),
            metrics=self.metrics_val,
            stitcher=self.stitcher,
            work_dir=self.work_dir,
            header="validation",
            lmds_kwargs={"kernel_size": (3, 3), "adapt_ts": 3.0, "neg_ts": 0.1},
        )
        up = True
        if self.stitcher is not None:
            up = False
        self.lmds = HerdNetLMDS(up=up, **self.herdnet_evaluator.lmds_kwargs)

    def batch_metrics(
        self, metric: PointsMetrics, batchsize: int, output: dict
    ) -> None:
        if batchsize >= 1:
            for i in range(batchsize):
                gt = {k: v[i] for k, v in output["gt"].items()}
                preds = {k: v[i] for k, v in output["preds"].items()}
                counts = output["est_count"][i]
                output_i = dict(gt=gt, preds=preds, est_count=counts)
                metric.feed(**output_i)
        else:
            raise NotImplementedError

    def prepare_feeding(
        self, targets: dict[str, torch.Tensor], output: list[torch.Tensor]
    ) -> dict:
        try:  # batchsize==1
            gt_coords = [p[::-1] for p in targets["points"].cpu().tolist()]
            gt_labels = targets["labels"].cpu().tolist()
        except Exception:  # batchsize>1
            gt_coords = [p[::-1] for p in targets["points"]]
            gt_labels = targets["labels"]

        # get predictions
        counts, locs, labels, scores, dscores = self.lmds(output)
        gt = dict(loc=gt_coords, labels=gt_labels)
        preds = dict(loc=locs, labels=labels, scores=scores, dscores=dscores)

        return dict(gt=gt, preds=preds, est_count=counts)

    def shared_step(self, stage, batch, batch_idx):
        # compute losses
        if stage == "train":
            images, targets = batch
            predictions, loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            self.log_dict(loss_dict)
            return loss.clamp(-5.0, 5.0)  # preventing exploding gradient

        else:
            images, targets = batch
            batchsize = images.shape[0]
            assert batchsize >= 1 and len(images.shape) == 4, (
                "Input image does not have the right shape > e.g. [b,c,h,w]"
            )
            predictions, _ = self.model(images)
            # compute metrics
            output = self.prepare_feeding(targets=targets, output=predictions)
            iter_metrics = self.metrics[stage]
            self.batch_metrics(metric=iter_metrics, batchsize=batchsize, output=output)
            return None

    def log_metrics(self, stage: str):
        assert stage != "train", "metrics only logged for val and test."

        iter_metrics = self.metrics[stage]

        # store for class level metrics computation
        self.herdnet_evaluator._stored_metrics = iter_metrics.copy()

        # aggregate results
        iter_metrics.aggregate()
        self.log(f"{stage}_recall", round(iter_metrics.recall(), 3))
        self.log(f"{stage}_precision", round(iter_metrics.precision(), 3))
        self.log(f"{stage}_f1-score", round(iter_metrics.fbeta_score(), 3))
        self.log(f"{stage}_MAE", round(iter_metrics.mae(), 3))
        self.log(f"{stage}_MSE", round(iter_metrics.mse(), 3))
        self.log(f"{stage}_RMSE", round(iter_metrics.rmse(), 3))

        # log perclass metrics
        per_class_metrics = self.herdnet_evaluator.results
        metrics_cols = [
            p
            for p in per_class_metrics.columns
            if p
            not in [
                "class",
            ]
        ]
        for _, row in per_class_metrics.iterrows():
            for col in metrics_cols:
                label = str(row.loc["class"])
                if label in self.class_mapping.keys():
                    class_name = self.class_mapping[label]
                    name = f"{class_name}_{col}"
                else:
                    name = label
                self.log(name, round(row.loc[col], 3))

    def on_validation_epoch_end(
        self,
    ):
        self.log_metrics(stage="val")

    def on_test_epoch_end(
        self,
    ):
        self.log_metrics(stage="test")

    def on_validation_epoch_start(
        self,
    ):
        self.metrics["val"].flush()

    def on_test_epoch_start(
        self,
    ):
        self.metrics["test"].flush()

    def training_step(self, batch, batch_idx):
        loss = self.shared_step("train", batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step("val", batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step("test", batch, batch_idx)
        return loss

    def predict_step(self, batch, batch_idx):
        images = batch
        predictions, _ = self.model(images)

        # compute metrics
        output = self.prepare_feeding(targets=None, output=predictions)
        output.pop("gt")  # empty
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.epochs,
            T_mult=1,
            eta_min=self.hparams.lr * self.hparams.lrf,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]


class ImageClassifier(L.LightningModule):
    def __init__(
        self,
        cls_is_features: bool,
        epochs: int = 50,
        num_classes: int = 2,
        threshold: float = 0.5,
        label_smoothing: float = 0.0,
        lr: float = 1e-3,
        lrf: float = 1e-2,
        weight_decay: float = 5e-3,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "threshold"])

        # use a pretrained ResNet backbone
        self.model = get_image_classifier_module(
            cls_is_features=cls_is_features, num_classes=num_classes
        )
        # replace final layer
        # in_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(in_features, num_classes)

        # metrics
        cfg = dict(task="multiclass", num_classes=num_classes, average="macro")
        self.accuracy = Accuracy(**cfg)
        self.precision = Precision(threshold=threshold, **cfg)
        self.recall = Recall(threshold=threshold, **cfg)
        self.f1score = F1Score(threshold=threshold, **cfg)
        self.ap = AUROC(**cfg)

        self.metrics = dict(
            accuracy=self.accuracy,
            precision=self.precision,
            recall=self.recall,
            f1score=self.f1score,
        )

        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, x) -> torch.Tensor:
        out = self.model(x)

        if isinstance(out, Sequence):
            logits = out[1]  # yolo cls
        else:
            logits = out

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        classes = y.cpu().flatten().tolist()
        weight = [
            len(classes) / (classes.count(i) + 1e-6) for i in range(self.num_classes)
        ]
        weight = torch.Tensor(weight).float().clamp(1.0, 1e2).to(y.device)

        logits = self(x)
        loss = F.cross_entropy(
            logits,
            y.long().squeeze(1),
            label_smoothing=self.label_smoothing,
            weight=weight,
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long().squeeze(1)

        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        for name, metric in self.metrics.items():
            metric.update(logits, y)
            self.log(f"val_{name}", metric, prog_bar=True, on_epoch=True)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def predict_step(self, x):
        with torch.no_grad():
            probs = self.forward(x).softmax(dim=1)
            pred = probs.argmax(1)

        return probs, pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.epochs,
            T_mult=1,
            eta_min=self.hparams.lr * self.hparams.lrf,
        )
        return [optimizer], [lr_scheduler]


def get_image_classifier_module(num_classes: int, cls_is_features: bool = False):
    if cls_is_features:
        model = torch.nn.Sequential(
            torch.nn.LazyLinear(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.LazyLinear(128),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(num_classes),
        )
    else:
        model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        model.classifier = torch.nn.LazyLinear(num_classes)

    return model

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
            self.detection_model = UltralyticsDetectionModel(
                model=YOLO(path_to_weights, task="detect"),
                confidence_threshold=self.config.confidence_threshold,
                image_size=self.config.imgsz,
                device=self.config.device,
            )
            logger.info(f"Computing device: {self.config.device}")

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
            if len(detections) < 1:
                continue

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
