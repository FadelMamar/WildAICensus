import logging
from abc import abstractmethod, ABC
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
import requests, base64


from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics.engine.results import Results as UltralyticsResults
from typing import Sequence
import lightning as L
import yaml
from animaloc.eval import HerdNetEvaluator, PointsMetrics
from animaloc.eval.lmds import HerdNetLMDS
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC
from torchvision import models

from ..common.base import Detection
from ..common.config import PredictionConfig


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
        """
        Initialize the HerdnetTrainer LightningModule.

        Args:
            data_config_yaml (str): Path to data config YAML file.
            lr (float): Learning rate.
            model (torch.nn.Module): Model to train.
            weight_decay (float): Weight decay for optimizer.
            work_dir (str): Working directory for outputs.
            eval_radius (int, optional): Evaluation radius for metrics.
            classification_threshold (float, optional): Threshold for classification.
            epochs (int, optional): Number of epochs.
            lrf (float, optional): Learning rate factor.
        """
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
        """
        Feed batch metrics to the provided metric object.

        Args:
            metric (PointsMetrics): Metrics object to update.
            batchsize (int): Batch size.
            output (dict): Output dictionary with predictions and ground truth.
        """
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
        """
        Prepare ground truth and predictions for feeding into metrics.

        Args:
            targets (dict): Ground truth targets.
            output (list): Model predictions.

        Returns:
            dict: Dictionary with ground truth and predictions.
        """
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
        """
        Shared step for training, validation, and test.

        Args:
            stage (str): Stage ('train', 'val', or 'test').
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            Loss or None depending on stage.
        """
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
        """
        Log metrics for the given stage.

        Args:
            stage (str): Stage ('val' or 'test').
        """
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
        """
        Called at the end of validation epoch to log metrics.
        """
        self.log_metrics(stage="val")

    def on_test_epoch_end(
        self,
    ):
        """
        Called at the end of test epoch to log metrics.
        """
        self.log_metrics(stage="test")

    def on_validation_epoch_start(
        self,
    ):
        """
        Called at the start of validation epoch to flush metrics.
        """
        self.metrics["val"].flush()

    def on_test_epoch_start(
        self,
    ):
        """
        Called at the start of test epoch to flush metrics.
        """
        self.metrics["test"].flush()

    def training_step(self, batch, batch_idx):
        """
        Training step for LightningModule.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        loss = self.shared_step("train", batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for LightningModule.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        loss = self.shared_step("val", batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for LightningModule.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        loss = self.shared_step("test", batch, batch_idx)
        return loss

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for LightningModule.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            Output dictionary.
        """
        images = batch
        predictions, _ = self.model(images)

        # compute metrics
        output = self.prepare_feeding(targets=None, output=predictions)
        output.pop("gt")  # empty
        return output

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers for LightningModule.

        Returns:
            Tuple of optimizer and scheduler.
        """
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


def get_image_classifier_module(
    num_classes: int, cls_is_features: bool = False
) -> torch.nn.Module:
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

        self.save_hyperparameters(ignore=["threshold"])

        # use a pretrained ResNet backbone
        self.model = get_image_classifier_module(
            cls_is_features=cls_is_features, num_classes=num_classes
        )

        self.label_to_class_map = dict()

        # metrics
        cfg = dict(task="multiclass", num_classes=num_classes, average=None)
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

    def set_label_class_map(self, class_to_label_map: dict):
        assert isinstance(class_to_label_map, dict), "Provide a dict[str,int]"
        self.label_to_class_map = {v: k for k, v in class_to_label_map.items()}

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
            # self.log(f"val_{name}_{i}", metric, prog_bar=True, on_epoch=True)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # print(self.label_to_class_map)

        for name, metric in self.metrics.items():
            score = metric.compute().cpu()
            self.log(f"val_{name}", score.mean())
            for i, score in enumerate(score):
                cls_name = self.label_to_class_map.get(i, i)
                self.log(f"val_{name}_class_{cls_name}", score)

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


class Detector(ABC):
    def __init__(self, config: PredictionConfig):
        self.config = config

    def warmup(self, image_size=(640, 640), **kwargs):
        try:
            self.predict(Image.new("RGB", image_size), **kwargs)
        except:
            logger.info(
                "Failed to warmup the model, please check the model and image size."
            )

    # def set_prediction_config(self, config: PredictionConfig):
    #     assert isinstance(config, PredictionConfig), (
    #         "config must be an instance of PredictionConfig"
    #     )
    #     self.config = config

    def load_image_and_resize(
        self, image: Image.Image | torch.Tensor, target_size: int = 640
    ) -> torch.Tensor:
        assert isinstance(image, Image.Image) or isinstance(image, torch.Tensor), (
            f"Received unexpected type:{type(image)}"
        )
        assert isinstance(target_size, int), (
            f"received {target_size} of type {type(target_size)}"
        )
        transform = T.Resize(
            (target_size, target_size),
            interpolation=T.InterpolationMode.NEAREST,
        )

        if isinstance(image, Image.Image):
            transform = T.Compose([T.PILToTensor(), transform])

        image_tensor = transform(image).float()

        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = (
            image_tensor / 255.0 if image_tensor.max() > 1.0 else image_tensor
        )
        return image_tensor

    @abstractmethod
    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocesses the input image for prediction.

        Args:
            image (Image.Image): The input image to preprocess.

        Returns:
            Image.Image: The preprocessed image.
        """
        pass

    @abstractmethod
    def postprocess(self, detections: list[Detection]) -> list[Detection]:
        """
        Postprocesses the detections.

        Args:
            detections (list[Detection]): The raw detections to postprocess.

        Returns:
            list[Detection]: The postprocessed detections.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def predict(self, image: Image.Image, **kwargs) -> list[Detection]:
        """
        Predicts detections in the given image.

        Args:
            image (Image.Image): The input image to predict on.

        Returns:
            list[Detection]: A list of detected objects.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @staticmethod
    def predict_url(
        image: torch.Tensor,
        inference_service_url: str,
        timeout: int = 360,
        nms_iou: float = 0.5,
        confidence_threshold: float = 0.25,
    ) -> list[dict]:
        assert isinstance(image, torch.Tensor), "Image must be a torch.Tensor"
        assert len(image.shape) == 4, "Image tensor must be B,C,H,W"

        as_bytes = image.cpu().numpy().tobytes()
        payload = {
            "tensor": base64.b64encode(as_bytes).decode("utf-8"),
            "shape": list(image.shape),
            "iou_nms": nms_iou,
            "conf": confidence_threshold,
        }

        res = requests.post(
            url=inference_service_url, json=payload, timeout=timeout
        ).json()

        res = res.get("detections", "FAILED")

        if res == "FAILED":
            logger.error("Failed to get predictions from the inference service.")
            return None

        return res


def build_detector(
    detection_model_type: str,
    model_path: str,
    config: PredictionConfig,
    model=None,
    text_instruction: str = "detect wildlife species",
) -> Detector:
    if detection_model_type == "ultralytics":
        return UltralyticsDetector(
            model_path=model_path, config=config, task="detect", model=model
        )

    elif detection_model_type == "hf-groundingdino":
        return GroundingDinoDetector(
            model_path=model_path, config=config, instruction=text_instruction
        )

    else:
        raise NotImplementedError(f"Detector type '{detection_model_type}' is not implemented.")


class UltralyticsDetector(Detector):
    def __init__(
        self,
        model_path: str,
        config: PredictionConfig,
        task: str = "detect",
        model: YOLO = None,
    ):
        """
        Initializes the UltralyticsDetector with a YOLO model.

        Args:
            model_path (str): Path to the YOLO model file.
            device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        """

        super().__init__(config=config)

        assert sum([model_path is None, model is None]) == 1, (
            "Provide either 'model_path' or 'model'"
        )
        if model_path:
            self.model = YOLO(model_path, task=task)
        else:
            self.model = model

    def preprocess(
        self, image: Image.Image | torch.Tensor, target_size: int = 640
    ) -> Image.Image:
        """
        Preprocesses the input image for prediction.
        """
        return self.load_image_and_resize(image, target_size=target_size)

    def postprocess(self, detections: list[UltralyticsResults]) -> list[Detection]:
        """
        Postprocesses the detections.
        """

        out = []

        for result in detections:
            if result.obb is not None:
                boxes = result.obb
            else:
                boxes = result.boxes

            o = dict(
                bbox=boxes.xyxy.cpu().tolist(),
                label=boxes.cls.cpu().flatten().tolist(),
                score=boxes.conf.cpu().flatten().tolist(),
            )

            for i in range(len(o["bbox"])):
                xmin, ymin, xmax, ymax = o["bbox"][i]
                label = o["label"][i]
                score = o["score"][i]
                out.append(
                    Detection(
                        x_min=xmin,
                        y_min=ymin,
                        x_max=xmax,
                        y_max=ymax,
                        label=label,
                        class_name=result.names.get(label),
                        score=score,
                        parent_image=None,
                    )
                )

        return out

    def predict(
        self,
        image: Image.Image | torch.Tensor,
    ) -> list[Detection]:
        """
        Predicts detections in the given image.

        Args:
            image (Image.Image): The input image to predict on.

        Returns:
            list[Detection]: A list of detected objects.
        """

        assert isinstance(image, Image.Image) or isinstance(image, torch.Tensor), (
            f"Received unexpected type:{type(image)}"
        )

        if isinstance(image, Image.Image):
            assert image.mode == "RGB", "Image must be in RGB mode"

            image = self.preprocess(
                image, target_size=[self.config.imgsz, self.config.imgsz]
            )

        results = self.model.predict(
            image,
            device=self.config.device,
            imgsz=self.config.imgsz,
            conf=self.config.confidence_threshold,
            verbose=self.config.verbose,
            max_det=300,
        )

        return self.postprocess(results)


class GroundingDinoDetector(Detector):
    def __init__(
        self,
        config: PredictionConfig,
        model_path: str = "IDEA-Research/grounding-dino-tiny",
        instruction: str = "detect wildlife species",
        model=None,
    ):
        """
        Initializes the GroundingDinoDetector with a YOLO model.
        """
        super().__init__(config=config)

        if model_path is None:
            raise NotImplementedError
        else:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path)

        self.transform = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.model = self.model.eval().to(self.config.device)

        self.instruction = instruction

    def preprocess(
        self, image: Image.Image | torch.Tensor, text: str, target_size: int = 640
    ) -> Image.Image:
        """
        Preprocesses the input image for prediction.
        """
        assert isinstance(image, Image.Image) or isinstance(image, torch.Tensor), (
            f"Received type:{type(image)}. Expected Image.Image or torch.Tensor"
        )
        assert isinstance(text, str), f"Received type {type(image)}"

        image = self.load_image_and_resize(image, target_size=target_size)
        text = [[text]] * image.shape[0]

        inputs = self.transform(
            images=image, text=text, return_tensors="pt", do_rescale=False
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        return inputs

    def postprocess(
        self,
        detections: list,
        ids: torch.LongTensor,
        target_sizes: list,
        box_threshold: float = 0.4,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        results = self.transform.post_process_grounded_object_detection(
            detections,
            ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )

        out = []

        for result in results:
            o = dict(
                bbox=result["boxes"].cpu().tolist(),
                label=result["labels"].cpu().tolist(),
                score=result["scores"].cpu().tolist(),
                class_name=result["text_labels"],
            )

            for i in range(len(o["bbox"])):
                xmin, ymin, xmax, ymax = o["bbox"][i]
                label = o["label"][i]
                score = o["score"][i]
                class_name = o["class_name"][i]
                out.append(
                    Detection(
                        x_min=xmin,
                        y_min=ymin,
                        x_max=xmax,
                        y_max=ymax,
                        label=label,
                        class_name=class_name,
                        score=score,
                        parent_image=None,
                    )
                )

        return out

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image | torch.Tensor,
        text: str = None,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        text = text or self.instruction
        inputs = self.preprocess(image, text, target_size=self.config.imgsz)
        results = self.model(**inputs)
        batchsize = inputs["pixel_values"].shape[0]
        target_sizes = [(self.config.imgsz, self.config.imgsz)] * batchsize
        return self.postprocess(
            detections=results,
            ids=inputs["input_ids"],
            target_sizes=target_sizes,
            box_threshold=self.config.confidence_threshold,
            text_threshold=text_threshold,
        )


# TODO: Implement MMDetectionDetector
class MMDetectionDetector(Detector):
    def __init__(self, model_path: str, config: PredictionConfig):
        """
        Initializes the MMDetectionDetector
        """
        super().__init__(config=config)

        self.device = config.device

        self.model = ...

    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocesses the input image for prediction.
        """

        raise NotImplementedError

    def postprocess(self, detections: list) -> list[Detection]:
        """
        Postprocesses the detections.
        """
        raise NotImplementedError

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image | torch.Tensor,
    ) -> list[Detection]:
        """
        Predicts detections in the given image.
        """

        raise NotImplementedError
