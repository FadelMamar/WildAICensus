from PIL import Image
import numpy as np
import torch
import albumentations as A
from abc import ABC, abstractmethod
from typing import Sequence
import logging
from tqdm import tqdm
from ..common.base import Detection


def get_processor(name: str):
    """Return the processor class based on the given name.
    Args:
        name (str): Name of the processor ('feature_extractor', 'classifier', or 'detections_post').
    Returns:
        type: Corresponding processor class.
    Raises:
        NotImplementedError: If the processor name is not recognized.
    """
    if name == "feature_extractor":
        return FeatureExtractor

    elif name == "classifier":
        return Classifier

    elif name == "detections_post":
        return DetectionsPostprocessor

    else:
        raise NotImplementedError


class Processor(ABC):
    """Abstract base class for all processors."""

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the processor on the given arguments."""
        pass


def check_images_sequences(images: Sequence[Image.Image]):
    """Assert that the input is a sequence of PIL Images."""
    assert isinstance(images, Sequence)
    for a in images:
        assert isinstance(a, Image.Image)


# =============================================================================
# # Image processors
# =============================================================================
class FeatureExtractor(Processor):
    """Feature extractor using a HuggingFace model."""

    def __init__(self, hf_model_path="facebook/dinov2-with-registers-small"):
        """Initialize the feature extractor with a HuggingFace model path.
        Args:
            hf_model_path (str): Path or name of the HuggingFace model.
        """
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(hf_model_path)
        self.extractor = AutoModel.from_pretrained(
            hf_model_path, torch_dtype="auto", device_map="auto"
        )
        self.device = self.extractor.device

    def run(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Extract features from a sequence of images.
        Args:
            images (Sequence[np.ndarray]): List of images as numpy arrays.
        Returns:
            np.ndarray: Extracted features.
        """
        images = [Image.fromarray(image) for image in images]

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.extractor(**inputs)
        features = outputs.pooler_output.cpu().reshape(len(images), -1).numpy()

        return features


class SuperResolution(Processor):
    """Super-resolution processor (not implemented)."""

    def __init__(
        self,
    ):
        """Initialize the super-resolution processor."""
        pass

    def run(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Run super-resolution on a sequence of images (not implemented)."""
        check_images_sequences(images)
        pass


class Classifier(Processor):
    """Image classifier using a PyTorch model."""

    def __init__(
        self,
        model: torch.nn.Module,
        label_map: dict,
        feature_extractor: FeatureExtractor = None,
        imgsz: int = 96,
        transform=None,
        device: str = "cpu",
    ):
        """Initialize the classifier.
        Args:
            model (torch.nn.Module): PyTorch model for classification.
            label_map (dict): Mapping from class indices to class names.
            feature_extractor (FeatureExtractor, optional): Feature extractor to use.
            imgsz (int): Image size for preprocessing.
            transform: Albumentations transform for preprocessing.
            device (str): Device to run the model on.
        """
        self.model = model
        self.label_map = label_map

        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()

        self.feature_extractor: FeatureExtractor = feature_extractor

        self.transform = transform
        if transform is None:
            self.transform = A.Compose(
                [
                    A.PadIfNeeded(min_height=imgsz, min_width=imgsz),
                    # A.CenterCrop(height=imgsz,width=imgsz,pad_if_needed=True)
                ]
            )

    def _pil_to_numpy(self, image: Image.Image):
        """Convert a PIL Image to a numpy array."""
        image = image.convert("RGB")
        image = np.asarray(image)
        return image

    def run(self, images: Sequence[Image.Image]) -> list[str]:
        """Classify a sequence of images.
        Args:
            images (Sequence[Image.Image]): List of PIL Images.
        Returns:
            list[str]: List of predicted class names.
        """
        check_images_sequences(images)

        preprocessed = [
            self.transform(image=self._pil_to_numpy(image))["image"] for image in images
        ]

        if self.feature_extractor:
            preprocessed = self.feature_extractor.run(preprocessed)
            preprocessed = torch.Tensor(preprocessed).to(self.device)

        with torch.no_grad():
            probs = self.model(preprocessed).softmax(1)
            pred = probs.argmax(1).cpu().long().flatten().tolist()

        pred = list(map(lambda x: self.label_map[x], pred))

        return pred


# =============================================================================
# # Detections processors
# =============================================================================
class DetectionsPostprocessor(Processor):
    """Post-process detections using a classifier to filter by class."""

    def __init__(self, keep_classes: list[str] = ["groundtruth"]):
        """Initialize the postprocessor.
        Args:
            keep_classes (list[str]): List of class names to keep.
        """
        self.classifier: Classifier = None
        self.keep = keep_classes
        self.logger = logging.getLogger("DetectionsPostprocessor")

    def set_classifier(self, classifier: Classifier):
        """Set the classifier to use for filtering detections.
        Args:
            classifier (Classifier): Classifier instance.
        """
        self.classifier = classifier

    def run(
        self,
        detections: list[Detection],
        image: Image.Image,
        box_size: int = 96,
        verbose: bool = True,
    ) -> list[Detection]:
        """Filter detections by running a classifier on cropped image regions.
        Args:
            detections (list[Detection]): List of detection objects.
            image (Image.Image): Source image.
            box_size (int): Size of the crop around each detection.
            verbose (bool): Whether to show progress bar.
        Returns:
            list[Detection]: Filtered detections.
        """
        assert isinstance(image, Image.Image)
        assert self.classifier, "Provide a handler using self.set_classifier"

        self.logger.debug("Filtering detections...")

        image = image.convert("RGB")

        if len(detections) < 1:
            return []

        dets = []

        img_width, img_height = image.size

        loader = tqdm(detections, desc="ROI based filtering") if verbose else detections

        for det in loader:
            x_center = det.x
            y_center = det.y

            x1 = int(max(x_center - box_size, 0))
            y1 = int(max(y_center - box_size, 0))

            x2 = int(min(x_center + box_size, img_width))
            y2 = int(min(y_center + box_size, img_height))
            box = (x1, y1, x2, y2)
            dets.append(image.crop(box))

        preds = self.classifier.run(dets)

        out = [det for i, det in enumerate(detections) if preds[i] in self.keep]

        return out
