from PIL import Image
import numpy as np
import torch
import albumentations as A
from abc import ABC, abstractmethod
from typing import Sequence


from .config import Detection


def get_processor(name: str):
    if name == "feature_extractor":
        return FeatureExtractor

    elif name == "classifier":
        return Classifier

    elif name == "detections_post":
        return DetectionsPostprocessor

    else:
        raise NotImplementedError


class Processor(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass


# =============================================================================
# # Image processors
# =============================================================================
class FeatureExtractor(Processor):
    def __init__(self, hf_model_path="facebook/dinov2-with-registers-small"):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(hf_model_path)
        self.extractor = AutoModel.from_pretrained(hf_model_path,torch_dtype="auto", device_map="auto")
        self.device = self.extractor.device

    def run(self, images: Sequence[np.ndarray]) -> np.ndarray:
        
        assert isinstance(images, Sequence)
        for a in images:
            assert isinstance(a,np.ndarray)

        images = [Image.fromarray(image) for image in images]

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.extractor(**inputs)
        features = outputs.pooler_output.cpu().reshape(len(images),-1).numpy()

        return features
            


class SuperResolution(Processor):
    def __init__(
        self,
    ):
        pass

    def run(self, image: np.ndarray) -> np.ndarray:
        pass


class Classifier(Processor):
    def __init__(
        self,
        model: torch.nn.Module,
        label_map: dict,
        feature_extractor=None,
        imgsz: int = 96,
        transform=None,
        device: str = "cpu",
    ):
        self.model = model
        self.label_map = label_map

        self.device = device
        self.model = self.model.to(self.device)

        self.feature_extractor = feature_extractor

        self.transform = transform
        if transform is None:
            self.transform = A.Compose(
                [
                    A.PadIfNeeded(min_height=imgsz, min_width=imgsz),
                    # A.CenterCrop(height=imgsz,width=imgsz,pad_if_needed=True)
                ]
            )

    def run(self, images: list[np.ndarray]) -> list[str]:
        preprocessed = [self.transform(image=image)["image"] for image in images]

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
    def __init__(self, keep_classes: list[str] = ["groundtruth"]):
        self.handler = None
        self.keep = keep_classes

    def set_handler(self, handler):
        self.handler = handler

    # TODO: implement
    def run(
        self,
        detections: list[Detection],
        image: Image.Image,
        box_size: int = 96,
    ) -> list[Detection]:
        assert isinstance(image, Image.Image)
        assert self.handler, "Provide a handler using self.set_handler"

        if len(detections) < 1:
            return []

        dets = []

        image = image.convert("RGB")
        image = np.asarray(image)

        for det in detections:
            x_center = det.x
            y_center = det.y

            x1 = int(max(x_center - box_size, 0))
            y1 = int(max(y_center - box_size, 0))

            x2 = int(min(x_center + box_size, image.shape[1]))
            y2 = int(min(y_center + box_size, image.shape[0]))

            dets.append(image[y1:y2, x1:x2])

        preds = self.handler.run(dets)

        out = [det for i, det in enumerate(detections) if preds[i] in self.keep]

        return out
