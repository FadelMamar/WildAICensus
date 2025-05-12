from PIL import Image
import numpy as np
import torch
import albumentations as A


class FeatureExtractor:
    def __init__(self, hf_model_path="facebook/dinov2-with-registers-small"):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(hf_model_path)
        self.extractor = AutoModel.from_pretrained(hf_model_path)

    def get_features(self, image: np.ndarray) -> np.ndarray:
        assert isinstance(image, np.ndarray)

        image = Image.fromarray(image)

        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.extractor(**inputs)
        features = outputs.pooler_output.cpu().numpy().flatten()

        return features


class Classifier(object):
    def __init__(
        self,
        model: torch.nn.Module,
        label_map: dict,
        feature_extractor: FeatureExtractor = None,
        imgsz: int = 96,
        device: str = "cpu",
    ):
        self.model = model
        self.label_map = label_map

        self.device = device
        self.model = self.model.to(self.device)

        self.feature_extractor = feature_extractor

        self.transform = A.Compose(
            [
                A.PadIfNeeded(min_height=imgsz, min_width=imgsz),
                # A.CenterCrop(height=imgsz,width=imgsz,pad_if_needed=True)
            ]
        )

    def run(self, images: list[np.ndarray]):
        preprocessed = [self.transform(image=image)["image"] for image in images]

        if self.feature_extractor:
            preprocessed = [
                self.feature_extractor.get_features(image) for image in preprocessed
            ]
            preprocessed = torch.Tensor(np.stack(preprocessed)).to(self.device)

        with torch.no_grad():
            probs = self.model(preprocessed).softmax(1)
            pred = probs.argmax(1).cpu().long().flatten().to_list()

        pred = list(map(lambda x: self.label_map[x], pred))

        return pred


# TODO
class DetectionsPostprocessor(object):
    def __init__(self, classifier: Classifier, keep_classes: str = ["groundtruth"]):
        self.classifier = classifier
        self.keep = keep_classes

    # TODO: implement
    def run(self, detections: list[dict], image: np.ndarray) -> list[dict]:
        dets = []

        for det in detections:
            x1 = det["x_min"]
            y1 = det["y_min"]
            x2 = det["x_max"]
            y2 = det["y_max"]

            dets.append(image[y1:y2, x1:x2])

        preds = self.classifier.run(dets)

        out = [det for i, det in enumerate(detections) if preds[i] in self.keep]

        return out
