from PIL import Image
import numpy as np
import torch


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


from PIL import Image


# TODO
def Postprocessor(object):
    def __init__(self, classifier):
        self.classifier = classifier

    # TODO: implement
    def run(self, detections: list[dict], image: Image.Image):
        pass
