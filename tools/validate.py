import json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from datargs import parse
from ultralytics.data import converter
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix


@dataclass
class Flags:
    # data_config file
    data_config: str = None

    # split
    splits: Sequence[str] = ("val",)

    # weights
    weights: str = None

    # inference
    batch_size: int = 32
    imgsz: int = 800
    iou_threshold: float = 0.6
    conf_threshold: float = 0.25
    device: str = "cuda"
    max_det: int = 50
    half = True
    augment: bool = False  # Enables test-time augmentation (TTA) during validation, potentially improving detection accuracy at the cost of inference speed
    is_detector: bool = False

    # logging
    save_hybrid = False  # Can cause false mAP ! used for autolabeling
    save_txt: bool = (
        False  # saves detection results in text files, with one file per image,
    )
    save_json: bool = False  # saves the results to a JSON file for further analysis, integration with other tools, or submission to evaluation servers like COCO.
    name: str = "val"
    project_name: str = "wildAI-detection"
    plots: bool = False


class CustomValidator(DetectionValidator):
    """From https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/val.py
    Adapted to compute confusion matrix for a given iou threshold
    """

    def init_metrics(self, model):
        """
        Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (
                val.endswith(f"{os.sep}val2017.txt")
                or val.endswith(f"{os.sep}test-dev2017.txt")
            )
        )  # is COCO
        self.is_lvis = (
            isinstance(val, str) and "lvis" in val and not self.is_coco
        )  # is LVIS
        self.class_map = (
            converter.coco80_to_coco91_class()
            if self.is_coco
            else list(range(1, len(model.names) + 1))
        )
        self.args.save_json |= (
            self.args.val and (self.is_coco or self.is_lvis) and not self.training
        )  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(
            nc=self.nc, conf=self.args.conf, iou_thres=self.args.iou, task="detect"
        )
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])


def ultralytics_val(args: Flags):
    # from pathlib import Path
    from datalabeling.train.utils import remove_label_cache

    # remove label.cache files
    remove_label_cache(data_config_yaml=args.data_config)

    for split in args.splits:
        print("-" * 20, split, "-" * 20)

        val_args = dict(model=args.weights, data=args.data_config)

        run_name = (
            args.name
            + "#"
            + split
            + f"#{round(args.conf_threshold * 100)}#{round(args.iou_threshold * 100)}#{args.augment}#{args.max_det}-"
        )

        validator = CustomValidator(
            args=val_args, save_dir=Path(args.project_name) / run_name
        )

        # set args
        validator.args.conf = args.conf_threshold
        validator.args.iou = args.iou_threshold
        validator.args.mode = "val"
        validator.args.imgsz = args.imgsz
        validator.args.batch = args.batch_size
        validator.args.device = args.device
        validator.args.augment = args.augment
        validator.args.split = split
        validator.args.name = run_name
        validator.args.project = args.project_name
        validator.args.max_det = args.max_det
        validator.args.save_crop = False
        validator.args.save_json = False
        validator.args.plots = args.plots
        validator.args.save_hybridd = args.save_hybrid
        validator.args.save_txt = args.save_txt
        validator.args.save_conf = args.save_txt

        # run evaluation
        results = validator()

        cf_matrix = validator.confusion_matrix.matrix
        labels = list(validator.names.values()) + ["background"]

        for i, label in enumerate(labels + ["background"]):
            if label == "background":
                break

            tp = cf_matrix[i, i]
            actual_positive = cf_matrix[:, i].sum()
            predicted_positive = cf_matrix[i, :].sum()
            # fp = predicted_positive - tp
            # fn = actual_positive - tp

            precision = tp / (predicted_positive + 1e-8)
            recall = tp / (actual_positive + 1e-8)
            f1score = 2 * precision * recall / (precision + recall + 1e-8)

            results = dict(
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1score=round(f1score, 4),
            )

            print(f"results for {label} : ", results, end="\n")


if __name__ == "__main__":
    args = parse(Flags)

    print("\nargs:\n", json.dumps(args.__dict__, indent=2), "\n", flush=True)

    ultralytics_val(args)
