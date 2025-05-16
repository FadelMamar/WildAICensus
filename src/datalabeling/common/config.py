from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
import math
import torch
import geopy
from PIL import Image


@dataclass  # (frozen=True)
class Detection:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    label: int
    class_name: str
    score: float = None
    gps_loc: str = None
    image_gps_loc: str = None
    parent_image: str = None

    @classmethod
    def from_coco(
        cls,
        coco: dict,
        parent_image: str,
        image_gps_loc: str = None,
        gps_loc: str = None,
    ):
        bbox = coco["bbox"]
        label = coco["category_id"]
        class_ = coco["category_name"]
        score = coco.get("score", None)

        det = cls(
            x_min=int(bbox[0]),
            y_min=int(bbox[1]),
            x_max=int(bbox[0] + bbox[2]),
            y_max=int(bbox[1] + bbox[3]),
            class_name=class_,
            label=label,
            score=score,
            image_gps_loc=image_gps_loc,
            gps_loc=gps_loc,
            parent_image=parent_image,
        )

        return det
    
    @property
    def is_empty(self):
        return any([self.x is None, self.y is None, self.w is None, self.h is None])

    def to_dict(
        self,
    ):
        return vars(self)

    def to_ls(
        self, from_name, to_name, label_type, img_height: int, img_width: int
    ) -> dict:
        # formatting the prediction to work with Label studio
        score = self.score
        if not isinstance(score, float):
            score = 0.0
        template = {
            "from_name": from_name,
            "to_name": to_name,
            "type": label_type,
            "original_width": img_width,
            "original_height": img_height,
            "image_rotation": 0,
            "value": {
                label_type: [
                    self.class_name,
                ],
                "x": self.x_min / img_width * 100,
                "y": self.y_min / img_height * 100,
                "width": self.w / img_width * 100,
                "height": self.h / img_height * 100,
                "rotation": 0,
            },
            "score": score,
        }
        return template

    def get_base_image(self) -> Image.Image:
        assert self.parent_image is not None, "Parent image is not defined"
        return Image.open(self.parent_image)

    @property
    def area(
        self,
    ):
        return self.w * self.h

    @property
    def x(
        self,
    ):
        return math.floor((self.x_min + self.x_max) / 2)

    @property
    def y(
        self,
    ):
        return math.floor((self.y_min + self.y_max) / 2)

    @property
    def w(
        self,
    ):
        return int(self.x_max - self.x_min)

    @property
    def h(
        self,
    ):
        return int(self.y_max - self.y_min)

    @property
    def gps_as_decimals(
        self,
    ):
        assert isinstance(self.gps_loc, str)

        point = geopy.Point.from_string(self.gps_loc)

        lat = point.latitude
        long = point.longitude
        alt = point.altitude * 1e3  # converting to meters

        return lat, long, alt


@dataclass
class DataConfig:
    root_dir: str = "D:\\"
    save_dir: Optional[Path] = None
    results_filename: str = "detection_results.csv"
    ground_truth_filename: str = "ground_truth.csv"

    dotenv_path: str = ""

    # slicing cfg
    slice_width: int = 640
    slice_height: int = 640
    overlap_ratio: float = 0.2
    min_area_ratio: float = 0.8
    empty_ratio: float = 1.0
    clear_output: bool = False
    save_all: bool = False
    save_only_empty: bool = False
    load_coco_annotations: bool = False

    parse_ls_config: bool = False

    dest_path_labels: str = ""
    dest_path_images: str = ""

    coco_json_dir: str = ""
    ls_json_dir: str = ""

    yolo_data_config_yaml: str = ""

    is_single_cls: bool = False

    verbose: bool = False


@dataclass
class PredictionConfig:
    imgsz: int = 960
    tilesize: int = 960
    overlap_ratio: float = 0.2
    confidence_threshold: float = 0.25
    min_area: int = 10 * 10
    max_area: int = None
    use_sliding_window: bool = True
    nms_iou: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Image classifier imgsz
    cls_imgsz: int = 96


@dataclass
class TrainingConfig:
    # model type
    is_single_cls: bool = False
    is_rtdetr: bool = False
    task: str = "detect"  # "detect" "obb" "segment"
    model_type: str = "ultralytics"

    # active learning flags
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_model_alias: str = None

    # training data
    yolo_yaml: str = None  # os.path.join(CUR_DIR,'../../../data/data_config.yaml')
    yolo_arch_yaml: str = None

    ultralytics_pos_weight: float = 1.0

    # training flags
    imgsz: int = 800
    path_weights: str = None
    lr0: float = 1e-4
    lrf: float = 1e-2
    warmup_epochs: int = 3
    batchsize: int = 32
    epochs: int = 50
    seed = 41
    optimizer: str = "AdamW"
    optimizer_momentum: float = 0.99
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 10
    val: str = "True"
    yolo_iou_val: float = 0.6
    dfl: float = 1.5
    cls: float = 0.5
    box: float = 7.5

    # classifier
    cls_num_classes: int = 2
    cls_label_smoothing: float = 0.0
    cls_thrs: float = 0.5
    cls_workdir: str = "runs-classifier"
    cls_data_dir: str = None
    cls_monitor_metric: str = "val_f1score"
    cls_monitor_mode: str = "max"
    cls_auto_augment: str = "augmix"
    cls_is_features: bool = False
    cls_tn_ratio:float=1.0

    # herdnet
    herdnet_training_backend: str = "original"  # pl or original
    herdnet_pl_ckpt: str = None
    herdnet_val_batchsize: int = 1
    herdnet_num_classes: int = 2  # binary, includes background
    herdnet_ce_weight = None
    herdnet_down_ratio: int = 2
    herdnet_ptr_model_classes: int = 4
    herndet_empty_ratio: float = 0.0
    herdnet_valid_freq: int = 4
    herdnet_work_dir: str = "./runs-herndet"
    herdnet_lr_milestones: Sequence[int] = (20,)
    herdnet_warmup_iters: int = 100

    # pretraining
    use_pretraining: bool = False
    ptr_data_config_yaml: str = None
    ptr_tilesize: int = 640
    ptr_batchsize: int = 32
    ptr_epochs: int = 10
    ptr_lr0: float = 1e-4
    ptr_lrf: float = 1e-1
    ptr_freeze: int = None

    # continual learning flags
    use_continual_learning: bool = False
    cl_ratios: Sequence[float] = (1.0,)  # ratio = num_empty/num_non_empty
    cl_epochs: Sequence[int] = (20,)
    cl_freeze: Sequence[int] = (0,)
    cl_lr0s: Sequence[float] = (5e-5,)
    cl_save_dir: str = None  # should be given!
    cl_data_config_yaml: str = None
    cl_batch_size: int = 16

    # hard negative data sampling learning mode
    use_hn_learning: bool = False
    hn_save_dir: str = None
    hn_data_config_yaml: str = None
    hn_imgsz: int = 1280  # used to resize the input image
    hn_tilesize: int = 1280  # used for sliding window based detections
    hn_num_epochs: int = 10
    hn_freeze: int = 20
    hn_lr0: float = 5e-5
    hn_lrf: float = 1e-1
    hn_batch_size: int = 16
    hn_is_yolo_obb: bool = False
    hn_use_sliding_window = True  # can't change thru cli
    hn_overlap_ratio: float = 0.2
    hn_map_thrs: float = (
        0.35  # mAP threshold. lower than it is considered sample of interest
    )
    hn_score_thrs: float = 0.7
    hn_confidence_threshold: float = 0.25
    hn_ratio: int = 20  # ratio = num_empty/num_non_empty. Higher allows to look at all saved empty images
    hn_uncertainty_thrs: float = 5  # helps to select those with high uncertainty
    hn_uncertainty_method: str = "entropy"
    hn_load_results: bool = False

    # regularization
    dropout: float = 0.0  # used only for classification tasks
    weight_decay: float = 5e-4

    # transfer learning
    freeze: int = None

    # lr scheduling
    cos_annealing: bool = True

    # run and project name MLOps
    run_name: str = "debug"
    project_name: str = "wildAI"
    tag: Sequence[str] = ("",)

    # data augmentation https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters
    rotation_degree: float = 45.0
    mixup: float = 0.0
    shear: float = 10.0
    copy_paste: float = 0.0
    erasing: float = 0.0
    scale: float = 0.0
    fliplr: float = 0.5
    flipud: float = 0.5
    hsv_h: float = 0.0
    hsv_s: float = 0.3
    hsv_v: float = 0.3
    translate: float = 0.2
    mosaic: float = 0.0
    multi_scale: bool = False


@dataclass
class LabelConfig:
    discard: Optional[List[str]] = None
    keep: Optional[List[str]] = None
    label_map: Optional[str] = None


@dataclass
class EvaluationConfig:
    score_threshold: float = 0.7
    map_threshold: float = 0.3
    uncertainty_method: str = "entropy"
    uncertainty_threshold: float = 4
    tp_iou_threshold: float = 0.4
    fp_tp_ratio_threshold: float = 0.2
    fn_tp_ratio_threshold: float = 0.2
    is_yolo_obb: bool = False
    score_col: str = "max_scores"
    load_results: bool = False


@dataclass
class ExportConfig:
    export_format: str = None
    export_batch_size: int = 1
    export_model_weights: str = ""
    half: bool = False  # to use FP16
    int8: bool = False  # int8 quantization\
    dynamic: bool = False  # allows dynamic input sizes
