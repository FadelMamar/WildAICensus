if __name__ == "__main__":
    pass

    # import os
    # os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

    from ultralytics import settings

    # Update a setting
    settings.update({"mlflow": False})

    # =============================================================================
    # %%     Yolo architecture
    # =============================================================================
    from ultralytics import YOLO
    import torch
    from datalabeling.ml.utils import DetectionSystem, CustomYOLO
    import os
    # import numpy as np
    # from torchvision.datasets import ImageFolder

    # def predict_with_uncertainty(model, img_path, n_iter=10):
    #     """Monte Carlo Dropout for uncertainty estimation"""

    #     from baal.bayesian.dropout import patch_module
    #     from baal.active.heuristics import BALD

    #     all_preds = []
    #     heuristic = BALD(reduction="mean")
    #     _model = patch_module(model, inplace=False)

    #     num_classes = model.nc

    #     with torch.no_grad():
    #         for _ in range(n_iter):
    #             pred = _model(img_path)[0]  # YOLO prediction
    #             try:
    #                 pred = pred.boxes.data.cpu().numpy()
    #             except:
    #                 pred = pred.obb.data.cpu().numpy()

    #             idx = pred[:, -1].astype(int)
    #             dummy = np.zeros((pred.shape[0], num_classes, 5))
    #             dummy[idx] = pred[:, :-2]
    #             all_preds.append(dummy)

    #     # Calculate uncertainty using BALD
    #     stacked = np.stack(all_preds, axis=-1)
    #     uncertainty = heuristic(stacked)

    #     return {"boxes": stacked, "uncertainty": uncertainty}

    # detector = YOLO(r"D:\datalabeling\base_models_weights\best.pt", task="detect")

    # model = DetectionSystem(
    #                             count_regressor_layers=22,
    #                             area_regressor_layers=16,
    #                             roi_classifier_layers={"p3": 16, "p4": 19},
    #                             fp_tp_loss_weight=3.0,
    #                             is_fp_tp_multiplier=False,
    #                             count_loss_weight=0.0,
    #                             area_loss_weight=0.0,
    #                             roi_scale_factor=[2.0, 3.0],
    #                             # cfg="yolo11s.yml",
    #                             ch=3,
    #                             nc=1,
    #                             verbose=True
    #                             )

    # # base_model = detector.model.model

    # detector.eval()

    # (a,(b,c)) = detector(torch.rand(5,3,256,256))

    model = CustomYOLO(
        count_regressor_layers=27,
        area_regressor_layers=None,
        mask_p3_layer_indx=21,
        pos_weight=None,
        roi_classifier_layers={"p3": 21, "p4": 24},
        fp_tp_loss_weight=3.0,
        mask_loss_weight=0.0,
        count_loss_weight=0.0,
        area_loss_weight=0.0,
        roi_scale_factor=[
            2.0,
        ],
        model=r"D:\datalabeling\runs\detect\train6\weights\best.pt",
        # model = r"..\configs\yolo_configs\models\yolov8-p2.yaml",
    )  # .load(r"..\base_models_weights\best.pt")

    # model.train(
    #     data=r"..\configs\yolo_configs\data\data_config.yaml",
    #     epochs=5,
    #     batch=8,
    #     freeze=10,
    #     imgsz=800,
    #     workers=0,
    # )

    # model.val(data=r"..\configs\yolo_configs\data\dataset_identification-detection.yaml",batch=16)


# =============================================================================
# %% boundingbox to mask
# =============================================================================
from ultralytics.data.utils import polygon2mask
import numpy as np
from torch import nn
import torch
from itertools import product
import matplotlib.pyplot as plt
from random import shuffle


def generate_synthetic_boxes(
    img_height: int, img_width: int, area_thresh: float, num: int = 10
) -> torch.Tensor:
    w = int(np.sqrt(area_thresh))
    h = w

    xs = torch.randint(
        low=0,
        high=img_width - w,
        size=(num,),
    )

    ys = torch.randint(
        low=0,
        high=img_height - h,
        size=(num,),
    )

    boxes = []
    pairs = list(product(xs, ys))
    shuffle(pairs)  # shuffle
    for x, y in pairs[:num]:
        box = torch.Tensor([x, y, x + w, y + h])
        boxes.append(box)

    return torch.vstack(boxes)


def bbox_to_mask(bboxes: torch.Tensor, imgsz: tuple = (800, 800)):
    """
    Convert multiple bounding boxes to polygons in batch.

    Args:
        bboxes: Tensor/array of shape (N, 4) with N bounding boxes
        format: Output format - 'numpy' or 'torch'

    Returns:
        Polygons in shape (N, 4, 2) - N polygons with 4 vertices each
    """
    if isinstance(bboxes, list):
        bboxes = torch.Tensor(bboxes)
    elif isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes).float()
    else:
        bboxes = bboxes.cpu()

    # Extract coordinates
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # Stack coordinates to create (N, 4, 2) tensor
    polygons = (
        torch.stack(
            [
                torch.stack([x1, y1], dim=1),  # top-left
                torch.stack([x2, y1], dim=1),  # top-right
                torch.stack([x2, y2], dim=1),  # bottom-right
                torch.stack([x1, y2], dim=1),  # bottom-left
            ],
            dim=1,
        )
        .cpu()
        .tolist()
    )

    polygons = [np.array(p) for p in polygons]

    mask = polygon2mask(
        imgsz,  # tuple
        polygons,  # input as list
        color=1,  # 8-bit binary
        downsample_ratio=1,
    )

    mask = torch.from_numpy(mask).int()

    return mask


bboxes = generate_synthetic_boxes(800, 800, 20**2, num=10)

mask = bbox_to_mask(bboxes, imgsz=(800, 800))

plt.imsave("mask.jpg", mask)
