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

    # detector = DetectionSystem(roi_classifier_layers=set(),
    #                            count_regressor_layers=19,
    #                            area_regressor_layers=16,
    #                            cls_num_classes=2,
    #                            cfg=r"..\configs\yolo_configs\models\yolo11-obb.yaml",
    #                            ch=3,
    #                            nc=1,
    #                            verbose=True
    #                            )

    # # base_model = detector.model.model

    # detector.eval()

    # (a,(b,c)) = detector(torch.rand(5,3,256,256))

    model = CustomYOLO(
        count_regressor_layers=15,
        area_regressor_layers=18,
        roi_classifier_layers={"p3": 15, "p4": 24},
        model=r"..\configs\yolo_configs\models\yolov8-p2.yaml",
    )  # load(r"..\base_models_weights\best.pt")

    model.model(
        torch.rand(1, 3, 512, 512),
    )

    model.train(
        data=r"..\configs\yolo_configs\data\data_config.yaml",
        epochs=5,
        batch=4,
        freeze=9,
        imgsz=640,
        workers=0,
    )

    # # hook
    # activation = {}
    # def get_activation(name):
    #     def hook(module, args, output):
    #         activation[name] = output #.detach()
    #         return None
    #     return hook

    # # register forward hook
    # layer_number = 5
    # base_model[layer_number].register_forward_hook(get_activation(layer_number))

    # out = detector(torch.rand(1,3,512,512),verbose=False)

    # activation[layer_number]

    # ]
    # exts = [e.lower() for e in exts] + [e.capitalize() for e in exts]

    # image_paths = chain.from_iterable([Path(image_dir).glob(ext) for ext in exts])

    # image_paths = list(Path(image_dir).glob("*.jpg"))[:1]
    # len(image_paths)

    # results = handler.predict_directory(
    #     path_to_dir=None,
    #     images_paths=image_paths,
    #     as_dataframe=True,
    #     save_path=None,
    # )
