if __name__ == "__main__":
    pass

    # from datalabeling.ml.models import Detector

    # path = r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images\00a033fefe644429a1e0fcffe88f8b39_0_4_512_512_1152_1152.jpg"

    # result = Detector.predict_url(image_path=path,
    #                               inference_service_url="http://localhost:4141/predict",
    #                               return_gps=True,

    #                               )
    # print(result)
    # =============================================================================
    # %%     Annotator
    # =============================================================================
    # from dotenv import load_dotenv

    # load_dotenv(r"D:\datalabeling\.env")

    # from datalabeling.ml import Annotator
    # import os
    # from pathlib import Path
    # import torch
    # from PIL import Image
    # from tqdm import tqdm

    # use_sliding_window = True

    # handler = Annotator(
    #     mlflow_model_alias="demo",
    #     mlflow_model_name="labeler",
    #     tilesize=800,
    #     overlapratio=0.1,
    #     use_sliding_window=use_sliding_window,
    #     confidence_threshold=0.5,
    #     # device="NPU", # "cpu", "cuda"
    #     # tag_to_append=f"-sahi:{use_sliding_window}",
    #     dotenv_path=r"D:\datalabeling\.env",
    # )
    # img = Image.open(
    #     r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images\00a033fefe644429a1e0fcffe88f8b39_0_4_512_512_1152_1152.jpg"
    # )
    # pred = handler.predict(img)

    # project_id = 3  # insert correct project_id by loooking at the url
    # top_n=10
    # handler.upload_predictions(project_id=project_id,top_n=top_n)

    # instances_count, images_count = handler.get_project_stats(
    #     project_id=project_id, annotator_id=0
    # )

    # from label_studio_sdk.client import LabelStudio
    # LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
    # API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

    # ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

    # project = ls.projects.get(3)

    # tasks = ls.tasks.list(project=project.id,)

    # =============================================================================
    # %%     Yolo architecture
    # =============================================================================
    # from ultralytics import YOLO
    # import torch
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
    # classifier = YOLO(
    #     r"D:\datalabeling\base_models_weights\yolo11s-cls\weights\best.pt",
    #     task="classify",
    # )

    # model = classifier.model.train()
    # out = model(torch.rand(4,3,96,96))

    # x = r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images\00a033fefe644429a1e0fcffe88f8b39_0_4_512_512_1152_1152.jpg"

    # (out1,) = detector(x)

    # xyxy = out1.obb.xyxy.cpu().long().tolist()[0]
    # x_min,y_min,x_max,y_max = xyxy
    # det_crop_img = out1.orig_img[y_min:y_max+1, x_min:x_max+1, :] # https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode BGR

    # # out = predict_with_uncertainty(model,x,n_iter=3)

    # image = r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\cls\train\false_positives"

    # out = classifier(image,batch=16,verbose=False)
    # pred = np.array([o.probs.top1 for o in out])
    # accuracy = (pred == np.zeros_like(pred)).sum()/len(pred)
    # # print(out.probs)

    # out1_1, = classifier(det_crop_img)
    # # print(out1_1.probs)

    # data = ImageFolder(root=r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\cls\train")

    # out.probs

    # =============================================================================
    # %%     Training image classifier
    # =============================================================================
    # from datalabeling.ml.train import ImageClassifier
    # from datalabeling.common.io import ClassifierDataModule
    # from lightning import Trainer
    # from torchvision import models

    # dm = ClassifierDataModule(
    #     train_dir=r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\cls\train",
    #     val_dir=r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\cls\val",
    #     batch_size=8,
    #     num_workers=1,
    #     img_size=96,
    # )
    # dm.setup("fit")

    # # loader = dm.train_dataloader()

    # # batch = next(iter(loader))

    # # model = classifier.model.train()
    # # for p in model.parameters():
    # #     p.require_grad = True

    # model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    # model.classifier = torch.nn.Linear(576, dm.num_classes)

    # routine = ImageClassifier(
    #     model=model, num_classes=2, threshold=0.5, label_smoothing=0.0, lr=1e-3
    # )

    # trainer = Trainer(
    #     max_epochs=5,
    # )
    # trainer.fit(routine, dm)
    # %% Detector

    from ultralytics import YOLO
    from PIL import Image
    import time
    import numpy as np
    from datalabeling.ml import Detector, Annotator
    from datalabeling.common.config import Detection
    from dotenv import load_dotenv
    from pathlib import Path
    import matplotlib.pyplot as plt

    # %matplotlib inline
    import folium
    from folium.plugins import MarkerCluster, FastMarkerCluster
    import pandas as pd
    from itertools import chain

    # det = Detection()

    handler = Detector(
        path_to_weights=r"D:\datalabeling\base_models_weights\best.pt",
        imgsz=960,
        tilesize=960,
        confidence_threshold=0.25,
        use_sliding_window=True,
    )

    image_dir = r"D:\savmap_dataset_v2\raw\tmp"
    save_path = Path(image_dir).parent / "detection_gps.csv"

    exts = [
        "*.jpg",
        "*.jpeg",
        "*.png",
    ]
    exts = [e.lower() for e in exts] + [e.capitalize() for e in exts]

    image_paths = chain.from_iterable([Path(image_dir).glob(ext) for ext in exts])

    image_paths = list(Path(image_dir).glob("*.jpg"))[:1]
    len(image_paths)

    results = handler.predict_directory(
        path_to_dir=None,
        images_paths=image_paths,
        as_dataframe=True,
        save_path=None,
    )
