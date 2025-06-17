import streamlit as st
import pandas as pd
from typing import List, Sequence
import requests, base64
from dotenv import load_dotenv
from pathlib import Path
import os
import traceback
from datalabeling.common.io import load_yaml
from datalabeling.common.annotation_utils import GPSUtils
import logging

from label_studio_sdk.client import LabelStudio
from itertools import chain
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

from datalabeling.common.config import (
    TilingConfig,
    TrainingConfig,
    DataConfig,
    LabelConfig,
)
from datalabeling.common.dataset_loader import LabelingDataset
from datalabeling.common.config import PredictionConfig
from datalabeling.common.io import get_images_from_dirs
from datalabeling.ml.interface import InferenceEngine


DOT_ENV = Path(__file__) / "../.env"
load_dotenv(DOT_ENV)


class StreamlitLogHandler(logging.Handler):
    def __init__(self, widget_func):
        super().__init__()
        self.widget = widget_func  # e.g. st.empty().code

    def emit(self, record):
        msg = self.format(record)
        self.widget(msg)


def main():
    st.set_page_config(page_title="Labeling Workflow Manager", layout="wide")

    st.title("Labeling Workflow Management")

    # Sidebar for common controls
    # with st.sidebar:
    #     st.header("API Configuration")
    #     label_studio_token = st.text_input("Label Studio Token", type="password")
    # training_api_token = st.text_input("Training API Token", type="password")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Upload Annotations",
            "Project Analytics",
            "Model Training",
            "GPS",
            "Inference",
        ]
    )

    with tab1:
        st.header("Upload to Label Studio")
        with st.form("upload_annotations"):
            project_id = st.number_input("Project ID", min_value=0, step=1)
            model_alias = st.text_input("Model Alias", value="demo").strip()
            detector_name = st.text_input("Detector name", value="labeler").strip()
            roi_classifier_path = st.text_input(
                "roi_classifier_path", value="base_models_weights/roi_classifier.ckpt"
            ).strip()
            roi_cls_is_features = st.number_input(
                "roi_cls_is_features", min_value=0, max_value=1, step=1
            )
            confidence_threshold = st.text_input(
                "Confidence threshold", value=0.2
            ).strip()
            # path_to_weights = st.text_input(
            #     "Path to model weights",
            #     # value=None,
            # ).strip()
            tile_size = st.number_input("Tile size", min_value=800, step=1)
            top_n = st.number_input("Top n", min_value=0, step=1)

            annotator_kwargs = {
                # "path_to_weights": path_to_weights,
                "mlflow_model_alias": model_alias,
                "mlflow_model_name": detector_name,
                "tilesize": tile_size,
                "overlapratio": 0.1,
                "use_sliding_window": True,
                "confidence_threshold": 0.1,
                "roi_classifier_path": roi_classifier_path,
                "roi_cls_is_features": bool(roi_cls_is_features),
            }

            log_widget = st.empty().code
            handler = StreamlitLogHandler(log_widget)
            logger = logging.getLogger()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            if st.form_submit_button("Upload Annotations"):
                try:
                    with st.spinner("Uploading annotations...", show_time=True):
                        upload_to_label_studio(
                            project_id=project_id,
                            top_n=top_n,
                            annotator_kwargs=annotator_kwargs,
                        )
                        st.success("Done!")

                except Exception as e:
                    traceback.print_exc()

    with tab2:
        st.header("Project Analytics")
        with st.form("project_stats"):
            stats_project_id = st.number_input(
                "Analytics Project ID", min_value=0, step=1
            )

            log_widget = st.empty().code
            handler = StreamlitLogHandler(log_widget)
            logger = logging.getLogger()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            if st.form_submit_button("Show Statistics"):
                try:
                    with st.spinner("Computing statistics...", show_time=True):
                        instances_count, images_count = get_project_statistics(
                            stats_project_id
                        )
                    st.dataframe(instances_count, use_container_width=False)
                    st.dataframe(images_count, use_container_width=False)

                except Exception as e:
                    st.error(f"Failed to fetch statistics: {str(e)}")

        with st.form("train_val_test_stats"):
            path_to_yaml = st.text_input(
                "Path to data.yaml file",
                value=r"..\configs\yolo_configs\data\data_config.yaml",
            ).strip()
            split = st.text_input(
                "Split to select", value="train", help="train val or test"
            ).strip()

            if st.form_submit_button("Show Statistics"):
                with st.spinner("Computing statistics...", show_time=True):
                    stats = visualize_splits_distribution(
                        data_yaml_path=path_to_yaml, split=split
                    )

                st.bar_chart(
                    stats["instances_count"],
                    x="class",
                    y="count",
                    x_label="Species",
                    y_label="Instance count",
                )
                st.bar_chart(
                    stats["images_count"],
                    x="class",
                    y="image",
                    x_label="Species",
                    y_label="Images count",
                )

    with tab3:
        st.header("Training and Registering")

        st.subheader("Training")
        with st.form("model_training"):
            training_project_id = st.number_input("Project ID", min_value=0, step=1)
            service_url = st.text_input(
                "service_url", value="http://localhost:5500/train"
            ).strip()
            root_dir = st.text_input("data_root_drive", value="D:\\").strip()
            epochs = st.slider("Training Epochs", 1, 50, 5)
            batch_size = st.selectbox("Batch Size", [16, 32, 64])
            lr0 = st.selectbox("lr0", [1e-4, 1e-3, 5e-5, 1e-5])
            lrf = st.selectbox(
                "lrf",
                [
                    1e-2,
                    1e-1,
                ],
            )
            patience = st.number_input(
                "patience",
                min_value=3,
            )
            yolo_yaml = st.text_input(
                "validation data_config_yaml",
                value="configs\yolo_configs\data\data_config.yaml",
            ).strip()
            yolo_arch_yaml = st.text_input(
                "yolo_arch_yaml", value="configs\yolo_configs\models\yolov8-p2.yaml"
            ).strip()
            path_weights = st.text_input(
                "path_weights",
            ).strip()

            mlflow_model_alias = st.text_input(
                "mlflow_model_alias", value="demo"
            ).strip()
            mlflow_model_name = st.text_input(
                "mlflow_model_name", value="labeler"
            ).strip()

            cls_num_classes = st.number_input("cls_num_classes", min_value=2, step=1)
            cls_label_smoothing = st.number_input("cls_label_smoothing", min_value=0)
            cls_data_dir = st.text_input(
                "cls_data_dir",
            ).strip()
            cls_training_backend = st.selectbox(
                "cls_training_backend",
                [
                    "pl",
                ],
            )

            imgsz = st.number_input("empty_ratio", min_value=128, step=32, value=800)
            object_detector_arch = st.selectbox(
                "object_detector_arch", ["yolo", "rtdetr"]
            )  # "yolo", "rtdetr",  # not included for now"custom_yolo"
            ultralytics_pos_weight = st.slider("ultralytics_pos_weight", 1, 10, 1)
            weight_decay = st.number_input(
                "weight_decay",
                min_value=0,
            )

            cl_freeze = st.slider("freeze", 1, 25, 1)
            cl_save_dir = st.text_input("save_dir_results", value=".tmp").strip()
            cl_ratios = st.number_input(
                "empty_ratio",
                min_value=0,
            )

            model_type = st.selectbox(
                "model_type",
                [
                    "detector",
                ],
            )  # detector, classifier, herdnet
            project_name = st.text_input("project_name", value="demo").strip()
            run_name = st.text_input("run_name", value="demo").strip()
            device = st.selectbox("computing_device", ["cpu", "cuda:0"])

            if st.form_submit_button("Start Training"):
                training_cfg = TrainingConfig()

                training_cfg.yolo_arch_yaml = yolo_arch_yaml
                training_cfg.path_weights = path_weights

                training_cfg.mlflow_model_alias = mlflow_model_alias
                training_cfg.mlflow_model_name = mlflow_model_name

                training_cfg.cls_label_smoothing = cls_label_smoothing
                training_cfg.cls_num_classes = cls_num_classes
                training_cfg.cls_is_features = True
                training_cfg.cls_data_dir = cls_data_dir
                training_cfg.cls_training_backend = cls_training_backend

                training_cfg.imgsz = imgsz

                training_cfg.model_type = model_type

                training_cfg.batchsize = batch_size
                training_cfg.epochs = epochs

                training_cfg.task = "detect"  # ultralytics
                training_cfg.project_name = project_name
                training_cfg.run_name = run_name

                training_cfg.lr0 = lr0
                training_cfg.lrf = lrf
                training_cfg.patience = patience

                training_cfg.object_detector_arch = object_detector_arch
                training_cfg.custom_yolo_kwargs = dict(
                    count_regressor_layers=22,  # p5
                    area_regressor_layers=16,
                    mask_p3_layer_indx=16,
                    mask_loss_weight=0.0,
                    roi_classifier_layers={"p3": 16, "p4": 19},
                    fp_tp_loss_weight=3.0,
                    count_loss_weight=0.0,
                    area_loss_weight=0.0,
                    roi_scale_factor=[
                        2.0,
                    ],
                )

                training_cfg.ultralytics_pos_weight = ultralytics_pos_weight
                training_cfg.weight_decay = weight_decay

                training_cfg.warmup_epochs = 0
                training_cfg.dfl = 1.5  # 1.5
                training_cfg.cls = 0.5  # 0.5
                training_cfg.box = 7.5  # 7.5

                training_cfg.cl_batch_size = (training_cfg.batchsize,)
                training_cfg.use_continual_learning = False
                training_cfg.cl_ratios = (cl_ratios,)  # ratio = num_empty/num_non_empty
                training_cfg.cl_epochs = (training_cfg.epochs,)
                training_cfg.cl_freeze = (cl_freeze,)
                training_cfg.cl_lr0s = (training_cfg.lr0,)
                training_cfg.cl_save_dir = cl_save_dir

                training_cfg.device = device

                # data set creation config
                data_config = DataConfig(
                    is_single_cls=True,
                    root_dir=root_dir,
                    yolo_data_config_yaml=training_cfg.yolo_yaml,
                    dotenv_path=DOT_ENV,
                    tilesize=training_cfg.imgsz,
                    overlap_ratio=0.2,
                    save_all=False,
                    dest_dir=cl_save_dir,  # folder holding images and labels in yolo format
                    save_only_empty=False,
                    load_coco_annotations=False,
                    parse_ls_config=True,
                    empty_ratio=training_cfg.cl_ratios[0],
                )

                label_map = (
                    Path(__file__) / "../exported_annotations/label_mapping.json"
                )
                label_config = LabelConfig(
                    discard=[
                        "other",
                        "rocks",
                        "vegetation",
                        "detection",
                        "termite mound",
                        "label",
                    ],
                    label_map=label_map,
                )

                ref_data_config = load_yaml(yolo_yaml)
                if isinstance(ref_data_config["val"], Sequence):
                    vals = [
                        os.path.join(ref_data_config["path"], v)
                        for v in ref_data_config["val"]
                    ]
                else:
                    vals = [
                        os.path.join(ref_data_config["path"], ref_data_config["val"]),
                    ]

                training_cfg.yolo_yaml = dict(
                    path=root_dir,
                    train=[
                        data_config.dest_path_images,
                    ],
                    val=[os.path.relpath(v, start=data_config.root_dir) for v in vals],
                    nc=ref_data_config["nc"],
                    names=ref_data_config["names"],
                )
                # training_cfg.cl_data_config_yaml = training_cfg.yolo_yaml

                with st.spinner("Running training job...", show_time=True):
                    start_training(
                        args=training_cfg,
                        data_config=data_config,
                        label_config=label_config,
                        service_url=service_url,
                        slice_data=False,
                        project_id=training_project_id,
                    )

        st.subheader("Registering")
        with st.form("model_registeration"):
            pass

            if st.form_submit_button("Register"):
                register_model(
                    weights_path=f"D:/datalabeling/base_models_weights/best.pt",
                    name="labeler",
                    export_format="pt",
                    imgsz=800,
                    device="cpu",
                    mlflow_tracking_uri="http://localhost:5000",
                )

                pass

    with tab4:
        st.header("Visualizations")

        st.subheader("Image GPS")
        with st.form("gps_coords"):
            image_dir = st.text_input(
                "Path to images directory (without quotes)"
            ).strip()

            save_path_map = st.text_input(
                "Path to save map (without quotes)",
                help="something like my_map.html",
                # value='map.html'
            ).strip()

            map_style = st.radio(
                label="map stype",
                options=[
                    "Esri.WorldImagery",
                    "OpenStreetMap",
                ],
            )

            if st.form_submit_button("Get coordinates"):
                with st.spinner("Running...", show_time=True):
                    df_gps_coords = get_gps_coords(
                        image_paths=None, image_dir=image_dir
                    )

                    st.dataframe(df_gps_coords, use_container_width=False)

                    folium_static(
                        get_map_with_detections(
                            locations=df_gps_coords,
                            map_style=map_style,
                            save_path=save_path_map if len(save_path_map) > 5 else None,
                        )
                    )

        st.subheader("Detections GPS")
        with st.form("folium_map_det"):
            detections_path = st.text_input(
                "Path to detections csv (without quotes)"
            ).strip()

            save_path_map = st.text_input(
                "Path to save map (without quotes)",
                help="something like my_map.html",
                # value='map.html'
            ).strip()

            map_style = st.radio(
                label="map stype",
                options=[
                    "Esri.WorldImagery",
                    "OpenStreetMap",
                ],
            )

            if st.form_submit_button("Visualize"):
                with st.spinner("Running...", show_time=True):
                    df_results_px = pd.read_csv(detections_path)

                    st.dataframe(df_results_px, use_container_width=False)

                    folium_static(
                        get_map_with_detections(
                            locations=df_results_px,
                            map_style=map_style,
                            save_path=save_path_map if len(save_path_map) > 5 else None,
                        )
                    )

        st.subheader("Groundtruth GPS")
        with st.form("folium_map_gt"):
            project_id = st.number_input("Project ID", min_value=0, step=1)
            top_n = st.number_input("top_n", min_value=0, step=1, value=0)
            overlapfactor = st.number_input(
                "overlapfactor", min_value=0.0, max_value=0.9, value=0.1
            )
            ratiowidth = st.number_input(
                "ratiowidth", min_value=0.0, max_value=1.0, value=0.5
            )
            ratioheight = st.number_input(
                "ratioheight", min_value=0.0, max_value=1.0, value=0.5
            )
            rmheight = st.number_input(
                "rmheight", min_value=0.0, max_value=1.0, value=0.1
            )
            rmwidth = st.number_input(
                "rmwidth", min_value=0.0, max_value=1.0, value=0.1
            )
            flight_height = st.number_input(
                "flight_height in [m]", min_value=10.0, value=180.0
            )
            sensor_height = st.number_input(
                "sensor_height in [mm]", min_value=0.0, value=24.0
            )
            gsd = st.number_input("gsd in [cm/px]", min_value=0.0, value=2.26)
            dest = st.text_input(
                "destination directory (without quotes)", value="D:\Phd"
            ).strip()
            do_tiling = st.radio(
                label="Tile data",
                options=[
                    True,
                    False,
                ],
            )

            root_images_dir = st.text_input(
                "Path to UNTILED images directory (without quotes)",
                help="something like my_map.html",
                value=None,
            )
            if root_images_dir:
                root_images_dir = root_images_dir.strip()

            save_path_map = st.text_input(
                "Path to save map (without quotes)",
                help="something like my_map.html",
                # value='map.html'
            ).strip()

            map_style = st.radio(
                label="map stype",
                options=[
                    "Esri.WorldImagery",
                    "OpenStreetMap",
                ],
            )

            if st.form_submit_button("Visualize"):
                with st.spinner("Running...", show_time=True):
                    config = TilingConfig(
                        root=root_images_dir,
                        overlapfactor=overlapfactor,
                        ratiowidth=ratiowidth,
                        ratioheight=ratioheight,
                        rmheight=rmheight,
                        rmwidth=rmwidth,
                        flight_height=flight_height,
                        sensor_height=sensor_height,
                        gsd=gsd,
                        dest=dest,
                        save_coords_only=not do_tiling,  # set to False to save tiles i.e. patches
                    )

                    df_gt = get_gps_coords_from_ls(
                        config=config,
                        project_id=project_id,
                        top_n=top_n,
                        load_existing_metadata=True,
                    )
                    st.dataframe(df_gt, use_container_width=False)
                    folium_static(
                        get_map_with_detections(
                            locations=df_gt,
                            map_style=map_style,
                            save_path=save_path_map if len(save_path_map) > 5 else None,
                        )
                    )

    with tab5:
        st.header("Inference")

        with st.form("inference"):
            model_alias = st.text_input("Model Alias", value="yolov12s").strip()
            model_name = st.text_input("Model name", value="detector").strip()
            confidence_threshold = st.text_input(
                "Confidence threshold", value=0.15
            ).strip()
            image_dir = st.text_input(
                "Path to images directory (without quotes)"
            ).strip()
            save_path = st.text_input(
                "Save path (without quotes)", value="detections.csv"
            ).strip()

            # annotation_file = st.file_uploader("Annotation File (JSON)", type=["json"])

            log_widget = st.empty().code
            handler = StreamlitLogHandler(log_widget)
            logger = logging.getLogger()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            if st.form_submit_button("Get predictions"):
                with st.spinner("Running...", show_time=True):
                    df_results_px = run_inference(
                        image_dir=image_dir,
                        alias=model_alias,
                        save_path=save_path,
                        image_paths=None,
                        confidence_threshold=confidence_threshold,
                        dotenv_path=DOT_ENV,
                        name=model_name,
                        exts=[
                            "*.jpg",
                            "*.jpeg",
                            "*.png",
                        ],
                    )
                st.dataframe(df_results_px, use_container_width=False)

                # plot detections on folium map
                folium_static(get_map_with_detections(locations=df_results_px))


def get_inference_engine(annotator_kwargs: dict) -> InferenceEngine:
    config = PredictionConfig(
        imgsz=annotator_kwargs.get("tilesize", 800),
        tilesize=annotator_kwargs.get("tilesize", 800),
        overlap_ratio=annotator_kwargs.get("overlap_ratio", 0.2),
        confidence_threshold=annotator_kwargs.get("confidence_threshold", 0.2),
        inference_service_url=annotator_kwargs.get("inference_service_url", None),
        # min_area=100,
        # max_area=None,
        cls_imgsz=98,
        # device='cuda'
    )

    annotator, _ = InferenceEngine.load_engine(
        pred_config=config,
        roi_classifier_path=annotator_kwargs.get("roi_classifier_path", None),
        roi_cls_is_features=annotator_kwargs.get("roi_cls_is_features", True),
        roi_cls_label_map={0: "gt", 1: "tn"},
        roi_keep_classes=["gt"],
        detection_label_map=None,  # {0: "wildlife"},
        feature_extractor_path="facebook/dinov2-with-registers-small",
        detection_model=None,
        mlflow_model_alias="demo",
        mlflow_model_name="labeler",
        set_ls_client=True,
    )

    return annotator


@st.cache_data
def run_inference(
    image_dir: str,
    annotator_kwargs: dict,
    save_path: str = None,
    image_paths: list[None] = None,
) -> None:
    engine = get_inference_engine(annotator_kwargs=annotator_kwargs)

    if image_paths is None:
        assert image_dir is not None
        image_paths = get_images_from_dirs([image_dir])

    results = engine.inference(images_paths=image_paths, return_as_df=True)

    if save_path:
        results[["Latitude", "Longitude", "Elevation"]].to_csv(save_path, index=False)

    return results


@st.cache_data
def get_gps_coords(
    image_dir: str,
    image_paths: list[str] = None,
):
    if image_paths is None:
        assert image_dir is None
        image_paths = get_images_from_dirs([image_dir])

    gps_coords = [
        GPSUtils.get_gps_coord(file_name=path, return_as_decimal=True)[0]
        for path in image_paths
    ]

    gps_coords = pd.DataFrame(
        data=gps_coords, columns=["Latitude", "Longitude", "Elevation"]
    )

    return gps_coords


@st.cache_data
def get_gps_coords_from_ls(
    config: TilingConfig, project_id: int, top_n=0, load_existing_metadata=True
) -> pd.DataFrame:
    load_dotenv(DOT_ENV)

    LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
    API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
    # print("API-Key",API_KEY)
    labelstudio_client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

    # check connection
    labelstudio_client.projects.get(id=project_id)

    # print(config)

    assert config.root is not None, "Provide path to the original images directory"

    dataset = LabelingDataset.from_ls(
        labelstudio_client,
        project_id=project_id,
        config=config,
        top_n=top_n,
        load_existing_metadata=load_existing_metadata,
    )
    gps_data = dataset.export_detections_gps()

    return gps_data


def get_map_with_detections(
    locations: pd.DataFrame,
    map_style: str = "Esri.WorldImagery",
    zoom_start=14,
    save_path: str = None,
) -> folium.Map:
    m = folium.Map(
        location=[locations["Latitude"].mean(), locations["Longitude"].mean()],
        zoom_start=zoom_start,
        # attr=map_style.replace('.',"-")
    )

    folium.TileLayer(
        map_style,
        name=map_style.replace(".", " "),
        control=True,
        attr=map_style.replace(".", "-"),
    ).add_to(m)

    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in locations.iterrows():
        folium.Marker(location=[row.Latitude, row.Longitude], popup=row.name).add_to(
            marker_cluster
        )

    if save_path:
        m.save(save_path)

    return m


# TODO: debug
def upload_to_label_studio(project_id: int, annotator_kwargs: dict, top_n: int = 0):
    # import subprocess

    annotator = get_inference_engine(annotator_kwargs=annotator_kwargs)
    annotator.upload_predictions(
        project_id=project_id,
        top_n=top_n,
    )

    # script_path = "tools/upload_predictions.py"
    # args = ...
    # kwargs = {}
    # cwd = Path(__file__).parent.parent

    # cmd = ["uv run", script_path] + list(args)

    # result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=True)

    # return {
    #     "success": True,
    #     "stdout": result.stdout,
    #     "stderr": result.stderr,
    #     "returncode": result.returncode,
    # }


@st.cache_data
def get_project_statistics(
    project_id: int, annotator_id=0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # instances_count, images_count = Annotator.get_project_stats(
    #     LABEL_STUDIO_CLIENT, project_id=project_id, annotator_id=annotator_id
    # )

    load_dotenv(DOT_ENV)
    LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
    API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
    assert API_KEY is not None, "Set 'LABEL_STUDIO_API_KEY' env variable"
    labelstudio_client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
    project = labelstudio_client.projects.get(id=project_id)

    images_count = dict()
    # Iterating
    tasks = project.get_tasks()
    # because there is
    labels = []

    for task in tasks:
        try:
            result = task["annotations"][annotator_id]["result"]
        except Exception:
            traceback.print_exc()
            continue

        img_labels = []
        for annot in result:
            img_labels = annot["value"]["rectanglelabels"] + img_labels
        labels = labels + img_labels
        # update stats holder
        for label in set(img_labels):
            try:
                images_count[label] += 1
            except:
                images_count[label] = 1

    instances_count = {f"{k}": labels.count(k) for k in set(labels)}
    # print("Number of instances for each label is:\n",instances_count,end="\n\n")
    # print("Number of images for each label is:\n",images_count)

    instances_count = pd.DataFrame.from_dict(instances_count)
    instances_count.rename(
        columns={col: col + "_num_instances" for col in instances_count.columns},
        inplace=True,
    )

    images_count = pd.DataFrame.from_dict(images_count)
    images_count.rename(
        columns={col: col + "_num_images" for col in images_count.columns}, inplace=True
    )

    return instances_count, images_count


@st.cache_data
def visualize_splits_distribution(
    data_yaml_path: str,
    split="train",
):
    from tqdm import tqdm

    logger = logging.getLogger(__file__)

    # load yaml
    yolo_config = load_yaml(data_yaml_path)

    label_map = yolo_config["names"]

    path_dataset = [os.path.join(yolo_config["path"], p) for p in yolo_config[split]]

    # iter_labels = path_dataset.glob("*.txt")

    iter_labels = chain.from_iterable(
        [Path(p.replace("images", "labels")).glob("*.txt") for p in path_dataset]
    )

    # total_number_images = len(list(iter_labels))
    # path_dataset = path_dataset.replace('images','labels')
    # total_number_of_positive_images = len(list(Path(path_dataset).glob('*')))

    labels = list()
    for txtfile in tqdm(iter_labels, desc="Reading labels from data.yaml"):
        df = pd.read_csv(txtfile, sep=" ", header=None)
        df["class"] = df.iloc[:, 0].astype(int)
        df["image"] = txtfile.stem
        labels.append(df)

    df = pd.concat(labels, axis=0).reset_index(drop=True)
    df["class"] = df["class"].map(label_map)

    images_count = df.groupby("class")["image"].count().reset_index()
    instances_count = df["class"].value_counts().reset_index()

    #
    # stats = dict(num_negative=total_number_images-total_number_of_positive_images,
    #              num_positive=total_number_of_positive_images
    #              )
    stats = dict()
    stats.update({"instances_count": instances_count, "images_count": images_count})

    return stats


def start_training(
    project_id: int,
    service_url: str,
    data_config: DataConfig,
    args: TrainingConfig,
    label_config: LabelConfig = None,
    slice_data: bool = True,
):
    # ls client
    load_dotenv(DOT_ENV)
    LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
    API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
    if LABEL_STUDIO_URL is None:
        raise ValueError("env variable LABEL_STUDIO_URL is not set.")
    if API_KEY is None:
        raise ValueError("env variable API_KEY is not set.")
    labelstudio_client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

    # create dataset
    dataset = LabelingDataset.from_ls(
        project_id=project_id, max_workers=2, labelstudio_client=labelstudio_client
    )

    if slice_data:
        dataset.slice_and_save_as_yolo(data_config, label_config, max_workers=2)
    else:
        dataset.to_yolo(dir_path=data_config.dest_dir)

    train_config = {k: v for k, v in vars(args).items() if "herdnet" not in k}

    assert isinstance(args.yolo_yaml, dict), "Provide data config as dict"
    train_config["yolo_yaml"] = (train_config["yolo_yaml"], "data_config.yaml")
    # train_config["cl_data_config_yaml"] = train_config["yolo_yaml"]

    for key in [
        "yolo_arch_yaml",
    ]:
        if train_config[key]:
            p = train_config[key]
            file_name = Path(p).name
            # if key == "yolo_arch_yaml":
            #     file_name = yolo_arch_yaml
            train_config[key] = (load_yaml(p), file_name)

    if train_config["path_weights"]:
        with open(train_config["path_weights"], "rb") as file:
            weight_data = base64.b64encode(file.read()).decode("utf-8")
        train_config["path_weights"] = weight_data

    payload = dict(train_config=train_config)

    # print(json.dumps(train_config,indent=2))

    res = requests.post(url=service_url, json=payload).json()

    st.write(res)

    # print(res)
    return None


def register_model(
    weights_path: str,
    name: str = "labeler",
    export_format: str = "pt",
    imgsz: int = 800,
    batch=8,
    device: str = "cpu",
    mlflow_tracking_uri: str = "http://localhost:5000",
    dynamic: bool = False,
    task: str = "detect",
):
    import subprocess

    script_path = "tools/register_model.py"
    args = [
        "register_detector",
        weights_path,
        name,
        export_format,
        imgsz,
        batch,
        device,
        mlflow_tracking_uri,
        dynamic,
        task,
    ]

    args = [str(a) for a in args]

    cwd = Path(__file__).parent.parent

    cmd = ["python", script_path] + list(args)

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=True)

    print(result.stderr)

    return {
        "success": True,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }

    pass


if __name__ == "__main__":
    main()
