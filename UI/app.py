import streamlit as st
import pandas as pd
from typing import List
import requests
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
from datalabeling.ml.train import ImageClassifier

from datalabeling.common.processor import get_processor, DetectionsPostprocessor
from datalabeling.common.config import TilingConfig
from datalabeling.common.dataset_loader import LabelingDataset
from datalabeling.common.config import PredictionConfig
from datalabeling.ml.interface import Annotator


DOT_ENV = Path(__file__) / "../.env"
load_dotenv(DOT_ENV)

# LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL",'http://localhost:8080')
# LABEL_STUDIO_API_KEY = os.environ.get("LABEL_STUDIO_API_KEY")

# LABEL_STUDIO_CLIENT = LabelStudio(
#     base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY
# )

# TRAINING_API_URL = ...


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
        st.header("Train Object Detector")
        with st.form("model_training"):
            training_projects = st.text_input("Project IDs (comma-separated)")
            epochs = st.slider("Training Epochs", 1, 100, 10)
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64])

            if st.form_submit_button("Start Training"):
                raise NotImplementedError

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
                "overlapfactor", min_value=0.0, max_value=0.9
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
                "sensor_height in [mm]",
                min_value=0.0,
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
                "Path to images directory (without quotes)",
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


def get_annotator(annotator_kwargs: dict):
    config = PredictionConfig(
        imgsz=annotator_kwargs.get("tilesize", 800),
        tilesize=annotator_kwargs.get("tilesize", 800),
        overlap_ratio=annotator_kwargs.get("overlap_ratio", 0.2),
        confidence_threshold=annotator_kwargs.get("confidence_threshold", 0.2),
        # min_area=100,
        # max_area=None,
        cls_imgsz=96,
        # device='cuda'
    )

    # get image classifier
    path = r"D:\datalabeling\src\tests\runs-classifier\best-v6.ckpt"
    model = ImageClassifier.load_from_checkpoint(
        path, cls_is_features=True, map_location="cpu"
    )
    handler = get_processor("classifier")(
        model,
        label_map={0: "gt", 1: "tn"},
        device=config.device,
        feature_extractor=get_processor("feature_extractor")(),
        imgsz=config.cls_imgsz,
    )

    # build postprocessor
    processor = DetectionsPostprocessor(
        keep_classes=["gt"],
    )
    processor.set_handler(handler)

    # get annotator
    # path_to_weights = annotator_kwargs.get('path_to_weights')
    # if not Path(path_to_weights).exists():
    path_to_weights = None

    annotator = Annotator(
        config=config,
        dotenv_path=DOT_ENV,
        path_to_weights=path_to_weights,
        mlflow_model_alias=annotator_kwargs.get("mlflow_model_alias"),
        mlflow_model_name=annotator_kwargs.get("mlflow_model_name"),
    )

    annotator.set_processor(image_processor=None, detection_processor=processor)

    return annotator


@st.cache_data
def run_inference(
    image_dir: str,
    annotator_kwargs: dict,
    save_path: str = None,
    image_paths: list[None] = None,
    exts: list[str] = [
        "*.jpg",
        "*.jpeg",
        "*.png",
    ],
) -> None:
    raise NotImplementedError
    handler = get_annotator(annotator_kwargs=annotator_kwargs)

    exts = [e.lower() for e in exts] + [e.capitalize() for e in exts]

    if image_paths is None:
        image_paths = chain.from_iterable([Path(image_dir).glob(ext) for ext in exts])

    results = handler.batch_inference(
        images_paths=image_paths,
        save_path=save_path,
        as_dataframe=True,
        inference_service_url=annotator_kwargs.get("inference_service_url", None),
    )

    if save_path:
        results[["Latitude", "Longitude", "Elevation"]].to_csv(save_path, index=False)

    return results


@st.cache_data
def get_gps_coords(
    image_dir: str,
    image_paths: list[str] = None,
    exts: list[str] = [
        "*.jpg",
        "*.jpeg",
        "*.png",
    ],
):
    exts = [e.lower() for e in exts] + [e.capitalize() for e in exts]

    if image_paths is None:
        image_paths = chain.from_iterable([Path(image_dir).glob(ext) for ext in exts])

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
    labelstudio_client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
    # print(API_KEY)
    # check connection
    project = labelstudio_client.projects.get(id=project_id)

    print(config)

    if config.root is None:
        data_dir = labelstudio_client.import_storage.local.get(project_id).path
        print(f"Loading data from {data_dir}")

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


# Mock API client functions (implement according to your API specs)
def upload_to_label_studio(project_id: int, annotator_kwargs: dict, top_n: int = 0):
    annotator = get_annotator(annotator_kwargs=annotator_kwargs)

    annotator.upload_predictions(
        project_id=project_id, top_n=top_n, download_resources=False
    )


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
    labelstudio_client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
    project = labelstudio_client.get_project(id=project_id)

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


def start_training(project_ids: List[int], epochs: int, batch_size: int, token: str):
    """Mock function for training initialization"""
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"project_ids": project_ids, "epochs": epochs, "batch_size": batch_size}
    return requests.post(f"{TRAINING_API_URL}/train", headers=headers, json=payload)


if __name__ == "__main__":
    main()
