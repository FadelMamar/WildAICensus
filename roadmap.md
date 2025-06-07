# ML Microservices Monorepo Implementation Roadmap

This document outlines a 6-week plan (16 hours/week) to consolidate existing services and build new components in a single GitHub repository.
> https://chatgpt.com/c/68120a64-a1d4-8001-a6a3-a7dcc81de7bb
> https://chat.deepseek.com/a/chat/s/869b5735-78fb-4d25-9d0b-372d469ead66
> https://chatgpt.com/c/68114057-e398-8001-89f3-86cf4e61d097

## 📁 Monorepo Structure
```
plain
ml_microservices/
├── .github/                   
│   └── workflows/               # CI / workflows
|
|
├── gateway/                    # Entry point for frontend and orchestration
|
|
├── shared/                      # Shared components between services
│   ├── models/
│   │   ├── schemas.py
│   │   └── config.py
│   ├── utils/
│   │   └── logger.py
│   └── setup.py                 # Installable as a local package
│
|
├── inference_service/           # Handles model predictions
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   └── endpoints.py
│   │   ├── services/
│   │   │   └── predictor.py
│   │   ├── models/
│   │   │   └── schemas.py
│   │   └── core/
│   │       ├── data_loader.py
│   │       └── preprocessing.py
│   └── Dockerfile
│   └── Logs → Prometheus
│   └── Sends inputs/stats → EvidentlyAI
|
|
yolo_baal_backend/
├── main.py               # FastAPI app
├── config.py             # Config management
├── model/
│   ├── __init__.py
│   ├── yolo_baal.py      # Model wrapper with YOLOv8 + Baal
│   └── trainer.py        # Training loop
├── data/
│   ├── loader.py         # Dataset I/O (e.g., from Label Studio export)
│   └── transform.py      # Image transforms
├── utils/
│   ├── logger.py         # Structured logging
│   └── ls_interface.py   # Label Studio formatting helpers
├── requirements.txt
└── README.md
|
|
├── labeling_service/          
│   └── Sends retraining trigger → Training Service
|
|
├── dashboard/                 # ← Streamlit;
│
|
├── training_service/            # Handles training and retraining
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   └── endpoints.py
│   │   ├── services/
│   │   │   └── trainer.py
│   │   ├── models/
│   │   │   ├── config.py
│   │   │   └── schemas.py
│   │   └── core/
│   │       ├── mlflow_wrapper.py
│   │       └── annotation_converter.py
│   └── Dockerfile
│   └── Logs metrics → Prometheus
│   └── Sends data snapshots → EvidentlyAI|
|
|
calibration_service/
├── app/
│   ├── api/
│   │   └── endpoints.py           # POST /calibrate
│   ├── services/
|   │   ├── calibrator.py          # Coordinates multiple strategies
|   │   ├── fp_classifier.py       # Trains & evaluates FP filter
|   │   └── evaluator.py           # Calculates precision/recall before/after filtering
│   ├── models/
│   │   └── schemas.py            # Pydantic request/response models
│   └── main.py                   # FastAPI app
|   ├── Dockerfile
|
|
├── data_service/               # Preprocessing, annotation formatting
|
|
├── mlflow_service/
│   ├── app/
│   │   └── mlflow_server.sh     # Entrypoint to launch tracking server
│   ├── mlruns/                  # Local volume (if not using MinIO backend)
│   └── Dockerfile
|
|
├── storage/                   # MinIO compose
│   └── Used as MLflow artifact store
│   └── docker-compose config or manifest
|
|
├── database/                  # Weaviate compose
│   └── docker-compose config or manifest
|
|
├── monitoring/                       # Prometheus, Grafana, Evidently
│   ├── prometheus/                   # Metrics collection
│   ├── grafana/                      # Visualization
│   └── evidently_service/            # Model/data drift monitoring
│       └── REST or shared volume reports
│
└── docker-compose.yml         # Orchestrates all services
└── README.md
```

**Internal Communication Plan**
| From                   | To                     | Method                         | Purpose                            |
|------------------------|------------------------|----------------------------------|-------------------------------------|
| **Dashboard**          | Gateway                | HTTP (REST)                     | Frontend requests                   |
| **Gateway**            | Other Services         | REST or gRPC                    | API routing                         |
| **Labeling Backend**   | Label Studio API       | REST (Label Studio SDK)         | Pull new annotations                |
| **Labeling Backend**   | Training Service       | REST or Message Queue (e.g., Celery, Kafka) | Trigger retraining    |
| **Inference Service**  | Database               | Direct DB/API                   | Log/query predictions               |
| **Training Service**   | MinIO                  | S3 API (boto3, MinIO SDK)       | Save/load datasets and models       |
| **Training Service**   | Database               | Direct DB/API                   | Store vectors, metadata             |
| **Data Service**       | MinIO, Label Studio    | S3 + REST                       | Upload raw/annotated data           |
| **All services**       | Shared config/models   | Mounted or installed dependency | Common data structures              |



---

# 🗓 6-Week Roadmap (16 h/week)

### Week 1 – Monorepo & Dev-Ops Setup

**Goals:** Consolidate repos, scaffold services, set up CI

- Consolidate existing `inference_service`, `labeling_service`, and `dashboard` into one repo
- Create top-level `docker-compose.yml` to bring up:
  - Inference, Labeling, Dashboard
  - MinIO, Weaviate
- Scaffold directories for new services: `training_service`, `calibration_service`, `mlflow_service`, `monitoring`
  - Add empty `main.py` and Dockerfile in each
- Setup GitHub Actions to lint, format, and build all Docker images

> **Milestone:** Single repo builds all services and infra containers.

### Week 2 – Training  Service

**Goals:** Implement training logic, integrate MLflow

- Develop `training_service`:
  - YOLOv8 training wrapper in `trainer.py`
  - REST endpoint to start training jobs
  - Log runs 
  - Export and register model weights 

> **Milestone:** Train a model, view it in MLflow UI, and store artifacts in MinIO.

### Week 3 – Calibration Service & FP Classifier

**Goals:** Build calibration logic and integrate with MLflow

- Create `calibration_service`:
  - API endpoint `/calibrate`
  - Implement strategies: threshold/NMS tuning, FP classifier training in `fp_classifier.py`
  - Log calibrated model versions in MLflow
- Update `inference_service` to load “Production” or “Calibrated” model from MLflow

> **Milestone:** Calibrate a registered model, register the calibrated version, and serve it.

### Week 4 – Database & Storage Enhancements

**Goals:** Wire up Weaviate and secure MinIO access

- Instrument `inference_service` to push embeddings and prediction metadata into Weaviate
- Ensure `training_service` writes metadata and model vectors to Weaviate or another structured store
- Centralize MinIO credentials and buckets in shared config
- Add health-check endpoints for MinIO and Weaviate

> **Milestone:** Full data flow: inference ↔ Weaviate, training artifacts in MinIO.

### Week 5 – Monitoring Stack

**Goals:** Implement infrastructure and model monitoring

- Deploy Prometheus & Grafana in `monitoring/`:
  - Instrument FastAPI apps with `prometheus_fastapi_instrumentator`
  - Create Grafana dashboards for throughput, latency, and error rates
- Stand up EvidentlyAI:
  - Stream inference data into Evidently for drift/quality reports
  - Expose HTML/JSON reports via shared volume or REST endpoint
- Surface Grafana/Evidently links in the Streamlit dashboard

> **Milestone:** Collect infra metrics and model/data drift stats with visualization.

### Week 6 – Active-Learning Loop & Final Polish

**Goals:** Close the feedback loop and document

- Extend labeling backend to orchestrate active learning:
  1. Pull unlabeled samples from storage or Weaviate
  2. Trigger new labeling round in Label Studio
  3. On completion, invoke training → calibration workflows
- Perform an end-to-end test:
  ```plain
  Ingestion → Labeling → Training → Calibration → Inference → Monitoring
  ```
- Write developer documentation, `README.md`, and a `Makefile` or CLI:
  - `make up` / `make down`
  - `make train` / `make calibrate`

> **Milestone:** A self-contained monorepo enabling full ML pipeline with observability.
````
graph TD
    A[Initial Dataset] --> B[Train YOLO]
    B --> C[Predict with Uncertainty]
    C --> D[Select Uncertain Samples]
    D --> E[Label in Label Studio]
    E --> F[Retrain Model]
    F --> C
````
