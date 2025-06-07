from src.inference_service.endpoints import run_inference_server
import os


if __name__ == "__main__":
    run_inference_server(
        port=int(os.environ.get("INFERENCE_PORT", 4141)),
        workers_per_device=int(os.environ.get("NUM_WORKERS", 1)),
        max_batch_size=int(os.environ.get("BATCH_SIZE", 1)),
    )
