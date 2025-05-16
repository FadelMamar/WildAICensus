from inference_service.endpoints import run_inference_server
import os


if __name__ == "__main__":
    run_inference_server(
        port=os.environ.get("INFERENCE_PORT", 4141),
        workers_per_device=os.environ.get("NUM_WORKERS", 2),
        max_batch_size=1,
    )
