from .predictor import MyModelAPI
import litserve as ls
import torch


def run_inference_server(port=4141, max_batch_size=1, workers_per_device=1):
    device = "cuda" if torch.cuda.is_available() else "auto"

    api = MyModelAPI(max_batch_size=max_batch_size, enable_async=True)

    server = ls.LitServer(
        api,
        max_batch_size=max_batch_size,
        workers_per_device=workers_per_device,
        accelerator=device,
    )
    server.run(port=port, generate_client_file=True)
