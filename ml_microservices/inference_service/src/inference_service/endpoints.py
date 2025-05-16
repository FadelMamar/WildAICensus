from .predictor import MyModelAPI
import litserve as ls


def run_inference_server(port=4141, max_batch_size=1, workers_per_device=2):
    api = MyModelAPI()
    server = ls.LitServer(
        api, max_batch_size=max_batch_size, workers_per_device=workers_per_device
    )
    server.run(port=port)
