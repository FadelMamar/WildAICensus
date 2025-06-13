from trainer import TrainingConfig, TrainingManager
import litserve as ls
from fastapi import HTTPException
import logging
import json
import traceback
import yaml
import base64

logger = logging.getLogger("Trainer")


def save_yaml(cfg: dict, save_path: str, mode="w"):
    with open(save_path, mode, encoding="utf-8") as file:
        yaml.dump(cfg, file)


class TrainingAPI(ls.LitAPI):
    def setup(self, device):
        return

    def decode_request(self, request: dict) -> TrainingConfig:
        try:
            args = request["train_config"]
            for key in ["yolo_yaml", "yolo_arch_yaml"]:
                arg = args.pop(key)
                if arg is None:
                    continue
                elif isinstance(arg, list):
                    arg, save_path = arg
                    save_yaml(arg, save_path=save_path)
                    args[key] = save_path
                else:
                    print(arg)
                    raise ValueError()

            if args.get("path_weights"):
                weight_data = base64.b64decode(args.get("path_weights"))
                with open("weights.pt", "wb") as file:
                    file.write(weight_data)
                args["path_weights"] = "weights.pt"

            TrainingConfig(**args)  # attempting to load

            logger.info("Training args:\n" + json.dumps(args, indent=2))

            return args

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))

    def predict(self, x: dict) -> dict:
        try:
            cfg = TrainingConfig(**x)
            trainer = TrainingManager(
                args=cfg,
            )
            trainer.run()
            return {"status": "success"}
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    def encode_response(self, output: dict):
        return output


if __name__ == "__main__":
    api = TrainingAPI(api_path="/train")

    server = ls.LitServer(
        api,
    )
    server.run(port=5500, generate_client_file=False)
