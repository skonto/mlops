import torch
import numpy as np
from typing import List, Dict
from ts.torch_handler.base_handler import BaseHandler
import json

from models.iris import IrisDL
from models.torch_model_config import ModelParams

class IrisHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = properties.get("model_dir")
        model_path = f"{model_dir}/model.pt"

        model_params_config = ModelParams()
        model_params = model_params_config.to_model_dict()
        self.model = IrisDL(**model_params)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        inputs = []
        for row in data:
            raw = row.get("body") or row.get("data") or row
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON string in request body.")
            if not isinstance(raw, list) or not all(isinstance(inner, list) for inner in raw):
                raise ValueError("Input must be a list of lists of floats.")
            inputs.extend(raw)

        array = np.asarray(inputs, dtype=np.float32)
        return torch.from_numpy(array).to(self.device)

    def inference(self, inputs: torch.Tensor) -> List:
        with torch.no_grad():
            outputs = self.model(inputs)
            preds = torch.argmax(outputs, dim=1)
        return preds.cpu().tolist()

    def postprocess(self, outputs: List[int]) -> List[List[int]]:
        return [[p for p in outputs]]
