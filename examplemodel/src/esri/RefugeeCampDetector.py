import json
import os
from pathlib import Path

import numpy as np

try:
    import torch

    HAS_PYTORCH = True
except:
    HAS_PYTORCH = False

SCRIPT_DIR = Path(__file__).parent
LOG_PATH = SCRIPT_DIR / "rcd_debug.log"


def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


class RefugeeCampDetector:
    def __init__(self):
        log("Shree Ganeshaya Namah")
        self.name = "Refugee Camp Detector"
        self.description = "Classifies pixels as refugee camp or background"

    def initialize(self, **kwargs):
        log(f"INITIALIZE {kwargs}")
        model = kwargs.get("model")
        if not model:
            log("NO MODEL")
            return
        try:
            with open(model, "r") as f:
                self.json_info = json.load(f)
                log("LOADED JSON FILE")
        except FileNotFoundError:
            self.json_info = json.loads(model)
            log("LOADED JSON STRING")
        model_path = self.json_info["ModelFile"]
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(
                os.path.join(os.path.dirname(model), model_path)
            )
        self.model_path = model_path
        self.model_is_loaded = False
        log(f"MODEL PATH {self.model_path}")

    def load_model(self):
        log("LOAD_MODEL")
        if not HAS_PYTORCH:
            log("PYTORCH MISSING")
            raise Exception("PyTorch not installed")
        try:
            self.model = torch.jit.load(self.model_path, map_location="cpu")
            self.model.eval()
            self.model_is_loaded = True
            log(f"MODEL LOADED TYPE {type(self.model)}")
        except Exception as e:
            log(f"LOAD ERROR {e}")
            raise

    def getParameterInfo(self):
        return [
            {"name": "raster", "dataType": "raster", "required": True},
            {"name": "model", "dataType": "string", "required": True},
            {
                "name": "batch_size",
                "dataType": "numeric",
                "required": False,
                "value": self.json_info.get("BatchSize", 1),
            },
            {
                "name": "threshold",
                "dataType": "numeric",
                "required": False,
                "value": self.json_info.get("Threshold", 0.5),
            },
        ]

    def getConfiguration(self, **scalars):
        bs = int(scalars.get("batch_size", self.json_info.get("BatchSize", 1)))
        thr = float(scalars.get("threshold", self.json_info.get("Threshold", 0.5)))
        tx = self.json_info["ImageWidth"]
        ty = self.json_info["ImageHeight"]
        bands = tuple(self.json_info.get("ExtractBands", [1, 2, 3]))
        return {
            "batch_size": bs,
            "tx": tx,
            "ty": ty,
            "extractBands": bands,
            "dataRange": tuple(self.json_info.get("DataRange", [0, 1])),
            "inputMask": True,
            "inheritProperties": 2 | 4 | 8,
        }

    def updateRasterInfo(self, **kwargs):
        kwargs["output_info"]["bandCount"] = 1
        kwargs["output_info"]["pixelType"] = "u1"
        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        log(f"UPDATEPIXELS tile={tlc} shape={shape}")
        mask = pixelBlocks["raster_mask"]
        pix = pixelBlocks["raster_pixels"]  # (3, 256, 256)
        pix[mask == 0] = 0
        if not self.model_is_loaded:
            self.load_model()
        arr = pix.astype(np.float32)  # (3, 256, 256)
        log(f"ARRAY shape before batching: {arr.shape}")
        if arr.ndim == 3:
            arr = arr[np.newaxis]  # (1, 3, 256, 256)
        log(f"ARRAY shape after batching: {arr.shape}")
        bands = self.json_info.get("ExtractBands", [0, 1, 2])
        idx = [b if min(bands) == 0 else b - 1 for b in bands]
        arr = arr[:, idx, :, :]  # (1, 3, 256, 256)
        log(f"ARRAY shape after band selection hai guys : {arr.shape}")
        if arr.max() > 1.0:
            arr = arr / 255.0  # (1, 3, 256, 256)
        mp = self.json_info.get("ModelParameters", {})
        mean = np.array(mp.get("mean", [0.485, 0.456, 0.406]))[None, :, None, None]
        std = np.array(mp.get("std", [0.229, 0.224, 0.225]))[None, :, None, None]
        arr = (arr - mean) / std  # (1, 3, 256, 256)
        log(f"ARRAY shape after normalization: {arr.shape}")
        tensor = torch.from_numpy(arr).float()  # (1, 3, 256, 256)
        log(
            f"TENSOR shape={tuple(tensor.shape)} dtype={tensor.dtype} numel={tensor.numel()}"
        )
        with torch.no_grad():
            out = self.model(tensor)
        log(f"OUTPUT type={type(out)} shape={tuple(out.shape)}")
        if out.shape[1] == 1:
            preds = (torch.sigmoid(out) > self.json_info.get("Threshold", 0.5)).float()
        else:
            preds = torch.argmax(torch.softmax(out, 1), 1, keepdim=True).float()
        res = preds.cpu().numpy()  # shape: (1, 1, 256, 256)
        log(f"PREDICTIONS shape={res.shape} & dim = {res.ndim}")
        res = res[0, 0]  # (H, W)
        res = res.astype(np.uint8)
        pixelBlocks["output_pixels"] = res
        log(
            f"OUTPUT PIXELS shape= aba yo aayena vane i will be mad {pixelBlocks['output_pixels'].shape}"
        )
        return pixelBlocks
