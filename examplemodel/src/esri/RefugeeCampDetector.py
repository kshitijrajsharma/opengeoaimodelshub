import json
import os
import pathlib
import sys

import numpy as np

try:
    import torch

    HAS_PYTORCH = True
except Exception:
    HAS_PYTORCH = False


class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4


class RefugeeCampDetector:
    def __init__(self):
        self.name = "RefugeeCampDetector"
        self.description = "Detects refugee camp/buildings in imagery as polygons"

    def initialize(self, **kwargs):
        if "model" not in kwargs:
            return
        model = kwargs["model"]
        model_as_file = True
        try:
            with open(model, "r") as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            self.json_info = json.loads(model)
            model_as_file = False
        model_path = self.json_info["ModelFile"]
        if model_as_file and not os.path.isabs(model_path):
            model_path = os.path.abspath(
                os.path.join(os.path.dirname(model), model_path)
            )
        self.model_path = model_path
        self.model_is_loaded = False

    def load_model(self):
        if not HAS_PYTORCH:
            raise Exception("PyTorch is not installed.")
        self.model = torch.jit.load(self.model_path, map_location="cpu")
        self.model.eval()

    def getParameterInfo(self):
        required_parameters = [
            {
                "name": "raster",
                "dataType": "raster",
                "required": True,
                "displayName": "Raster",
                "description": "Input Raster",
            },
            {
                "name": "model",
                "dataType": "string",
                "required": True,
                "displayName": "Input Model Definition (EMD) File",
                "description": "Input model definition (EMD) JSON file",
            },
            {
                "name": "device",
                "dataType": "numeric",
                "required": False,
                "displayName": "Device ID",
                "description": "Device ID",
            },
            {
                "name": "batch_size",
                "dataType": "numeric",
                "required": False,
                "value": 1
                if "BatchSize" not in self.json_info
                else int(self.json_info["BatchSize"]),
                "displayName": "Batch Size",
                "description": "Batch Size",
            },
            {
                "name": "threshold",
                "dataType": "numeric",
                "required": False,
                "value": 0.5
                if "Threshold" not in self.json_info
                else float(self.json_info["Threshold"]),
                "displayName": "Confidence Score Threshold [0.0, 1.0]",
                "description": "Confidence score threshold value [0.0, 1.0]",
            },
        ]
        return required_parameters

    def getConfiguration(self, **scalars):
        self.batch_size = int(
            scalars.get("batch_size", self.json_info.get("BatchSize", 1))
        )
        self.thres = float(
            scalars.get("threshold", self.json_info.get("Threshold", 0.5))
        )
        self.ImageHeight = self.json_info["ImageHeight"]
        self.ImageWidth = self.json_info["ImageWidth"]
        configuration = {
            "batch_size": self.batch_size,
            "tx": self.ImageWidth,
            "ty": self.ImageHeight,
            "extractBands": tuple(self.json_info.get("ExtractBands", [1, 2, 3])),
            "dataRange": tuple(self.json_info.get("DataRange", [0, 1])),
            "inputMask": True,
            "inheritProperties": 2 | 4 | 8,
        }
        self.scalars = scalars
        return configuration

    def getFields(self):
        fields = [
            {"name": "OID", "type": "esriFieldTypeOID", "alias": "OID"},
            {"name": "Classname", "type": "esriFieldTypeString", "alias": "Classname"},
            {
                "name": "Classvalue",
                "type": "esriFieldTypeInteger",
                "alias": "Classvalue",
            },
            {
                "name": "Confidence",
                "type": "esriFieldTypeDouble",
                "alias": "Confidence",
            },
        ]
        return json.dumps(fields)

    def getGeometryType(self):
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):
        raster_mask = pixelBlocks["raster_mask"]
        raster_pixels = pixelBlocks["raster_pixels"]
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks["raster_pixels"] = raster_pixels
        if not self.model_is_loaded:
            self.load_model()
            self.model_is_loaded = True
        input_images = pixelBlocks["raster_pixels"]
        batch = input_images[np.newaxis] if input_images.ndim == 3 else input_images
        batch = batch.astype(np.float32) / 255.0
        batch = np.transpose(batch, (0, 3, 1, 2))
        batch_tensor = torch.from_numpy(batch)
        with torch.no_grad():
            logits = self.model(batch_tensor)
            if logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=1)
                masks = probs[:, 1] > self.thres
            else:
                probs = torch.sigmoid(logits)
                masks = probs > self.thres
            masks = masks.cpu().numpy()
        features = {"features": []}
        for idx, mask in enumerate(masks):
            from skimage import measure

            mask_bin = (mask > 0).astype(np.uint8)
            contours = measure.find_contours(mask_bin, 0.5)
            for i, contour in enumerate(contours):
                contour = np.flip(contour, axis=1)
                rings = [[[float(x), float(y)] for x, y in contour]]
                features["features"].append(
                    {
                        "attributes": {
                            "OID": i + 1,
                            "Classname": "refugee_camp",
                            "Classvalue": 1,
                            "Confidence": float(np.mean(mask_bin)),
                        },
                        "geometry": {"rings": rings},
                    }
                )
        return {"output_vectors": json.dumps(features)}

    def inference(self, batch, **scalars):
        pass
