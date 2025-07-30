import argparse
import json
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import onnx
import pystac
import pytorch_lightning as pl
import torch
from inference import log_inference_example
from mlflow.models.signature import infer_signature
from model import CampDataModule, LitRefugeeCamp
from pytorch_lightning.callbacks import ModelCheckpoint
from stac2esri import create_dlpk, stacmlm_to_emd
from stac_model.input import InputStructure, ModelInput
from stac_model.output import MLMClassification, ModelOutput, ModelResult
from stac_model.schema import MLModelExtension, MLModelProperties
from utils import log_confusion_matrix

from dotenv import load_dotenv

load_dotenv() # loads env from .env for the s3 configuration to log artifacts into the server

def create_stac_mlm_item(model, checkpoint_path, onnx_path):
    input_struct = InputStructure(
        shape=[-1, 3, 256, 256],
        dim_order=["batch", "channel", "height", "width"],
        data_type="float32",
    )

    model_input = ModelInput(
        name="satellite_image",
        bands=["red", "green", "blue"],
        input=input_struct,
        resize_type="interpolation-linear",
        description="RGB satellite image normalized with ImageNet statistics",
    )

    result_struct = ModelResult(
        shape=[-1, 1, 256, 256],
        dim_order=["batch", "channel", "height", "width"],
        data_type="float32",
    )
    classes = [
        MLMClassification(value=0, name="background"),
        MLMClassification(value=1, name="refugee_camp"),
    ]
    model_output = ModelOutput(
        name="segmentation_mask",
        tasks={"semantic-segmentation"},
        result=result_struct,
        description="Binary segmentation mask for refugee camp detection",
        classes=classes,
    )

    assets = {
        "pytorch-checkpoint": pystac.Asset(
            title="PyTorch Lightning Checkpoint",
            description="Trained U-Net model checkpoint for refugee camp detection",
            href=checkpoint_path,
            media_type="application/octet-stream",
            roles=["mlm:model", "mlm:checkpoint"],
            extra_fields={"mlm:artifact_type": "pytorch_lightning"},
        ),
        "pytorch-state-dict": pystac.Asset(
            title="PyTorch State Dict",
            description="PyTorch model state dictionary for refugee camp detection",
            href="meta/best_model.pt",
            media_type="application/octet-stream",
            roles=["mlm:model", "mlm:weights"],
            extra_fields={"mlm:artifact_type": "pytorch"},
        ),
        "onnx-model": pystac.Asset(
            title="ONNX Model",
            description="ONNX format model for inference",
            href=onnx_path,
            media_type="application/octet-stream",
            roles=["mlm:model", "mlm:inference"],
            extra_fields={"mlm:artifact_type": "onnx"},
        ),
    }

    accelerator = "cuda" if torch.cuda.is_available() else "amd64"

    ml_model_meta = MLModelProperties(
        name="RefugeeCampDetector",
        architecture="U-Net",
        tasks={"semantic-segmentation"},
        framework="pytorch",
        framework_version=torch.__version__,
        accelerator=accelerator,
        accelerator_constrained=False,
        memory_size=sum(p.numel() * p.element_size() for p in model.parameters()),
        total_parameters=sum(p.numel() for p in model.parameters()),
        input=[model_input],
        output=[model_output],
    )

    bbox = [-180, -90, 180, 90]
    geometry = {
        "type": "Polygon",
        "coordinates": [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]],
    }

    item = pystac.Item(
        id="refugee-camp-detector",
        geometry=geometry,
        bbox=bbox,
        datetime=datetime.now(),
        properties={
            "description": "U-Net model for detecting refugee camps in satellite imagery"
        },
        assets=assets,
    )

    item_mlm = MLModelExtension.ext(item, add_if_missing=True)
    item_mlm.apply(
        ml_model_meta.model_dump(
            by_alias=True, exclude_unset=True, exclude_defaults=True
        )
    )

    return item


def export_to_onnx(model, output_path, input_shape=(1, 3, 256, 256)):
    dummy_input = torch.randn(input_shape, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        dynamo=True,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported successfully to {output_path}")
    print(f"Input shape: {input_shape}")
    print(f"Expected output shape: (batch_size, 1, 256, 256)")


def main(args):
    mlflow.autolog(log_models=True)
    print(f"Using MLflow URI: {args.mlflow_uri}")
    mlflow.set_tracking_uri(args.mlflow_uri)
    # mlflow.set_experiment(args.experiment)
    mlflow.enable_system_metrics_logging()

    dm = CampDataModule(args.chips_dir, args.labels_dir, batch_size=args.batch_size)
    dm.setup()

    mlflow.log_artifacts(args.chips_dir, artifact_path="dataset/chips")
    mlflow.log_artifacts(args.labels_dir, artifact_path="dataset/labels")

    checkpoint = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath="checkpoints",
    )

    # with open("training.log", "w") as f, redirect_stdout(f), redirect_stderr(f):
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint],
        log_every_n_steps=3,
    )
    model = LitRefugeeCamp(lr=args.lr)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    if checkpoint.best_model_path:
        mlflow.log_artifact(checkpoint.best_model_path, artifact_path="checkpoints")
        model_to_log = LitRefugeeCamp.load_from_checkpoint(
            checkpoint_path=checkpoint.best_model_path, map_location="cpu"
        ).model.cpu()

        sample_input_numpy = np.random.randn(1, 3, 256, 256).astype(np.float32)

        # Infer model signature
        model_to_log.eval()
        with torch.no_grad():
            sample_input_tensor = torch.from_numpy(sample_input_numpy).cpu()
            sample_output = model_to_log(sample_input_tensor)
            sample_output_numpy = sample_output.cpu().numpy()

        signature = infer_signature(sample_input_numpy, sample_output_numpy)

        onnx_path = "meta/best_model.onnx"
        pt_path = "meta/best_model.pt"
        pth_path = "meta/best_model.pth"
        current_dir = Path(__file__).parent.absolute()
        esri_inference_path = current_dir / "esri" / "RefugeeCampDetector.py"
        export_to_onnx(model_to_log, onnx_path)

        sample_input_tensor = torch.from_numpy(sample_input_numpy).cpu()
        traced_model = torch.jit.trace(model_to_log, sample_input_tensor).cpu()
        traced_model.save(pt_path)

        torch.save(model_to_log.state_dict(), pth_path)

        stac_item = create_stac_mlm_item(
            model_to_log, checkpoint.best_model_path, onnx_path
        )

        stac_json_path = "meta/stac_item.json"
        with open(stac_json_path, "w") as f:
            json.dump(stac_item.to_dict(), f, indent=2)
        emd_path = stacmlm_to_emd(Path(stac_json_path), Path("meta"))
        dlpk_path = Path("meta") / "best_model.dlpk"
        create_dlpk(emd_path, pt_path, esri_inference_path, dlpk_path)

        mlflow.log_artifact(dlpk_path, artifact_path="esri")
        mlflow.log_artifact(emd_path, artifact_path="esri")

        mlflow.log_artifact(stac_json_path, artifact_path="metadata")
        mlflow.log_artifact(onnx_path, artifact_path="models")
        mlflow.log_artifact(pt_path, artifact_path="models")
        mlflow.log_artifact(pth_path, artifact_path="models")

        mlflow.pytorch.log_model(
            model_to_log,
            name="model",
            registered_model_name=args.model_name,
            input_example=sample_input_numpy,
            signature=signature,
        )

    log_confusion_matrix(model.model, dm, mlflow)
    log_inference_example(model.model, dm, mlflow)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_uri", type=str, default="http://localhost:5000")
    parser.add_argument("--experiment", type=str, default="RefugeeCampDetection3")
    parser.add_argument("--model_name", type=str, default="RefugeeCampDetector")
    parser.add_argument("--chips_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
