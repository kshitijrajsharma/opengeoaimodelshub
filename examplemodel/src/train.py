import argparse
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import mlflow
import psutil
import pytorch_lightning as pl
import torch
import torchvision
from inference import log_inference_example
from mlflow.models.signature import infer_signature
from model import CampDataModule, LitRefugeeCamp
from pytorch_lightning.callbacks import ModelCheckpoint
from stac2esri import create_dlpk
from utils import log_confusion_matrix

from dotenv import load_dotenv
load_dotenv()


def get_system_info() -> Dict[str, Any]:
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode()
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info.append({
                "name": name,
                "memory_total": memory_info.total,
                "memory_free": memory_info.free,
                "memory_used": memory_info.used
            })
    except Exception:
        gpu_info = []

    return {
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_info": gpu_info
    }


def calculate_dataset_statistics(data_module: CampDataModule) -> Dict[str, Any]:
    data_module.setup()
    train_loader = data_module.train_dataloader()
    
    total_samples = 0
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    
    for batch in train_loader:
        images = batch[0]
        batch_size = images.size(0)
        total_samples += batch_size
        
        images = images.view(batch_size, images.size(1), -1)
        channel_sum += images.mean(dim=2).sum(dim=0)
        channel_squared_sum += (images ** 2).mean(dim=2).sum(dim=0)
    
    mean = channel_sum / total_samples
    std = ((channel_squared_sum / total_samples) - (mean ** 2)).sqrt()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    return {
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "total_samples": total_samples,
        "image_channels": 3,
        "image_height": 256,
        "image_width": 256,
        "pixel_mean": mean.tolist(),
        "pixel_std": std.tolist(),
        "num_classes": 2
    }


def create_stac_mlm_item(
    model: LitRefugeeCamp,
    dataset_stats: Dict[str, Any],
    system_info: Dict[str, Any],
    model_performance: Dict[str, Any],
    checkpoint_path: str
) -> Dict[str, Any]:
    
    now = datetime.now()
    
    stac_item = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/mlm/v1.5.0/schema.json",
            "https://stac-extensions.github.io/file/v1.0.0/schema.json",
            "https://stac-extensions.github.io/processing/v1.1.0/schema.json"
        ],
        "id": "refugee-camp-detector-v1.0.0",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        },
        "bbox": [0, 0, 1, 1],
        "properties": {
            "datetime": now.isoformat() + "Z",
            "created": now.isoformat() + "Z",
            "updated": now.isoformat() + "Z",
            "title": "Refugee Camp Detection Model",
            "description": "Deep learning model for detecting refugee camps in satellite imagery using semantic segmentation",
            "keywords": ["refugee-camps", "humanitarian", "semantic-segmentation", "satellite-imagery", "u-net", "pytorch"],
            "license": "MIT",
            "mission": "humanitarian-response",
            "platform": "satellite",
            "instrument": "optical",
            "bands": [
                {
                    "name": "red",
                    "common_name": "red",
                    "description": "Red channel of RGB satellite imagery (630-690 nm)",
                    "center_wavelength": 0.665,
                    "full_width_half_max": 0.03
                },
                {
                    "name": "green",
                    "common_name": "green", 
                    "description": "Green channel of RGB satellite imagery (525-575 nm)",
                    "center_wavelength": 0.555,
                    "full_width_half_max": 0.03
                },
                {
                    "name": "blue",
                    "common_name": "blue",
                    "description": "Blue channel of RGB satellite imagery (450-515 nm)", 
                    "center_wavelength": 0.485,
                    "full_width_half_max": 0.03
                }
            ],
            "providers": [
                {
                    "name": "OpenGeoAIModelsHub",
                    "roles": ["producer"],
                    "url": "https://github.com/kshitijrajsharma/opengeoaimodelshub"
                }
            ],
            "mlm:name": "RefugeeCampDetector",
            "mlm:architecture": "U-Net",
            "mlm:tasks": ["semantic-segmentation"],
            "mlm:framework": "PyTorch",
            "mlm:framework_version": torch.__version__,
            "mlm:memory_size": sum(p.numel() * p.element_size() for p in model.parameters()),
            "mlm:total_parameters": sum(p.numel() for p in model.parameters()),
            "mlm:pretrained": True,
            "mlm:pretrained_source": "ImageNet",
            "mlm:accelerator": "cuda" if torch.cuda.is_available() else "cpu",
            "mlm:accelerator_constrained": False,
            "mlm:accelerator_summary": "CUDA compatible GPUs for optimal performance, CPU fallback available",
            "mlm:accelerator_count": torch.cuda.device_count() if torch.cuda.is_available() else 1,
            "mlm:batch_size_suggestion": 32,
            "mlm:input": [
                {
                    "name": "satellite_image",
                    "description": "RGB satellite image normalized with ImageNet statistics for transfer learning",
                    "bands": ["red", "green", "blue"],
                    "input": {
                        "shape": [-1, 3, 256, 256],
                        "dim_order": ["batch", "bands", "height", "width"],
                        "data_type": "float32"
                    },
                    "value_scaling": [
                        {
                            "type": "z-score",
                            "mean": 0.485,
                            "stddev": 0.229
                        },
                        {
                            "type": "z-score", 
                            "mean": 0.456,
                            "stddev": 0.224
                        },
                        {
                            "type": "z-score",
                            "mean": 0.406,
                            "stddev": 0.225
                        }
                    ]
                }
            ],
            "mlm:output": [
                {
                    "name": "segmentation_mask",
                    "description": "Binary segmentation mask identifying refugee camp areas with pixel-level predictions",
                    "tasks": ["semantic-segmentation"],
                    "result": {
                        "shape": [-1, 1, 256, 256],
                        "dim_order": ["batch", "channel", "height", "width"],
                        "data_type": "float32"
                    },
                    "classification:classes": [
                        {
                            "value": 0,
                            "name": "background",
                            "description": "Non-refugee camp areas including natural terrain, urban areas, and other land uses"
                        },
                        {
                            "value": 1,
                            "name": "refugee_camp",
                            "description": "Areas identified as refugee camps with temporary structures and settlements"
                        }
                    ]
                }
            ],
            "processing:facility": system_info.get("platform", "Unknown"),
            "processing:software": {
                "pytorch": torch.__version__,
                "lightning": pl.__version__,
                "python": system_info.get("python_version", "Unknown")
            },
            "processing:expression": {
                "format": "pytorch",
                "expression": "U-Net semantic segmentation with ImageNet pretrained backbone"
            }
        },
        "links": [
            {
                "rel": "self",
                "href": "./stac_item.json",
                "type": "application/json"
            },
            {
                "rel": "root",
                "href": "./",
                "type": "application/json"
            }
        ],
        "assets": {
            "pytorch-checkpoint": {
                "href": checkpoint_path,
                "type": "application/octet-stream; framework=pytorch-lightning",
                "title": "PyTorch Lightning Checkpoint",
                "description": "Complete model checkpoint with training state for PyTorch Lightning",
                "roles": ["mlm:model", "mlm:checkpoint"],
                "mlm:artifact_type": "pytorch_lightning",
                "file:size": Path(checkpoint_path).stat().st_size if Path(checkpoint_path).exists() else None
            },
            "pytorch-state-dict": {
                "href": "meta/best_model.pt",
                "type": "application/octet-stream; framework=pytorch",
                "title": "PyTorch TorchScript Model",
                "description": "TorchScript traced model for optimized inference",
                "roles": ["mlm:model", "mlm:weights"],
                "mlm:artifact_type": "torch.jit.trace",
                "mlm:compile_method": "jit"
            },
            "pytorch-state-dict-raw": {
                "href": "meta/best_model.pth",
                "type": "application/octet-stream; framework=pytorch", 
                "title": "PyTorch Raw State Dictionary",
                "description": "Raw PyTorch model state dictionary (.pth format)",
                "roles": ["mlm:model", "mlm:weights"],
                "mlm:artifact_type": "pytorch"
            },
            "onnx-model": {
                "href": "meta/best_model.onnx",
                "type": "application/octet-stream; framework=onnx",
                "title": "ONNX Model",
                "description": "ONNX format model for cross-platform inference and deployment",
                "roles": ["mlm:model", "mlm:inference"],
                "mlm:artifact_type": "onnx",
                "mlm:compile_method": "aot"
            },
            "source-code": {
                "href": "https://github.com/kshitijrajsharma/opengeoaimodelshub",
                "type": "text/html",
                "title": "Model Source Code Repository",
                "description": "Complete source code repository for training and inference",
                "roles": ["mlm:source_code", "code", "metadata"],
                "mlm:entrypoint": "src.inference:predict_image"
            },
            "inference-script": {
                "href": "src/inference.py",
                "type": "text/x-python",
                "title": "Enhanced Inference Script",
                "description": "Python script for running model inference with metadata tracking",
                "roles": ["mlm:source_code", "code"],
                "mlm:entrypoint": "predict_image_enhanced"
            },
            "training-script": {
                "href": "src/train.py", 
                "type": "text/x-python",
                "title": "Enhanced Training Script",
                "description": "Python script for model training pipeline with STAC-MLM metadata",
                "roles": ["mlm:training", "code"]
            },
            "container-image": {
                "href": "ghcr.io/kshitijrajsharma/opengeoaimodelshub:master",
                "type": "application/vnd.oci.image.index.v1+json",
                "title": "Docker Container Image",
                "description": "Docker container with complete runtime environment",
                "roles": ["mlm:container", "runtime"]
            },
            "requirements": {
                "href": "requirements.txt",
                "type": "text/plain", 
                "title": "Python Requirements",
                "description": "Python package requirements for running the model",
                "roles": ["runtime", "metadata"]
            },
            "esri-package": {
                "href": "meta/best_model.dlpk",
                "type": "application/zip",
                "title": "Esri Deep Learning Package", 
                "description": "Esri DLPK format for ArcGIS integration",
                "roles": ["mlm:model"],
                "mlm:artifact_type": "esri_dlpk"
            },
            "confusion-matrix": {
                "href": "meta/confusion_matrix.png",
                "type": "image/png",
                "title": "Model Evaluation Confusion Matrix",
                "description": "Model evaluation confusion matrix visualization",
                "roles": ["metadata", "overview"]
            },
            "example-input": {
                "href": "meta/example_input.png",
                "type": "image/png",
                "title": "Example Input Image",
                "description": "Example satellite image input for the model",
                "roles": ["metadata", "overview"]
            },
            "example-prediction": {
                "href": "meta/example_pred.png",
                "type": "image/png",
                "title": "Example Model Prediction",
                "description": "Example model prediction output showing detected refugee camps",
                "roles": ["metadata", "overview"]
            },
            "example-target": {
                "href": "meta/example_target.png",
                "type": "image/png",
                "title": "Example Ground Truth",
                "description": "Example ground truth segmentation mask for comparison",
                "roles": ["metadata", "overview"]
            },
            "model-metadata": {
                "href": "meta/model.emd",
                "type": "application/json",
                "title": "Esri Model Definition",
                "description": "Esri model definition file for ArcGIS Pro integration",
                "roles": ["metadata"]
            }
        }
    }
    
    return stac_item


def train_model(args):
    mlflow.set_experiment("refugee-camp-detection")
    
    with mlflow.start_run() as run:
        system_info = get_system_info()
        mlflow.log_params({
            "platform": system_info["platform"],
            "pytorch_version": system_info["pytorch_version"],
            "cuda_available": system_info["cuda_available"],
            "gpu_count": len(system_info["gpu_info"]),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "chips_dir": args.chips_dir,
            "labels_dir": args.labels_dir
        })
        
        data_module = CampDataModule(
            image_dir=args.chips_dir,
            label_dir=args.labels_dir,
            batch_size=args.batch_size
        )
        dataset_stats = calculate_dataset_statistics(data_module)
        
        mlflow.log_params({
            "train_samples": dataset_stats["train_samples"],
            "val_samples": dataset_stats["val_samples"],
            "test_samples": dataset_stats["test_samples"],
            "num_classes": dataset_stats["num_classes"]
        })
        
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="epoch={epoch}-step={step}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )
        
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            callbacks=[checkpoint_callback],
            logger=False
        )
        
        model = LitRefugeeCamp()
        trainer.fit(model, data_module)
        
        best_checkpoint = checkpoint_callback.best_model_path
        
        model_performance = {
            "best_val_loss": float(checkpoint_callback.best_model_score),
            "epochs_trained": trainer.current_epoch + 1
        }
        
        mlflow.log_metrics(model_performance)
        
        model = LitRefugeeCamp.load_from_checkpoint(best_checkpoint)
        model.eval()
        
        Path("meta").mkdir(exist_ok=True)
        
        torch.save(model.state_dict(), "meta/best_model.pth")
        
        clean_model = LitRefugeeCamp()
        clean_model.load_state_dict(model.state_dict())
        clean_model.eval()
        
        # Extract just the neural network for TorchScript tracing
        torch_model = clean_model.model
        torch_model.eval()
        traced_model = torch.jit.trace(torch_model, torch.randn(1, 3, 256, 256))
        torch.jit.save(traced_model, "meta/best_model.pt")
        
        dummy_input = torch.randn(1, 3, 256, 256)
        torch.onnx.export(
            torch_model, dummy_input, "meta/best_model.onnx",
            export_params=True, opset_version=11,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        
        # Create Esri package
        current_dir = Path(__file__).parent.absolute()
        esri_inference_path = current_dir / "esri" / "RefugeeCampDetector.py"
        emd_path = Path("meta/model.emd")
        pt_path = Path("meta/best_model.pt")
        dlpk_path = Path("meta/best_model.dlpk")
        create_dlpk(emd_path, pt_path, esri_inference_path, dlpk_path)
        
        stac_item = create_stac_mlm_item(
            clean_model, dataset_stats, system_info, model_performance, best_checkpoint
        )
        
        stac_output_path = "meta/stac_item.json"
        with open(stac_output_path, 'w') as f:
            json.dump(stac_item, f, indent=2)
        
        # Log artifacts with proper folder structure
        mlflow.log_artifact(stac_output_path, artifact_path="metadata")
        mlflow.log_artifact("meta/best_model.pth", artifact_path="models")
        mlflow.log_artifact("meta/best_model.pt", artifact_path="models")
        mlflow.log_artifact("meta/best_model.onnx", artifact_path="models")
        mlflow.log_artifact("meta/best_model.dlpk", artifact_path="models")
        mlflow.log_artifact(best_checkpoint, artifact_path="checkpoints")
        
        # Log training dataset structure
        mlflow.log_artifact(args.chips_dir, artifact_path="datasets/train/chips")
        mlflow.log_artifact(args.labels_dir, artifact_path="datasets/train/labels")
        
        log_inference_example(model, data_module)
        log_confusion_matrix(model, data_module, run)
        
        signature = infer_signature(dummy_input.numpy(), clean_model(dummy_input).detach().numpy())
        mlflow.pytorch.log_model(
            clean_model, "model",
            signature=signature,
            extra_files=[stac_output_path]
        )
        
        print(f"Training completed. STAC-MLM item saved to: {stac_output_path}")
        print(f"MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chips_dir", type=str, default="data/train/sample/chips")
    parser.add_argument("--labels_dir", type=str, default="data/train/sample/labels")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train_model(args)
