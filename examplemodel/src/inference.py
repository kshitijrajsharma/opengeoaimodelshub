import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2


def load_stac_item(stac_path: str) -> Dict[str, Any]:
    with open(stac_path, 'r') as f:
        return json.load(f)


def create_inference_metadata(
    image_path: str,
    model_info: Dict[str, Any],
    inference_time: float,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...]
) -> Dict[str, Any]:
    return {
        "inference_timestamp": datetime.now().isoformat() + "Z",
        "input_image_path": str(image_path),
        "model_name": model_info.get("mlm:name", "unknown"),
        "model_version": model_info.get("version", "unknown"),
        "model_framework": model_info.get("mlm:framework", "unknown"),
        "inference_time_seconds": inference_time,
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "preprocessing": {
            "resize": [256, 256],
            "normalization": "ImageNet",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "postprocessing": {
            "threshold": 0.5,
            "output_type": "binary_mask"
        }
    }


def preprocess_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def predict_image_enhanced(
    model: torch.nn.Module,
    image_path: str,
    stac_metadata: Dict[str, Any] = None,
    output_dir: str = "output"
) -> Dict[str, Any]:
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    input_tensor = preprocess_image(image_path)
    
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        inference_time = time.time() - start_time
    
    prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (prediction > 0.5).astype(np.uint8)
    
    model_info = stac_metadata.get("properties", {}) if stac_metadata else {}
    
    metadata = create_inference_metadata(
        image_path, model_info, inference_time,
        input_tensor.shape, output.shape
    )
    
    output_files = {}
    
    raw_output_path = output_path / "prediction_raw.npy"
    np.save(raw_output_path, prediction)
    output_files["raw_prediction"] = str(raw_output_path)
    
    mask_output_path = output_path / "prediction_mask.png"
    cv2.imwrite(str(mask_output_path), binary_mask * 255)
    output_files["binary_mask"] = str(mask_output_path)
    
    overlay_output_path = output_path / "prediction_overlay.png"
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (256, 256))
    
    overlay = original_image.copy()
    overlay[binary_mask == 1] = [0, 0, 255]
    result = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
    cv2.imwrite(str(overlay_output_path), result)
    output_files["overlay"] = str(overlay_output_path)
    
    metadata_path = output_path / "inference_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    output_files["metadata"] = str(metadata_path)
    
    results = {
        "prediction": prediction,
        "binary_mask": binary_mask,
        "metadata": metadata,
        "output_files": output_files,
        "refugee_camp_detected": bool(np.any(binary_mask))
    }
    
    return results


def predict_image(image_path: str, model_path: str = None) -> np.ndarray:
    if model_path:
        model = torch.jit.load(model_path)
    else:
        from model import LitRefugeeCamp
        model = LitRefugeeCamp()
        model.load_state_dict(torch.load("meta/best_model.pth"))
    
    input_tensor = preprocess_image(image_path)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    return (prediction > 0.5).astype(np.uint8)


def log_inference_example(model, data_module):
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    model.eval()
    device = next(model.parameters()).device  # Get model's device
    with torch.no_grad():
        for batch in test_loader:
            images, targets = batch
            images = images.to(device)  # Move images to model's device
            targets = targets.to(device)  # Move targets to model's device
            outputs = model(images[:1])
            
            input_img = images[0].cpu().numpy().transpose(1, 2, 0)
            input_img = (input_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            input_img = np.clip(input_img, 0, 1)
            
            prediction = torch.sigmoid(outputs[0]).squeeze().cpu().numpy()
            target = targets[0].squeeze().cpu().numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(input_img)
            axes[0].set_title("Input Image")
            axes[0].axis('off')
            
            axes[1].imshow(prediction, cmap='hot')
            axes[1].set_title("Prediction")
            axes[1].axis('off')
            
            axes[2].imshow(target, cmap='hot')
            axes[2].set_title("Ground Truth")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig("meta/example_input.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(5, 5))
            plt.imshow(prediction, cmap='hot')
            plt.axis('off')
            plt.savefig("meta/example_pred.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(5, 5))
            plt.imshow(target, cmap='hot')
            plt.axis('off')
            plt.savefig("meta/example_target.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log inference examples to MLflow
            import mlflow
            mlflow.log_artifact("meta/example_input.png", artifact_path="examples")
            mlflow.log_artifact("meta/example_pred.png", artifact_path="examples")
            mlflow.log_artifact("meta/example_target.png", artifact_path="examples")
            
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--model_path", help="Path to model file")
    parser.add_argument("--stac_path", help="Path to STAC metadata file")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--mlflow_tracking", action="store_true", help="Log to MLflow")
    
    args = parser.parse_args()
    
    stac_metadata = None
    if args.stac_path and Path(args.stac_path).exists():
        stac_metadata = load_stac_item(args.stac_path)
    
    if args.model_path:
        model = torch.jit.load(args.model_path)
    else:
        from model import LitRefugeeCamp
        model = LitRefugeeCamp()
        model.load_state_dict(torch.load("meta/best_model.pth"))
    
    results = predict_image_enhanced(
        model, args.image_path, stac_metadata, args.output_dir
    )
    
    if args.mlflow_tracking:
        with mlflow.start_run():
            mlflow.log_params(results["metadata"])
            mlflow.log_metric("inference_time", results["metadata"]["inference_time_seconds"])
            mlflow.log_metric("refugee_camp_detected", int(results["refugee_camp_detected"]))
            
            for file_type, file_path in results["output_files"].items():
                mlflow.log_artifact(file_path, f"inference_outputs/{file_type}")
    
    print(f"Inference completed. Results saved to: {args.output_dir}")
    print(f"Refugee camp detected: {results['refugee_camp_detected']}")


if __name__ == "__main__":
    main()
