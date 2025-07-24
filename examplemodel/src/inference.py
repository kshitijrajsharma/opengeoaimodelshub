import argparse

import matplotlib.pyplot as plt
import mlflow.pytorch
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

def predict_image(model, image_path):
    img = Image.open(image_path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        y_hat = model(x)
    mask = (y_hat[0,0] > 0.5).cpu().numpy().astype(np.uint8)
    return mask

def log_inference_example(model, datamodule, mlflow_run):
    loader = datamodule.test_dataloader()
    x, y = next(iter(loader))
    with torch.no_grad():
        pred = model(x[:1])

    from torchvision.utils import save_image
    save_image(x[0], "meta/example_input.png")
    save_image(pred[0], "meta/example_pred.png")
    save_image(y[0], "meta/example_target.png")
    mlflow_run.log_artifact("meta/example_input.png", artifact_path="inference")
    mlflow_run.log_artifact("meta/example_pred.png", artifact_path="inference")
    mlflow_run.log_artifact("meta/example_target.png", artifact_path="inference")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output_mask.png")
    args = parser.parse_args()
    model = mlflow.pytorch.load_model(args.model_path)
    mask = predict_image(model, args.image_path)
    plt.imsave(args.output_path, mask * 255, cmap="gray", vmin=0, vmax=255)
    print(f"Inference mask saved to {args.output_path}")