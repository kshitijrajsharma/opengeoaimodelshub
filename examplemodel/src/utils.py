import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def log_confusion_matrix(model, datamodule, mlflow_run, device=None):
    if device is None:
        device = next(model.parameters()).device  # Get model's device
    model.eval()
    y_true, y_pred = [], []
    loader = datamodule.test_dataloader()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds_bin = (preds > 0.5).cpu().numpy().astype(np.uint8)
            labels = y.cpu().numpy().astype(np.uint8)
            y_true.extend(labels.flatten())
            y_pred.extend(preds_bin.flatten())
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Pixel-wise Confusion Matrix")
    plt.savefig("meta/confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("meta/confusion_matrix.png", artifact_path="metrics")
    plt.close()

