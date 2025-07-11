import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def log_confusion_matrix(model, datamodule, mlflow_run, device="cpu"):
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
    plt.savefig("confusion_matrix.png")
    mlflow_run.log_artifact("confusion_matrix.png", artifact_path="metrics")
    plt.close()



def get_input_example(dataloader, device):
    try:
        sample_images, _ = next(iter(dataloader))
        return sample_images[:1].to(device)
    except StopIteration:
        return torch.zeros(1, 3, 256, 256).to(device)