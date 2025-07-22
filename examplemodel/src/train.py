import argparse

import mlflow
import pytorch_lightning as pl
import torch
from inference import log_inference_example
from model import CampDataModule, LitRefugeeCamp
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import get_input_example, log_confusion_matrix

# mlflow.autolog(log_models=True)

def main(args):
    # mlflow.autolog(log_models=True)
    print(f"Using MLflow URI: {args.mlflow_uri}")
    mlflow.set_tracking_uri(args.mlflow_uri)
    # mlflow.set_experiment(args.experiment)
    mlflow.enable_system_metrics_logging()


    dm = CampDataModule(args.chips_dir, args.labels_dir, batch_size=args.batch_size)
    dm.setup()

    mlflow.log_artifacts(args.chips_dir, artifact_path="dataset/chips")
    mlflow.log_artifacts(args.labels_dir, artifact_path="dataset/labels")

    checkpoint = ModelCheckpoint(
        monitor="val_acc", mode="max", save_top_k=1, save_last=True, dirpath="checkpoints"
    )


    # with open("training.log", "w") as f, redirect_stdout(f), redirect_stderr(f):
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint],
        log_every_n_steps=5,
    )
    model = LitRefugeeCamp(lr=args.lr)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    if checkpoint.best_model_path:
        mlflow.log_artifact(checkpoint.best_model_path, artifact_path="checkpoints")
        model_to_log = LitRefugeeCamp.load_from_checkpoint(checkpoint.best_model_path)
        # sample_input_to_log = get_input_example(dm.train_dataloader(), torch.device("cuda" if torch.cuda.is_available() else "cpu")).cpu().detach().numpy()

        mlflow.pytorch.log_model(
            model_to_log.model,
            name="model",
            registered_model_name=args.model_name,
            # input_example=sample_input_to_log,
        )

    log_confusion_matrix(model.model, dm, mlflow)


    log_inference_example(model.model, dm, mlflow)

        # mlflow.log_artifact("training.log")

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