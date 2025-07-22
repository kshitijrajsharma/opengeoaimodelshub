import argparse
import os

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from inference import log_inference_example
from model import CampDataModule, LitRefugeeCamp
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from utils import log_confusion_matrix


def main(args):
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.enable_system_metrics_logging()

    mlf_logger = MLFlowLogger(experiment_name=args.experiment)

    dm = CampDataModule(args.chips_dir, args.labels_dir, batch_size=args.batch_size)
    dm.setup()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath="checkpoints",
        filename="best",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=mlf_logger,
        log_every_n_steps=5,
    )

    model = LitRefugeeCamp(lr=args.lr)

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    run_id = mlf_logger.run_id
    print(run_id)

    mlflow.log_params(
        {
            "chips_dir": args.chips_dir,
            "labels_dir": args.labels_dir,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": 62,
        },
        run_id=run_id,
    )
    mlflow.log_artifacts(args.chips_dir, artifact_path="dataset/chips", run_id=run_id)
    mlflow.log_artifacts(args.labels_dir, artifact_path="dataset/labels", run_id=run_id)

    if checkpoint_callback.best_model_path:
        mlflow.log_artifact(
            checkpoint_callback.best_model_path,
            artifact_path="checkpoints",
            run_id=run_id,
        )
        best_model = LitRefugeeCamp.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        mlflow.pytorch.log_model(
            best_model.model,
            name="best_model",
            registered_model_name=args.model_name,
            run_id=run_id,
        )

    conf_matrix_path = log_confusion_matrix(model.model, dm, mlflow)
    if conf_matrix_path and os.path.exists(conf_matrix_path):
        mlflow.log_artifact(
            conf_matrix_path, artifact_path="confusion_matrix", run_id=run_id
        )

    inference_example_path = log_inference_example(model.model, dm, mlflow)
    if inference_example_path and os.path.exists(inference_example_path):
        mlflow.log_artifact(
            inference_example_path, artifact_path="inference_example", run_id=run_id
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_uri", type=str, default="http://localhost:5000")
    parser.add_argument("--experiment", type=str, default="RefugeeCamp")
    parser.add_argument("--model_name", type=str, default="RefugeeCampDetector")
    parser.add_argument("--chips_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
