## Background

This is an experiment on how to build a Geo-AI model that can be reproduced, deployed and scaled. I am setting up a simple CNN building detection model with standardized infrastructure and metadata. This model will serve as a template for how model could be documented or developed to be able to reused and reproduced.

## Setup 

1. Install UV 

Go [here](https://docs.astral.sh/uv/getting-started/installation/) 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install project 

```bash
uv sync
```

& you are done ! Easy Peasy ! 

3. Get Mlflow UI up 

```bash 
uv run mlflow ui
```

4. Preprocess
```bash 
uv run mlflow run . -e preprocess --env-manager local
```

5. Train 

```bash 
uv run mlflow run . -e train --env-manager local

```

## Info about tracking to server 

we use minio to store the artifacts in order to push : you need to configure following `.env` variable 

```bash
export AWS_ACCESS_KEY_ID=mlflow
export AWS_SECRET_ACCESS_KEY=mlflow123
export MLFLOW_S3_ENDPOINT_URL=http://your_remote_server_minio:9000
```

## About this model 

This is a simple building detection model (No pretrained backbone) , Uses images from OpenAerialMap and labels from OpenStreetMap . Has custom dataloader , preprocessing and postprocessing to replicate the custom nature of models ! 