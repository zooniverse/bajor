# Batch Job Runner

Azure Batch Job Runner - BaJoR

The Zooniverse API for scheduling Azure batch jobs for machine learning systems.

Linked to the work in https://github.com/zooniverse/kade

## Documentation

### Training API docs

- [Swagger Format](https://bajor.zooniverse.org/docs)
- [Redoc Format](https://bajor.zooniverse.org/redoc)
- [OpenAPI JSON Format](https://bajor.zooniverse.org/openapi.json)


## Per-job Hugging Face configuration

Training and prediction accept these optional fields in `opts`:

```json
{
  "huggingface_checkpoint_filename": "checkpoints/best.ckpt"
}
```

The repository is configured only by the deployment `HF_REPO_ID`, not by job options or KaDE context metadata. The filename supplied by KaDE is forwarded as `--hf-filename` to both batch scripts. Omitting the filename preserves the existing checkpoint basename defaults. Use the exact filename or path inside the model repository to select the same checkpoint for training and prediction. Hugging Face handles filename and file-existence errors during download.

Hugging Face downloads and uploads require both `HF_REPO_ID` and `HF_TOKEN`, including for public repositories. Keep `HF_TOKEN` in the deployment environment. When either setting is missing, the scripts use the existing local checkpoint path and skip Hub uploads. The scripts load Zoobot-compatible checkpoints; arbitrary Hub architectures are not supported. Existing training uploads use the same repository and filename, requiring write access. Existing custom-schema training may initialize a fresh Zoobot model if the Hub checkpoint is missing; other download errors fail the job.

Deploy the API and copy the updated scripts to both training and prediction code storage before sending these options from KaDE.

Prediction jobs pass `opts.workflow_name` to the batch script as `--workflow-name`. The script uses `args.workflow_name` to name Hugging Face prediction results as `predictions/{workflow_name}.json` (or `.csv` for CSV output), replacing that workflow's previous results on each successful upload. This argument supplies the upload name; prediction schema loading continues to use the existing custom-schema/default behavior. Training still receives `--schema`, while predictions do not generate that argument. The Azure results path remains specific to the job. Without `--workflow-name`, the script skips the Hub upload and logs a warning.

## Requirements

BaJoR uses Docker to manage its environment, the requirements listed below are also found in `docker-compose.yml`. The means by which a new instance is created with Docker is located in the `Dockerfile`. If you plan on using Docker to manage this application, skip ahead to Installation.

BaJoR is primarily developed against stable Python currently 3.10 and the Azure Batch python libraries https://docs.microsoft.com/en-us/python/api/overview/azure/batch?view=azure-python#install-the-libraries.

## Installation

We only support running BaJoR via Docker and Docker Compose. If you'd like to run it outside a container, see the above Requirements sections to get started.

## Usage

1. `docker-compose build`

2. `docker-compose up` to start the containers

    * Alternatively use the following command to start a bash terminal session in the container `docker compose run --service-ports --rm api bash`

    * Run the tests in the container `docker compose run --service-ports --rm api poetry run pytest`
