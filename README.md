# Batch Job Runner

Azure Batch Job Runner - BaJoR

The Zooniverse API for scheduling Azure batch jobs for machine learning systems.

Linked to the work in https://github.com/zooniverse/kade

## Documentation

### Training API docs

- [Swagger Format](https://bajor.zooniverse.org/docs)
- [Redoc Format](https://bajor.zooniverse.org/redoc)
- [OpenAPI JSON Format](https://bajor.zooniverse.org/openapi.json)


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
