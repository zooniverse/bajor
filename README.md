# Batch Job Runner

Azure Batch Job Runner - BaJoR

The Zooniverse API for scheduling Azure batch jobs for machine learning systems.

Linked to the work in https://github.com/zooniverse/kade

## Requirements

BaJoR uses Docker to manage its environment, the requirements listed below are also found in `docker-compose.yml`. The means by which a new instance is created with Docker is located in the `Dockerfile`. If you plan on using Docker to manage this application, skip ahead to Installation.

BaJoR is primarily developed against stable Python currently 3.10 and the Azure Batch python libraries https://docs.microsoft.com/en-us/python/api/overview/azure/batch?view=azure-python#install-the-libraries.

Optionally, you can also run the following:

* [Redis](http://redis.io) version >= 6

## Installation

We only support running BaJoR via Docker and Docker Compose. If you'd like to run it outside a container, see the above Requirements sections to get started.

## Usage

1. `docker-compose build`

2. `docker-compose up` to start the containers

    * If the above step reports a missing database error, kill the docker-compose process or open a new terminal window in the current directory and then run `docker-compose run --rm api bundle exec rake db:setup` to setup the database.

    * Alternatively use the following command to start a bash terminal session in the container `docker compose run --service-ports --rm api bash`

    * Run the tests in the container `docker compose run --service-ports --rm api RAILS_ENV=test bin/rspec`
