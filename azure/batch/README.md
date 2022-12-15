# Zoobot Training batch processing API developer readme

Zoobot is built form this Github Repo <https://github.com/mwalmsley/zoobot/>

## Build a development version of the docker image

Add features by testing training and prediction scripts via the same docker image env that the production batch system will use

This image depends on the upstream Zoobot project <https://github.com/mwalmsley/zoobot#direct-use>

``` sh
# build the image for use
docker-compose build
# run a bash console in the dev container
docker compose run --service-ports --rm zoobot bash
# do your dev work and test it!

# e.g. test your finetuning training system is working
 python scripts/train_model_finetune_on_catalog.py --save-dir data/external/finetuning_results/ --batch-size 3 --accelerator cpu --num-workers 2 --checkpoint data/external/models/checkpoints/decals5-zoobot-ckpt --catalog data/external/cosmic_dawn/catalogs/gz_cosmic_dawn_ortho_train_catalog_subset.csv

# or test a prediction system
 python scripts/predict_catalog_with_model.py --checkpoint-path data/external/checkpoints/zoobot.ckpt --catalog-url https://raw.githubusercontent.com/camallen/files-o-matic/main/hamlet-manifests/hamlet-subject-assistant-example-manifest.json --save-path data/external/prediction_results/results.json --batch-size 3 --num-workers 2 --accelerator cpu --gpus 0

# when you are ready to publish the image follow the steps below
```

## Build the Docker image for Azure Batch node pools

We need to build a Docker image with the necessary packages (ML system Pytorch or TensorFlow) to run the code in the Azure Batch ecosystem.

Azure Batch will pull this image from a private container registry, which needs to be in the same region as the Batch account.

Build the image from the default (CUDA) Dockerfile from the root of the zoobot repo:

``` sh
# export the image to the Zooniverse's 'zoobot' Azure container registry
export IMAGE_NAME=zoobot.azurecr.io/pytorch:1.10.1-gpu-py3
# registry name is the prefix in the above, e.g. zoobot.azurecr.io
export REGISTRY_NAME=zoobot
# build the zoobot image from the docker compose setup
docker-compose build zoobot
# tag the default zoobot image (cuda) for the container registry
# this tag will be used in the Azure batch nodepool specification
# to ensure the correct zoobot runtime image is used for batch compute resources
docker image tag zoobot:cuda $IMAGE_NAME
```

Optional - Test that CUDA can use the GPU in an interactive Python session. This step will only work if you:

- have an nvidia GPU
- the docker drivers setup on your machine
- attached the GPU to the docker containers

Most likely you won't have the correct setup on your dev machine (the azure batch system does) - though linux based folks should be able to use this.

``` sh
docker run --gpus all -it --rm $IMAGE_NAME /bin/bash

python
>>> import torch
>>> print(torch.cuda.is_available())
>>> print(torch.cuda.device_count())
>>> print(torch.cuda.current_device())
>>> print(torch.cuda.get_device_name(0))
# or via the published zoobot code at <https://github.com/mwalmsley/zoobot/blob/048543f21a82e10e7aa36a44bd90c01acd57422a/zoobot/pytorch/estimators/cuda_check.py>
>>> from zoobot.pytorch.estimators import cuda_check
>>> quit()
```

You can now exit/stop the container.

### Publish the built image to the Azure container registry for use in the Azure Batch system

Log in to the Azure Container Registry for the batch API project and push the image, note you may have to `az login` first:

``` sh
# ensure your machine with the built image has access to the container registry
az acr login --name $REGISTRY_NAME
# push it up to the azure container registry
docker image push $IMAGE_NAME
```

View the results of the image push in the azure portal in the `pytorch` repository, <https://docs.microsoft.com/en-us/azure/container-registry/container-registry-repositories#view-repositories-in-azure-portal>

## Create a Azure Batch compute Nodepool

We create a separate nodepools for each instance of a deployed Zoobot (ML system) image. These node pools supply the compute resources that process the submitted jobs. We use the nodepool compute resources to run the built Zoobot (ML system) docker images**.

Each node pool can specify different compute resources and/or different docker images (ML systems) that we can run batch jobs in. This abstraction allows us to create different nodepools for different ML systems / applications long term.

Currently we only have one `training` node pool for the Zoobot ML system but in theory we can have different nodepool targets as needed. For instance, we may build a different nodepool for the `inferencing` system if the `training` nodepool doesn't meet requirements and alteratively we may have a new ML system that uses say Tensorflow, we would create a new nodepool for this system and link a new docker image that can run the tensorflow version of this new ML code.

Follow the notebook in [examples/create_batch_pool_zoobot.ipynb](https://github.com/zooniverse/panoptes-python-notebook/blob/master/examples/create_batch_pool_zoobot.ipynb) to create a Zoobot nodepool for the ML training / inference workload.. You should only need to do this for new instances of Zoobot / ML systems or to add new compute resources, e.g. new GPU type or more GPUs, CPU, RAM, etc.

** Azure batch is responsible for the VM provisioning etc - as long as your node pool is setup correctly the docker images will run with the CPU, RAM and GPUs etc!

## Schedule Batch Jobs!

Now we have the Azure batch system setup to run code we can show how we use BaJoR to submit jobs to run. The system is currently setup to submit the zoobot training jobs to the training node pool only however the nodepool is a configurable ENV variable** for BaJoR deployments.

As specified in <https://github.com/zooniverse/bajor/blob/8567ebeb895c3d431f86a89bddbe6732e6afbf73/bajor/batch.py#L197-L200> we allow the nodepool id to be set when we submit the batch job for processing.

Finally we tell the Zoobot (ML system) code what to do via the job's task command specified at <https://github.com/zooniverse/bajor/blob/8567ebeb895c3d431f86a89bddbe6732e6afbf73/bajor/batch.py#L143>. Here we tell it to run the `train_model_finetune_on_catalog.py` code to train Zoobot but again this is configurable as required (either via the API or via a controlled list of operations we want to perform TBC).

**longer term we can extract to the API for configuration on each request TBC.
