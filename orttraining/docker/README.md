# AzureML Base GPU Docker Images
AzureML provides base docker images for running on AzureML Compute (https://hub.docker.com/_/microsoft-azureml-base-gpu).  However, these don't support CUDA 10.1 or OpenMPI 4.0.  We provide Dockerfiles for these variants, as well as Dockerfiles to build and package onnxruntime training into an AzureML-compatible docker image.

## Prerequisites
Install prerequisites.

1. Install Docker: https://docs.docker.com/install/
1. Install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest
1. $ pip install azure-cli-core azure-mgmt-containerregistry azureml-sdk
1. $ az login
1. $ docker login onnxtraining.azurecr.io --username onnxtraining --password $(az acr credential show --name onnxtraining --subscription 'ea482afa-3a32-437c-aa10-7de928a9e793' --resource-group 'onnx_training' --query passwords[0].value)

## Build onnxruntime in an AzureML Docker container
These docker images allow us to build onnxruntime with the desired library versions (CUDA/cuDNN/MPI/etc) and compatible with AzureML Compute.

Build onnxruntime using the desired docker image:
```
$ docker run --rm -v /path/to/onnxruntime:/onnxruntime onnxtraining.azurecr.io/azureml/build:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04 <optional-build-arguments>
$ docker run --rm -v /path/to/onnxruntime:/onnxruntime onnxtraining.azurecr.io/azureml/build:intelmpi2018.3-cuda10.0-cudnn7-ubuntu16.04 <optional-build-arguments>
$ docker run --rm -v /path/to/onnxruntime:/onnxruntime onnxtraining.azurecr.io/azureml/build:openmpi4.0.0-cuda10.1-cudnn7-ubuntu16.04 <optional-build-arguments>
$ docker run --rm -v /path/to/onnxruntime:/onnxruntime onnxtraining.azurecr.io/azureml/build:intelmpi2018.3-cuda10.1-cudnn7-ubuntu16.04 <optional-build-arguments>
```

## Package onnxruntime into an AzureML Docker container
Package the built onnxruntime_training_bert into an AzureML-compatible docker image (set the BASE_IMAGE corresponding to how onnxruntime was built above):

| Build image | BASE_IMAGE |
| ----------- | ---------- |
| onnxtraining.azurecr.io/azureml/build:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04 | mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04 |
| onnxtraining.azurecr.io/azureml/build:intelmpi2018.3-cuda10.0-cudnn7-ubuntu16.04 | mcr.microsoft.com/azureml/base-gpu:intelmpi2018.3-cuda10.0-cudnn7-ubuntu16.04 |
| onnxtraining.azurecr.io/azureml/build:openmpi4.0.0-cuda10.1-cudnn7-ubuntu16.04 | onnxtraining.azurecr.io/azureml/base-gpu:openmpi4.0.0-cuda10.1-cudnn7-ubuntu16.04 |
| onnxtraining.azurecr.io/azureml/build:intelmpi2018.3-cuda10.1-cudnn7-ubuntu16.04 | onnxtraining.azurecr.io/azureml/base-gpu:intelmpi2018.3-cuda10.1-cudnn7-ubuntu16.04 |

```
$ docker build --rm -t onnxtraining.azurecr.io/azureml/bert:<tag> --build-arg BASE_IMAGE=mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04 -f ./Dockerfile build/Linux/RelWithDebInfo
$ docker push onnxtraining.azurecr.io/azureml/bert:<tag>
```

## Run the AureML Docker container on local compute
The docker container can be run on a local machine.  The model needs to be mounted into the container:
```
$ docker run --rm --runtime=nvidia -v /path/to/model:/model onnxtraining.azurecr.io/azureml/bert:<tag> /workspace/onnxruntime_training_bert --model_name /model/bert_L-24_H-1024_A-16_V_30528_optimized_layer_norm --train_batch_size 4 --mode perf --num_of_perf_samples 400
```

## Run an Experiment on AzureML Compute
After the AzureML-compatible docker container is created and pushed, an AzureML Experiment can be submitted to run on AzureML Compute.  An Azure Portal link will be displayed to view progress and logs.  Two AzureML Compute targets are pre-created for use:

1. 'onnx-training': 16-node NC24s-v3 cluster (use Open MPI docker images)
1. 'intel-training': 16-node NC24rs-v3 cluster (use Intel MPI docker images)

```
// Submit a single-node, single-GPU experiment (defaults to OpenMPI 4.0.0, CUDA 10.1, bert_L-24_H-1024_A-16_V_30528_optimized_layer_norm.onnx):
$ python experiment.py --script_params='--train_batch_size=4 --mode=perf --num_of_perf_samples=400'
Experiment running at: https://mlworkspace.azure.ai/portal/subscriptions/ea482afa-3a32-437c-aa10-7de928a9e793/resourceGroups/onnx_training/providers/Microsoft.MachineLearningServices/workspaces/ort_training_dev/experiments/BERT-ONNX

// Submit a single-node, single-GPU experiment with mixed-precision training on a custom docker image using IntelMPI:
$ python experiment.py --container='onnxtraining.azurecr.io/azureml/bert:jesseb-cd29764cbe-intelmpi2018.3-cuda10.1-cudnn7-ubuntu16.04' --compute_target='intel-training' --script_params='--train_batch_size=8 --mode=perf --num_of_perf_samples=800 --use_mixed_precision=True'

// Submit a multi-node, multi-GPU experiment (4-node x 4-GPU) with gradient accumulation:
$ python experiment.py --node_count=4 --gpu_count=4 --script_params='--train_batch_size=4 --mode=perf --num_of_perf_samples=6400 --gradient_accumulation_steps=16'

// See all options for running experiments:
$ python experiment.py --help
```

## Tensorboard for AzureML Experiments
ONNX Training tensorboard events can be streamed to a local directory for interactive viewing (updates approximately every 5 seconds).

```
// Submit an Experiment with tensorboard logging enabled (must not be --mode=perf):
$ python experiment.py --script_params='--train_batch_size=4 --display_loss_steps=10 --log_dir=logs/tensorboard/'

// Stream the tensorboard logs locally, or download logs if Experiment is already complete:
$ python watch_experiment.py --remote_dir='logs/tensorboard/' --local_dir='C:/tensorboard' --run='BERT-ONNX_1578424056_a4b83d75'

// Start Tensorboard locally:
$ python -m tensorboard.main --logdir C:/tensorboard
```

# Additional AzureML GPU Docker images
AzureML base GPU docker images only support CUDA 9/10. We provide Dockerfiles for additional variants below.

## AzureML Docker image: CUDA 10.1, cuDNN 7.6.3, NCCL 2.4.8, OpenMPI 4.0.0
This image is already built and pushed to onnxtraining.azurecr.io. To build the docker image from scratch:
```
$ docker build --rm -t onnxtraining.azurecr.io/azureml/base-gpu:openmpi4.0.0-cuda10.1-cudnn7-ubuntu16.04 openmpi
$ docker push onnxtraining.azurecr.io/azureml/base-gpu:openmpi4.0.0-cuda10.1-cudnn7-ubuntu16.04
```

## AzureML Docker image: CUDA 10.1, cuDNN 7.6.3, NCCL 2.4.8, IntelMPI 2018.3.222
This image is already built and pushed to onnxtraining.azurecr.io. To build the docker image from scratch:
```
$ docker build --rm -t onnxtraining.azurecr.io/azureml/base-gpu:intelmpi2018.3-cuda10.1-cudnn7-ubuntu16.04 intelmpi
$ docker push onnxtraining.azurecr.io/azureml/base-gpu:intelmpi2018.3-cuda10.1-cudnn7-ubuntu16.04
```

## AzureML Docker image: Build environment
These docker images allow us to build onnxruntime with the various versions of libraries and package into an AzureML-compatible docker container.

These images are already built and pushed to onnxtraining.azurecr.io. To build the docker images from scratch:
```
$ docker build --rm -t onnxtraining.azurecr.io/azureml/build:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04 --build-arg BASE_IMAGE=mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04 build

$ docker build --rm -t onnxtraining.azurecr.io/azureml/build:intelmpi2018.3-cuda10.0-cudnn7-ubuntu16.04 --build-arg BASE_IMAGE=mcr.microsoft.com/azureml/base-gpu:intelmpi2018.3-cuda10.0-cudnn7-ubuntu16.04 build

$ docker build --rm -t onnxtraining.azurecr.io/azureml/build:openmpi4.0.0-cuda10.1-cudnn7-ubuntu16.04 --build-arg BASE_IMAGE=onnxtraining.azurecr.io/azureml/base-gpu:openmpi4.0.0-cuda10.1-cudnn7-ubuntu16.04 build

$ docker build --rm -t onnxtraining.azurecr.io/azureml/build:intelmpi2018.3-cuda10.1-cudnn7-ubuntu16.04 --build-arg BASE_IMAGE=onnxtraining.azurecr.io/azureml/base-gpu:intelmpi2018.3-cuda10.1-cudnn7-ubuntu16.04 build
```
