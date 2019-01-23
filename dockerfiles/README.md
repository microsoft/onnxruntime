# Quick-start Docker containers for ONNX Runtime

## CPU Version
#### Linux 16.04, Python Bindings, Compatible with Docker for Windows

1. Retrieve your docker image in one of the following ways.
  - Pull the official image from DockerHub.
    ```
    docker pull Microsoft/onnxruntime:cpu
    ```
  - Build the docker image from the DockerFile in this repository.
    ```
    # If you have a Linux machine, preface this command with "sudo"

    docker build -t onnxruntime-cpu -f Dockerfile.cpu .
    ```
2. Run the docker image
  ```
  # If you have a Linux machine, preface this command with "sudo"
  # If you have a Windows machine, preface this command with "winpty"

  docker run -it onnxruntime-cpu
  ```

## GPU Version
#### Linux 16.04, Python Bindings, CUDA 10, CuDNN7, Requires Nvidia-Docker version 2.0

0. Prerequisites: [Install Nvidia-Docker 2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))

1. Retrieve your docker image in one of the following ways.
  - Pull the official image from DockerHub.
    ```
    docker pull Microsoft/onnxruntime:gpu
    ```
  - Build the docker image from the DockerFile in this repository.
    ```
    # If you have a Linux machine, preface this command with "sudo"

    docker build -t onnxruntime-gpu -f Dockerfile.gpu .
    ```
    Note that you can change the base CUDA distribution to 9.1 and use nvidia-docker v1
    by building the docker image as shown above and replacing the first line with the base image below.
    ```
    FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
    ```
2. Run the docker image
  ```
  # If you have a Linux machine, preface this command with "sudo"
  # If you have a Windows machine, preface this command with "winpty"

  docker run -it --runtime=nvidia --rm nvidia/cuda onnxruntime-gpu
  ```
### Other options to get started with ONNX Runtime

- Deploy [inference for pretrained ONNX models](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx) for handwritten digit recognition (MNIST)
or facial expression recognition (FER+) using Azure Machine Learning

- Work with ONNX runtime in your local environment using the PyPi release ([CPU](https://pypi.org/project/onnxruntime/), [GPU](https://pypi.org/project/onnxruntime-gpu/))
    - ``pip install onnxruntime`` or ``pip install onnxruntime-gpu``

- Build ONNX Runtime from the source code by following [these instructions for developers](../BUILD.md).

### License
[MIT License](../LICENSE)
