# Quick-start Docker containers for ONNX Runtime

## CPU Version (Preview)
#### Linux 16.04, Python Bindings, Compatible with Docker for Windows

1. Retrieve your docker image in one of the following ways.

- Build the docker image from the DockerFile in this repository.
  ```
  # If you have a Linux machine, preface this command with "sudo"
  docker build -t onnxruntime-cpu -f Dockerfile.cpu .
  ```
 - Pull the official image from DockerHub.
 
   ```
   # Will be available with ONNX Runtime 0.2.0
   ```
2. Run the docker image

  ```
  # If you have a Linux machine, preface this command with "sudo"
  # If you have a Windows machine, preface this command with "winpty"

  docker run -it onnxruntime-cpu
  ```

## GPU Version (Preview)
#### Linux 16.04, Python Bindings, CUDA 10, CuDNN7, Requires Nvidia-Docker version 2.0

0. Prerequisites: [Install Nvidia-Docker 2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))

1. Retrieve your docker image in one of the following ways.    
  - Build the docker image from the DockerFile in this repository.
    ``` 
    # If you have a Linux machine, preface this command with "sudo"  
    
    docker build -t onnxruntime-gpu -f Dockerfile.gpu .
    ```
    Note that you can change the base CUDA distribution to 9.1 and use nvidia-docker v1
    by replacing the first line of the dockerfile with the base image below.
    ```
    FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
    ```
 - Pull the official image from DockerHub.
 
   ```
   # Will be available with ONNX Runtime 0.2.0
   ```

2. Run the docker image
  ```
  # If you have a Linux machine, preface this command with "sudo"
  # If you have a Windows machine, preface this command with "winpty"

  docker run -it --runtime=nvidia --rm nvidia/cuda onnxruntime-gpu
  ```

## nGraph Version (Preview)
#### Linux 16.04, Python Bindings

1. Build the docker image from the Dockerfile in this repository.
  ```
  # If you have a Linux machine, preface this command with "sudo"
  
  docker build -t onnxruntime-ngraph -f Dockerfile.ngraph .
  ```

2. Run the Docker image

  ```
  # If you have a Linux machine, preface this command with "sudo"
  
  docker run -it onnxruntime-ngraph
  ```

## ONNX Runtime Server (Preview)
#### Linux 16.04

1. Build the docker image from the directory that contains our server "onnxruntime_server"
  ```
  docker build -t {docker_image_name} .
  ```
  
2. Run the ONNXRuntime server in Docker image

  ```
  docker run -it -v {localModelAbsoluteFolder}:{dockerModelAbsoluteFolder} -e MODEL_ABSOLUTE_PATH={dockerModelAbsolutePath} -p {your_local_port}:8001 {imageName}
  ```
3. Send the request to server

  Send the request to the docker through the binding local port. Here is the full [usage document](https://github.com/Microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Server_Usage.md).
  ```
  curl  -X POST -d "@request.json" -H "Content-Type: application/json" http://0.0.0.0:{your_local_port}/v1/models/mymodel/versions/3:predict  
  ```

### Other options to get started with ONNX Runtime

- Deploy [inference for pretrained ONNX models](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx) for handwritten digit recognition (MNIST)
or facial expression recognition (FER+) using Azure Machine Learning

- Work with ONNX runtime in your local environment using the PyPi release ([CPU](https://pypi.org/project/onnxruntime/), [GPU](https://pypi.org/project/onnxruntime-gpu/))
    - ``pip install onnxruntime``
    - ``pip install onnxruntime-gpu``

- Build ONNX Runtime from the source code by following [these instructions for developers](../BUILD.md).

### License
[MIT License](../LICENSE)
