# Quick-start Docker containers for ONNX Runtime

### CPU Version
#### Linux 16.04, Python Bindings, Compatible with Docker-Windows
1. Retrieve your docker image in one of the following ways.
  a. Pull the official image from DockerHub.
  ```
  docker pull [ORG]/onnxruntime:gpu
  ```
  b. Build the docker image from the DockerFile in this repository.
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

### GPU Version
#### Linux 16.04, Python Bindings, CUDA 10, CuDNN7, Requires Nvidia-Docker v2
1. Retrieve your docker image in one of the following ways.
  a. Pull the official image from DockerHub.
  ```
  docker pull [ORG]/onnxruntime:gpu
  ```
  b. Build the docker image from the DockerFile in this repository.
  ```
  # If you have a Linux machine, preface this command with "sudo"

  nvidia-docker build -t onnxruntime-gpu -f Dockerfile.gpu .
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

  nvidia-docker run -it onnxruntime-gpu
  ```
### Other options
- Launch an ONNX ML Tools Jupyter Notebook environment with starter code notebooks!
    - convert your models from other frameworks (i.e. Caffe2, CoreML,
      Tensorflow, PyTorch, etc.) to ONNX files and run them using ONNX Runtime

- Deploy inference for pretrained ONNX models for handwritten digit recognition (MNIST)
or facial expression recognition (FER+) using Azure Machine Learning.

- Build ONNX runtime from the source code by following [these instructions for developers](../BUILD.md).

### Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### License
[MIT License](../LICENSE)
