# Quick-start Docker containers for ONNX Runtime

### CPU Version
#### Linux 16.04, Python Bindings, Compatible with Docker-Windows
1. Retrieve the docker image
  a. Pull the images from DockerHub

  b. Or build the docker image from the DockerFile in this repository.
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

### GPU Version -- nvidia docker
1. Retrieve your docker image
  a. Pull the images from DockerHub
  b. Or build the docker image from the DockerFile in this repository.
  ```
  # If you have a Linux machine, preface this command with "sudo"

  nvidia-docker build -t onnxruntime-gpu -f Dockerfile.gpu .
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
