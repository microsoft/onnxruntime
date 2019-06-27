# Docker containers for ONNX Runtime

- [Arm 32v7](Dockerfile.arm32v7)
- [Build from source](Dockerfile.source)
- [CUDA + CUDNN](Dockerfile.cuda)
- [nGraph](Dockerfile.ngraph)
- [TensorRT](Dockerfile.tensorrt)

# Quick-Start Docker containers for ONNX Runtime

- [CPU](Dockerfile.cpu)
- [GPU](Dockerfile.gpu)

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

1. Build the docker image from the Dockerfile in this repository
  ```
  docker build -t {docker_image_name} -f Dockerfile.server .
  ```
  
2. Run the ONNXRuntime server with the image created in step 1

  ```
  docker run -v {localModelAbsoluteFolder}:{dockerModelAbsoluteFolder} -e MODEL_ABSOLUTE_PATH={dockerModelAbsolutePath} -p {your_local_port}:8001 {imageName}
  ```
3. Send HTTP requests to the container running ONNX Runtime Server

  Send HTTP requests to the docker container through the binding local port. Here is the full [usage document](https://github.com/Microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Server_Usage.md).
  ```
  curl  -X POST -d "@request.json" -H "Content-Type: application/json" http://0.0.0.0:{your_local_port}/v1/models/mymodel/versions/3:predict  
  ```

## OpenVINO Version (Preview)
#### Linux 16.04, Python Bindings

1. Build the docker image from the Dockerfile in this repository.
  ```
  # If you have a Linux machine, preface this command with "sudo"

  docker build -t onnxruntime-openvino -f Dockerfile.openvino .
  ```
  To use GPU_FP32:
  ```
  docker build -t onnxruntime-openvino --build-arg TARGET_DEVICE=GPU_FP32 -f Dockerfile.openvino .
  ```

2. Run the Docker image

  ```
  # If you have a Linux machine, preface this command with "sudo"

  docker run -it onnxruntime-openvino
  ```