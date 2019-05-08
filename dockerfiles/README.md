# Quick-start Docker containers for ONNX Runtime

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
