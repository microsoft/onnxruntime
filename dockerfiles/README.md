# Docker containers for ONNX Runtime

- [Arm 32v7](Dockerfile.arm32v7)
- [Build from source (CPU)](Dockerfile.source)
- [CUDA + CUDNN](Dockerfile.cuda)
- [nGraph](Dockerfile.ngraph)
- [TensorRT](Dockerfile.tensorrt)
- [OpenVINO](Dockerfile.openvino)
- [ONNX Runtime Server](Dockerfile.server)

## Build from Source
#### Linux 16.04, CPU, Python Bindings

1. Build the docker image from the Dockerfile in this repository.
  ```
  # If you have a Linux machine, preface this command with "sudo"

  docker build -t onnxruntime-source -f Dockerfile.source .
  ```

2. Run the Docker image

  ```
  # If you have a Linux machine, preface this command with "sudo"

  docker run -it onnxruntime-source
  ```

## CUDA
#### Linux 16.04, CUDA 10.0, CuDNN 7

1. Build the docker image from the Dockerfile in this repository.
  ```
  # If you have a Linux machine, preface this command with "sudo"

  docker build -t onnxruntime-cuda -f Dockerfile.cuda .
  ```

2. Run the Docker image

  ```
  # If you have a Linux machine, preface this command with "sudo"

  docker run -it onnxruntime-cuda
  ```

## nGraph (Public Preview)
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

## TensorRT
#### Linux 16.04, TensorRT 5.0.2

1. Build the docker image from the Dockerfile in this repository.
  ```
  # If you have a Linux machine, preface this command with "sudo"

  docker build -t onnxruntime-trt -f Dockerfile.tensorrt .
  ```

2. Run the Docker image

  ```
  # If you have a Linux machine, preface this command with "sudo"

  docker run -it onnxruntime-trt
  ```

## OpenVINO (Public Preview)
#### Linux 16.04, Python Bindings

1. Build the onnxruntime image for all the accelerators supported as below 

   Retrieve your docker image in one of the following ways.

    -  For building the docker image, download OpenVINO online installer version 2019 R1.1 from [here](https://software.intel.com/en-us/openvino-toolkit/choose-download) and copy the openvino tar file in the same directory and build the image. The online installer size is only 16MB and the components needed for the accelerators are mentioned in the dockerfile. Providing the argument device enables onnxruntime for that particular device. You can also provide arguments ONNXRUNTIME_REPO and ONNXRUNTIME_BRANCH to test that particular repo and branch. Default values are http://github.com/microsoft/onnxruntime and repo is master
       ```
       docker build -t onnxruntime --build-arg DEVICE=$DEVICE .
       ```
    - Pull the official image from DockerHub.
   

2. DEVICE: Specifies the hardware target for building OpenVINO Execution Provider. Below are the options for different Intel target devices.

	| Device Option | Target Device |
	| --------- | -------- |
	| <code>CPU_FP32</code> | Intel<sup></sup> CPUs |
	| <code>GPU_FP32</code> |Intel<sup></sup> Integrated Graphics |
	| <code>GPU_FP16</code> | Intel<sup></sup> Integrated Graphics |
	| <code>MYRIAD_FP16</code> | Intel<sup></sup> Movidius<sup>TM</sup> USB sticks |
	| <code>VAD-M_FP16</code> | Intel<sup></sup> Vision Accelerator Design based on Movidius<sup>TM</sup> MyriadX VPUs |

## CPU 

1. Retrieve your docker image in one of the following ways.

   - Build the docker image from the DockerFile in this repository. 
      
     ```
     docker build -t onnxruntime-cpu --build-arg DEVICE=CPU_FP32 --network host .
     ```
   - Pull the official image from DockerHub.
     ```
     # Will be available with next release
     ```
2. Run the docker image
    ```
     docker run -it onnxruntime-cpu
    ```

## GPU

1. Retrieve your docker image in one of the following ways. 
   - Build the docker image from the DockerFile in this repository.
     ``` 
      docker build -t onnxruntime-gpu --build-arg DEVICE=GPU_FP32 --network host . 
     ```
   - Pull the official image from DockerHub.
     ```
       # Will be available with next release
     ```

2. Run the docker image
    ```
    docker run -it --device /dev/dri:/dev/dri onnxruntime-gpu:latest
    ```
## Myriad VPU Accelerator 

1. Retrieve your docker image in one of the following ways. 
   - Build the docker image from the DockerFile in this repository.
     ``` 
      docker build -t onnxruntime-myriad --build-arg DEVICE=MYRIAD_FP16 --network host . 
     ```
   - Pull the official image from DockerHub.
     ```
      # Will be available with next release
     ```
2. Install the Myriad rules drivers on the host machine according to the reference in [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#additional-NCS-steps)
3. Run the docker image by mounting the device drivers
    ```
    docker run -it --network host --privileged -v /dev:/dev  onnxruntime-myriad:latest

    ```
=======
## VAD-M Accelerator Version 

1. Retrieve your docker image in one of the following ways. 
   - Build the docker image from the DockerFile in this repository.
     ``` 
      docker build -t onnxruntime-vadr --build-arg DEVICE=VAD-M_FP16 --network host . 
     ```
   - Pull the official image from DockerHub.
     ```
      # Will be available with next release
     ```
2. Install the HDDL drivers on the host machine according to the reference in [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux_ivad_vpu.html)
3. Run the docker image by mounting the device drivers
    ```
    docker run -it --device --mount type=bind,source=/var/tmp,destination=/var/tmp --device /dev/ion:/dev/ion  onnxruntime-hddl:latest

    ```
## ONNX Runtime Server (Public Preview)
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

