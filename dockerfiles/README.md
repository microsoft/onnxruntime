# Docker Containers for ONNX Runtime

**Dockerfiles**


- CPU [Dockerfile](Dockerfile.source), [Instructions](#cpu)
- CUDA + CUDNN: [Dockerfile](Dockerfile.cuda), [Instructions](#cuda)
- nGraph: [Dockerfile](Dockerfile.ngraph), [Instructions](#ngraph)
- TensorRT: [Dockerfile](Dockerfile.tensorrt), [Instructions](#tensorrt)
- OpenVINO: [Dockerfile](Dockerfile.openvino), [Instructions](#openvino)
- Nuphar: [Dockerfile](Dockerfile.nuphar), [Instructions](#nuphar)
- ARM 32v7: [Dockerfile](Dockerfile.arm32v7), [Instructions](#arm-32v7)
- ONNX-Ecosystem (CPU + Converters): [Dockerfile](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/Dockerfile), [Instructions](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem)
- ONNX Runtime Server: [Dockerfile](Dockerfile.server), [Instructions](#onnx-runtime-server)

**Published Microsoft Container Registry (MCR) Images**

Use `docker pull` with any of the images and tags below to pull an image and try for yourself. Note that the CPU, CUDA, and TensorRT images include additional dependencies like miniconda for compatibility with AzureML image deployment.

**Example**: Run `docker pull mcr.microsoft.com/azureml/onnxruntime:latest-cuda` to pull the latest released docker image with ONNX Runtime GPU, CUDA, and CUDNN support.

| Build Flavor      | Base Image                            | ONNX Runtime Docker Image tags        | Latest                  |
|-------------------|---------------------------------------|---------------------------------------|-------------------------|
| Source (CPU)      | mcr.microsoft.com/azureml/onnxruntime | :v0.4.0, :v0.5.0, v0.5.1, :v1.0.0     | :latest                 |
| CUDA (GPU)        | mcr.microsoft.com/azureml/onnxruntime | :v0.4.0-cuda10.0-cudnn7, :v0.5.0-cuda10.1-cudnn7, v0.5.1-cuda10.1-cudnn7, :v1.0.0-cuda10.1-cudnn7 | :latest-cuda            |
| TensorRT (x86)    | mcr.microsoft.com/azureml/onnxruntime | :v0.4.0-tensorrt19.03, :v0.5.0-tensorrt19.06, v1.0.0-tensorrt19.09 | :latest-tensorrt        |
| OpenVino (VAD-M)  | mcr.microsoft.com/azureml/onnxruntime | :v0.5.0-openvino-r1.1-vadm            | :latest-openvino-vadm   |
| OpenVino (MYRIAD) | mcr.microsoft.com/azureml/onnxruntime | :v0.5.0-openvino-r1.1-myriad, :v1.0.0-openvino-r1.1-myriad | :latest-openvino-myriad |
| OpenVino (CPU)    | mcr.microsoft.com/azureml/onnxruntime | :v1.0.0-openvino-r1.1-cpu             | :latest-openvino-cpu    |
| nGraph            | mcr.microsoft.com/azureml/onnxruntime | :v1.0.0-ngraph-v0.26.0                | :latest-ngraph          |
| Server            | mcr.microsoft.com/onnxruntime/server  | :v0.4.0, :v0.5.0, v0.5.1, v1.0.0      | :latest                 |

---

# Building and using Docker images

## CPU
**Ubuntu 16.04, CPU, Python Bindings**

1. Build the docker image from the Dockerfile in this repository.
  ```
  docker build -t onnxruntime-source -f Dockerfile.source .
  ```

2. Run the Docker image

  ```
  docker run -it onnxruntime-source
  ```

## CUDA
**Ubuntu 16.04, CUDA 10.0, CuDNN 7**

1. Build the docker image from the Dockerfile in this repository.
  ```
  docker build -t onnxruntime-cuda -f Dockerfile.cuda .
  ```

2. Run the Docker image

  ```
  docker run -it onnxruntime-cuda
  ```

## nGraph
*Public Preview*

**Ubuntu 16.04, Python Bindings**

1. Build the docker image from the Dockerfile in this repository.
  ```
  docker build -t onnxruntime-ngraph -f Dockerfile.ngraph .
  ```

2. Run the Docker image

  ```
  docker run -it onnxruntime-ngraph
  ```

## TensorRT
**Ubuntu 16.04, TensorRT 5.0.2**

1. Build the docker image from the Dockerfile in this repository.
  ```
  docker build -t onnxruntime-trt -f Dockerfile.tensorrt .
  ```

2. Run the Docker image

  ```
  docker run -it onnxruntime-trt
  ```

## OpenVINO
*Public Preview*

**Ubuntu 16.04, Python Bindings**

1. Build the onnxruntime image for one of the accelerators supported below.

   Retrieve your docker image in one of the following ways.

    -  To build your docker image, download the OpenVINO online installer version 2019 R1.1 for Linux from [this link](https://software.intel.com/en-us/openvino-toolkit/choose-download) and copy the OpenVINO tar file to the same directory before building the Docker image. The online installer size is 16MB and the components needed for the accelerators are mentioned in the dockerfile. Providing the docker build argument DEVICE enables the onnxruntime build for that particular device. You can also provide arguments ONNXRUNTIME_REPO and ONNXRUNTIME_BRANCH to test that particular repo and branch. Default repository is http://github.com/microsoft/onnxruntime and default branch is master.
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

### OpenVINO on CPU

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

### OpenVINO on GPU

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
### OpenVINO on Myriad VPU Accelerator

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

### OpenVINO on VAD-M Accelerator Version

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
## ARM 32v7
*Public Preview*

The Dockerfile used in these instructions specifically targets Raspberry Pi 3/3+ running Raspbian Stretch. The same approach should work for other ARM devices, but may require some changes to the Dockerfile such as choosing a different base image (Line 0: `FROM ...`).

1. Install DockerCE on your development machine by following the instructions [here](https://docs.docker.com/install/)
2. Create an empty local directory
    ```bash
    mkdir onnx-build
    cd onnx-build
    ```
3. Save the Dockerfile to your new directory
    - [Dockerfile.arm32v7](./Dockerfile.arm32v7)
4. Run docker build

    This will build all the dependencies first, then build ONNX Runtime and its Python bindings. This will take several hours.
    ```bash
    docker build -t onnxruntime-arm32v7 -f Dockerfile.arm32v7 .
    ```
5. Note the full path of the `.whl` file

    - Reported at the end of the build, after the `# Build Output` line.
    - It should follow the format `onnxruntime-0.3.0-cp35-cp35m-linux_armv7l.whl`, but version number may have changed. You'll use this path to extract the wheel file later.
6. Check that the build succeeded

    Upon completion, you should see an image tagged `onnxruntime-arm32v7` in your list of docker images:
    ```bash
    docker images
    ```
7. Extract the Python wheel file from the docker image

    (Update the path/version of the `.whl` file with the one noted in step 5)
    ```bash
    docker create -ti --name onnxruntime_temp onnxruntime-arm32v7 bash
    docker cp onnxruntime_temp:/code/onnxruntime/build/Linux/MinSizeRel/dist/onnxruntime-0.3.0-cp35-cp35m-linux_armv7l.whl .
    docker rm -fv onnxruntime_temp
    ```
    This will save a copy of the wheel file, `onnxruntime-0.3.0-cp35-cp35m-linux_armv7l.whl`, to your working directory on your host machine.
8. Copy the wheel file (`onnxruntime-0.3.0-cp35-cp35m-linux_armv7l.whl`) to your Raspberry Pi or other ARM device
9. On device, install the ONNX Runtime wheel file
    ```bash
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip
    pip3 install numpy

    # Install ONNX Runtime
    # Important: Update path/version to match the name and location of your .whl file
    pip3 install onnxruntime-0.3.0-cp35-cp35m-linux_armv7l.whl
    ```
10. Test installation by following the instructions [here](https://microsoft.github.io/onnxruntime/)


## Nuphar
*Public Preview*

**Ubuntu 16.04, Python Bindings**

1. Build the docker image from the Dockerfile in this repository.
  ```
  docker build -t onnxruntime-nuphar -f Dockerfile.nuphar .
  ```

2. Run the Docker image

  ```
  docker run -it onnxruntime-nuphar
  ```

## ONNX Runtime Server
*Public Preview*

**Ubuntu 16.04**

1. Build the docker image from the Dockerfile in this repository
  ```
  docker build -t {docker_image_name} -f Dockerfile.server .
  ```

2. Run the ONNXRuntime server with the image created in step 1

  ```
  docker run -v {localModelAbsoluteFolder}:{dockerModelAbsoluteFolder} -p {your_local_port}:8001 {imageName} --model_path {dockerModelAbsolutePath}
  ```
3. Send HTTP requests to the container running ONNX Runtime Server

  Send HTTP requests to the docker container through the binding local port. Here is the full [usage document](https://github.com/Microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Server_Usage.md).
  ```
  curl  -X POST -d "@request.json" -H "Content-Type: application/json" http://0.0.0.0:{your_local_port}/v1/models/mymodel/versions/3:predict  
  ```
