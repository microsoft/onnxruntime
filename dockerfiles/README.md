# Docker Containers for ONNX Runtime

**Dockerfiles**


- CPU [Dockerfile](Dockerfile.source), [Instructions](#cpu)
- CUDA + CUDNN: [Dockerfile](Dockerfile.cuda), [Instructions](#cuda)
- nGraph: [Dockerfile](Dockerfile.ngraph), [Instructions](#ngraph)
- TensorRT: [Dockerfile](Dockerfile.tensorrt), [Instructions](#tensorrt)
- OpenVINO: [Dockerfile](Dockerfile.openvino), [Instructions](#openvino)
- Nuphar: [Dockerfile](Dockerfile.nuphar), [Instructions](#nuphar)
- ARM 32v7: [Dockerfile](Dockerfile.arm32v7), [Instructions](#arm-32v7)
- NVIDIA Jetson TX1/TX2/Nano/Xavier: [Dockerfile](Dockerfile.jetson), [Instructions](#nvidia-jetson-tx1tx2nanoxavier)
- ONNX-Ecosystem (CPU + Converters): [Dockerfile](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/Dockerfile), [Instructions](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem)
- ONNX Runtime Server: [Dockerfile](Dockerfile.server), [Instructions](#onnx-runtime-server)
- MIGraphX: [Dockerfile](Dockerfile.migraphx), [Instructions](#migraphx)

**Published Microsoft Container Registry (MCR) Images**

Use `docker pull` with any of the images and tags below to pull an image and try for yourself. Note that the CPU, CUDA, and TensorRT images include additional dependencies like miniconda for compatibility with AzureML image deployment.

**Example**: Run `docker pull mcr.microsoft.com/azureml/onnxruntime:latest-cuda` to pull the latest released docker image with ONNX Runtime GPU, CUDA, and CUDNN support.

| Build Flavor      | Base Image                            | ONNX Runtime Docker Image tags        | Latest                  |
|-------------------|---------------------------------------|---------------------------------------|-------------------------|
| Source (CPU)      | mcr.microsoft.com/azureml/onnxruntime | :v0.4.0, :v0.5.0, v0.5.1, :v1.0.0, :v1.2.0, :v1.3.0, :v1.4.0 | :latest |
| CUDA (GPU)        | mcr.microsoft.com/azureml/onnxruntime | :v0.4.0-cuda10.0-cudnn7, :v0.5.0-cuda10.1-cudnn7, :v0.5.1-cuda10.1-cudnn7, :v1.0.0-cuda10.1-cudnn7, :v1.2.0-cuda10.1-cudnn7, :v1.3.0-cuda10.1-cudnn7, :v1.4.0-cuda10.1-cudnn7 | :latest-cuda            |
| TensorRT (x86)    | mcr.microsoft.com/azureml/onnxruntime | :v0.4.0-tensorrt19.03, :v0.5.0-tensorrt19.06, :v1.0.0-tensorrt19.09, :v1.2.0-tensorrt20.01, :v1.3.0-tensorrt20.01, :v1.4.0-tensorrt20.01 | :latest-tensorrt        |
| OpenVino (VAD-M)  | mcr.microsoft.com/azureml/onnxruntime | :v0.5.0-openvino-r1.1-vadm, :v1.0.0-openvino-r1.1-vadm, :v1.4.0-openvino-2020.3.194-vadm | :latest-openvino-vadm |
| OpenVino (MYRIAD) | mcr.microsoft.com/azureml/onnxruntime | :v0.5.0-openvino-r1.1-myriad, :v1.0.0-openvino-r1.1-myriad, :v1.3.0-openvino-2020.2.120-myriad, :v1.4.0-openvino-2020.3.194-myriad | :latest-openvino-myriad |
| OpenVino (CPU)    | mcr.microsoft.com/azureml/onnxruntime | :v1.0.0-openvino-r1.1-cpu, :v1.3.0-openvino-2020.2.120-cpu, :v1.4.0-openvino-2020.3.194-cpu | :latest-openvino-cpu    |
| OpenVINO (GPU)    | mcr.microsoft.com/azureml/onnxruntime | :v1.3.0-openvino-2020.2.120-gpu, :v1.4.0-openvino-2020.3.194-gpu | :latest-openvino-gpu|
| nGraph            | mcr.microsoft.com/azureml/onnxruntime | :v1.0.0-ngraph-v0.26.0                | :latest-ngraph |
| Nuphar            | mcr.microsoft.com/azureml/onnxruntime |                                       | :latest-nuphar |
| Server            | mcr.microsoft.com/onnxruntime/server  | :v0.4.0, :v0.5.0, :v0.5.1, :v1.0.0      | :latest |
| MIGraphX (GPU)    | mcr.microsoft.com/azureml/onnxruntime | :v0.6                                 | :latest |
| Training ([usage](https://github.com/microsoft/onnxruntime-training-examples))| mcr.microsoft.com/azureml/onnxruntime-training | :0.1-rc1-openmpi4.0-cuda10.1-cudnn7.6-nccl2.4.8| 0.1-rc1-openmpi4.0-cuda10.1-cudnn7.6-nccl2.4.8|
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
  docker run --gpus all -it onnxruntime-cuda
  or
  nvidia-docker run -it onnxruntime-cuda

  ```

## nGraph
*Public Preview*

### **Deprecation Notice**

| | |
| --- | --- | 
| Deprecation Begins	| June 1, 2020 |
| Removal Date |	December 1, 2020 |

Starting with the OpenVINO™ toolkit 2020.2 release, all of the features previously available through nGraph have been merged into the OpenVINO™ toolkit. As a result, all the features previously available through ONNX RT Execution Provider for nGraph have been merged with ONNX RT Execution Provider for OpenVINO™ toolkit.

Therefore, ONNX RT Execution Provider for **nGraph** will be deprecated starting June 1, 2020 and will be completely removed on December 1, 2020. Users are recommended to migrate to the ONNX RT Execution Provider for OpenVINO™ toolkit as the unified solution for all AI inferencing on Intel® hardware. 

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
**Ubuntu 18.04, CUDA 11.0, TensorRT 7.1.3.4**

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

**Ubuntu 18.04, Python Bindings**

1. Build the onnxruntime image for one of the accelerators supported below.

   Retrieve your docker image in one of the following ways.

    -  Choose Dockerfile.openvino as the dockerfile for building an OpenVINO 2020.4 based Docker image. Providing the docker build argument DEVICE enables the onnxruntime build for that particular device. You can also provide arguments ONNXRUNTIME_REPO and ONNXRUNTIME_BRANCH to test that particular repo and branch. Default repository is http://github.com/microsoft/onnxruntime and default branch is master.
       ```
       docker build --rm -t onnxruntime --build-arg DEVICE=$DEVICE -f Dockerfile.openvino .
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
*This is the hardware accelerator target that is enabled by **default** in the container image. After building the container image for one default target, the application may explicitly choose a different target at run time with the same container by using the [Dynamic device selction API](https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/OpenVINO-ExecutionProvider.md#dynamic-device-selection).*


### OpenVINO on CPU

1. Build the docker image from the DockerFile in this repository.

     ```
     docker build --rm -t onnxruntime-cpu --build-arg DEVICE=CPU_FP32 --network host -f Dockerfile.openvino .
     ```
2. Run the docker image
    ```
     docker run -it onnxruntime-cpu
    ```

### OpenVINO on GPU

1. Build the docker image from the DockerFile in this repository.
     ```
      docker build --rm -t onnxruntime-gpu --build-arg DEVICE=GPU_FP32 --network host -f Dockerfile.openvino .
     ```
2. Run the docker image
    ```
    docker run -it --device /dev/dri:/dev/dri onnxruntime-gpu:latest
    ```
### OpenVINO on Myriad VPU Accelerator

1. Build the docker image from the DockerFile in this repository.
     ```
      docker build --rm -t onnxruntime-myriad --build-arg DEVICE=MYRIAD_FP16 --network host -f Dockerfile.openvino .
     ```
2. Install the Myriad rules drivers on the host machine according to the reference in [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#additional-NCS-steps)

3. Run the docker image by mounting the device drivers
    ```
    docker run -it --network host --privileged -v /dev:/dev  onnxruntime-myriad:latest

    ```

### OpenVINO on VAD-M Accelerator Version

1. Download OpenVINO **Full package** for version **2020.3** for Linux on host machine from [this link](https://software.intel.com/en-us/openvino-toolkit/choose-download) and install it with the help of instructions from [this link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)

2. Install the drivers on the host machine according to the reference in [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux_ivad_vpu.html)

3. Build the docker image from the DockerFile in this repository.
     ```
      docker build --rm -t onnxruntime-vadm --build-arg DEVICE=VAD-M_FP16 --network host -f Dockerfile.openvino .
     ```
4. Run hddldaemon on the host in a separate terminal session using the following command: 
     ```
      $HDDL_INSTALL_DIR/bin/hddldaemon
     ```
5. Run the docker image by mounting the device drivers
    ```
    docker run -it --device --mount type=bind,source=/var/tmp,destination=/var/tmp --device /dev/ion:/dev/ion  onnxruntime-vadm:latest

    ```
## ARM 32v7
*Public Preview*

The Dockerfile used in these instructions specifically targets Raspberry Pi 3/3+ running Raspbian Stretch. The same approach should work for other ARM devices, but may require some changes to the Dockerfile such as choosing a different base image (Line 0: `FROM ...`).

1. Install dependencies:

- DockerCE on your development machine by following the instructions [here](https://docs.docker.com/install/)
- ARM emulator: `sudo apt-get install -y qemu-user-static`

2. Create an empty local directory
    ```bash
    mkdir onnx-build
    cd onnx-build
    ```
3. Save the Dockerfile from this repo to your new directory: [Dockerfile.arm32v7](./Dockerfile.arm32v7)
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

## NVIDIA Jetson TX1/TX2/Nano/Xavier:

These instructions are for [JetPack SDK 4.4](https://developer.nvidia.com/embedded/jetpack).
The Dockerfile.jetson is using [NVIDIA L4T 32.4.3](https://developer.nvidia.com/embedded/linux-tegra) as base image.
Versions different from these may require modifications to these instructions.
Instructions assume you are on Jetson host in the root of onnxruntime git project clone(`https://github.com/microsoft/onnxruntime`)

Two-step installation is required:

1. Build Python 'wheel' for ONNX Runtime on host Jetson system;
2. Build Docker image using ONNX Runtime wheel from step 1. You can also install the wheel on the host directly.

Here are the build commands for each step:

1.1 Install ONNX Runtime build dependencies on Jetpack 4.4 host:
```
   sudo apt install -y --no-install-recommends \
    	build-essential software-properties-common cmake libopenblas-dev \
	libpython3.6-dev python3-pip python3-dev
```
1.2 Build ONNXRuntime Python wheel:
```
   ./build.sh --update --config Release --build --build_wheel \
   --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu
```
Note: You may add --use_tensorrt and --tensorrt_home options if you wish to use NVIDIA TensorRT (support is experimental), as well as any other options supported by [build.sh script](build.sh).

2. After the Python wheel is successfully built, use 'find' command for Docker to install the wheel inside new image:
```
   find . -name '*.whl' -print -exec sudo -H DOCKER_BUILDKIT=1 nvidia-docker build --build-arg WHEEL_FILE={} -f ./dockerfiles/Dockerfile.jetson . \;
```
Note: Resulting Docker image will have ONNX Runtime installed in /usr, and ONNX Runtime wheel copied to /onnxruntime directory.
Nothing else from ONNX Runtime source tree will be copied/installed to the image.

Note: When running the container you built in Docker, please either use 'nvidia-docker' command instead of 'docker', or use Docker command-line options to make sure NVIDIA runtime will be used and appropiate files mounted from host. Otherwise, CUDA libraries won't be found. You can also [set NVIDIA runtime as default in Docker](https://github.com/dusty-nv/jetson-containers#docker-default-runtime).

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

## MIGraphX 
**Ubuntu 16.04, rocm3.3, AMDMIGraphX v0.7**

1. Build the docker image from the Dockerfile in this repository.
  ```
  docker build -t onnxruntime-migraphx -f Dockerfile.migraphx .
  ```

2. Run the Docker image

  ```
  docker run -it --device=/dev/kfd --device=/dev/dri --group-add video onnxruntime-migraphx
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
