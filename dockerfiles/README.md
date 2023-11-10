# Dockerfiles
**Execution Providers**
- CPU: [Dockerfile](Dockerfile.source), [Instructions](#cpu)
- CUDA/cuDNN: [Dockerfile](Dockerfile.cuda), [Instructions](#cuda)
- MIGraphX: [Dockerfile](Dockerfile.migraphx), [Instructions](#migraphx)
- ROCm: [Dockerfile](Dockerfile.rocm), [Instructions](#rocm)
- OpenVINO: [Dockerfile](Dockerfile.openvino), [Instructions](#openvino)
- TensorRT: [Dockerfile](Dockerfile.tensorrt), [Instructions](#tensorrt)
- VitisAI: [Dockerfile](Dockerfile.vitisai)

**Platforms**
- ARM 32v7: [Dockerfile](Dockerfile.arm32v7), [Instructions](#arm-3264)
- ARM 64: [Dockerfile](Dockerfile.arm64), [Instructions](#arm-3264)
- NVIDIA Jetson TX1/TX2/Nano/Xavier: [Dockerfile](Dockerfile.jetson), [Instructions](#nvidia-jetson-tx1tx2nanoxavier)

**Other**
- ORT Training (torch-ort): [Dockerfiles](https://github.com/pytorch/ort/tree/main/docker)
- ONNX-Ecosystem (CPU + Converters): [Dockerfile](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/Dockerfile), [Instructions](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem)



# Instructions

## CPU
**Ubuntu 22.04, CPU, Python Bindings**

1. Update submodules
```
git submodule update --init
```

2. Build the docker image from the Dockerfile in this repository.
  ```
  docker build -t onnxruntime-source -f Dockerfile.source ..
  ```

3. Run the Docker image

  ```
  docker run -it onnxruntime-source
  ```

## CUDA
**Ubuntu 20.04, CUDA 11.4, CuDNN 8**

1. Update submodules
```
git submodule update --init
```

2. Build the docker image from the Dockerfile in this repository.
  ```
  docker build -t onnxruntime-cuda -f Dockerfile.cuda ..
  ```

3. Run the Docker image

  ```
  docker run --gpus all -it onnxruntime-cuda
  or
  nvidia-docker run -it onnxruntime-cuda

  ```

## TensorRT
**Ubuntu 20.04, CUDA 11.8, TensorRT 8.5.1**

1. Update submodules
```
git submodule update --init
```

2. Build the docker image from the Dockerfile in this repository.
  ```
  docker build -t onnxruntime-trt -f Dockerfile.tensorrt .
  ```

3. Run the Docker image

  ```
  docker run --gpus all -it onnxruntime-trt
  or
  nvidia-docker run -it onnxruntime-trt
  ```

## OpenVINO
*Public Preview*

**Ubuntu 20.04, Python & C# Bindings**
**RHEL 8.4, Python Binding**

### **1. Using pre-built container images for Python API**

The unified container image from [Dockerhub](https://hub.docker.com/repository/docker/openvino/onnxruntime_ep_ubuntu20) can be used to run an application on any of the target accelerators. In order to select the target accelerator, the application should explicitly specify the choice using the `device_type`  configuration option for OpenVINO Execution provider. Refer to [OpenVINO EP runtime configuration documentation](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#configuration-options) for details on specifying this option in the application code.
If the `device_type` runtime config option is not explicitly specified, CPU will be chosen as the hardware target execution.
### **2. Building from Dockerfile**

1. Build the onnxruntime image for one of the accelerators supported below.

   Retrieve your docker image in one of the following ways.

    -  Choose Dockerfile.openvino for Python API or Dockerfile.openvino-csharp for C# API as <Dockerfile> for building latest OpenVINO based Docker image for Ubuntu20.04 and Dockerfile.openvino-rhel for Python API for RHEL 8.4. Providing the docker build argument DEVICE enables the onnxruntime build for that particular device. You can also provide arguments ONNXRUNTIME_REPO and ONNXRUNTIME_BRANCH to test that particular repo and branch. Default repository is http://github.com/microsoft/onnxruntime and default branch is main.
       ```
       docker build --rm -t onnxruntime --build-arg DEVICE=$DEVICE -f <Dockerfile> .
       ```
    - Pull the official image from DockerHub.

2. DEVICE: Specifies the hardware target for building OpenVINO Execution Provider. Below are the options for different Intel target devices.

  | Device Option | Target Device |
  | --------- | -------- |
  | <code>CPU_FP32</code> | Intel<sup></sup> CPUs |
  | <code>CPU_FP16</code> | Intel<sup></sup> CPUs |
  | <code>GPU_FP32</code> |Intel<sup></sup> Integrated Graphics |
  | <code>GPU_FP16</code> | Intel<sup></sup> Integrated Graphics |
  | <code>MYRIAD_FP16</code> | Intel<sup></sup> Movidius<sup>TM</sup> USB sticks |
  | <code>VAD-M_FP16</code> | Intel<sup></sup> Vision Accelerator Design based on Movidius<sup>TM</sup> MyriadX VPUs |
  | <code>HETERO:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>...</code> | All Intel<sup>®</sup> silicons mentioned above |
  | <code>MULTI:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>...</code> | All Intel<sup>®</sup> silicons mentioned above |
  | <code>AUTO:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>...</code> | All Intel<sup>®</sup> silicons mentioned above |

  Specifying Hardware Target for HETERO or MULTI or AUTO Build:

  HETERO:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>..
  MULTI:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>..
  AUTO:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>..
  The <DEVICE_TYPE> can be any of these devices from this list ['CPU','GPU','MYRIAD','HDDL']

  A minimum of two DEVICE_TYPE'S should be specified for a valid HETERO or MULTI or AUTO Build.

  Example:
  HETERO:MYRIAD,CPU  HETERO:HDDL,GPU,CPU  MULTI:MYRIAD,GPU,CPU AUTO:GPU,CPU

*This is the hardware accelerator target that is enabled by **default** in the container image. After building the container image for one default target, the application may explicitly choose a different target at run time with the same container by using the [Dynamic device selction API](https://github.com/microsoft/onnxruntime/blob/main/docs/execution_providers/OpenVINO-ExecutionProvider.md#dynamic-device-selection).*


### OpenVINO on CPU

1. Build the docker image from the DockerFile in this repository.

     ```
     docker build --rm -t onnxruntime-cpu --build-arg DEVICE=CPU_FP32 -f <Dockerfile> .
     ```
2. Run the docker image
    ```
     docker run -it --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb onnxruntime-cpu:latest
    ```

### OpenVINO on GPU

1. Build the docker image from the DockerFile in this repository.
     ```
     docker build --rm -t onnxruntime-gpu --build-arg DEVICE=GPU_FP32 -f <Dockerfile> .
     ```
2. Run the docker image
    ```
    docker run -it --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb --device /dev/dri:/dev/dri onnxruntime-gpu:latest
    ```
    If your host system is Ubuntu 20, use the below command to run. Please find the alternative steps [here](https://github.com/openvinotoolkit/docker_ci/blob/master/configure_gpu_ubuntu20.md).
    ```
    docker run -it --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb --device /dev/dri:/dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) onnxruntime-gpu:latest
    ```

### OpenVINO on Myriad VPU Accelerator

1. Build the docker image from the DockerFile in this repository.
     ```
      docker build --rm -t onnxruntime-myriad --build-arg DEVICE=MYRIAD_FP16 -f <Dockerfile> .
     ```
2. Install the Myriad rules drivers on the host machine according to the reference in [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#additional-NCS-steps)

3. Run the docker image by mounting the device drivers
    ```
    docker run -it --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb onnxruntime-myriad:latest

    ```

### OpenVINO on VAD-M Accelerator Version

1. Download OpenVINO **Full package** for latest version for Linux on host machine from [this link](https://software.intel.com/en-us/openvino-toolkit/choose-download) and install it with the help of instructions from [this link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)

2. Install the drivers on the host machine according to the reference in [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux_ivad_vpu.html)

3. Build the docker image from the DockerFile in this repository.
     ```
      docker build --rm -t onnxruntime-vadm --build-arg DEVICE=VAD-M_FP16 -f <Dockerfile> .
     ```
4. Run hddldaemon on the host in a separate terminal session using the following steps:
    - Initialize the OpenVINO environment.
      ```
        source <openvino_install_directory>/setupvars.sh
      ```
    - Edit the hddl_service.config file from $HDDL_INSTALL_DIR/config/hddl_service.config and change the field “bypass_device_number” to 8.
    - Restart the hddl daemon for the changes to take effect.
     ```
      $HDDL_INSTALL_DIR/bin/hddldaemon
     ```
    - Note that if OpenVINO was installed with root permissions, this file has to be changed with the same permissions.
5. Run the docker image by mounting the device drivers
    ```
    docker run -itu root:root --rm --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb --mount type=bind,source=/var/tmp,destination=/var/tmp --device /dev/ion:/dev/ion  onnxruntime-vadm:latest
    ```

### OpenVINO on HETERO or Multi-Device Build

1. Build the docker image from the DockerFile in this repository.

     for HETERO:
     ```
      docker build --rm -t onnxruntime-HETERO --build-arg DEVICE=HETERO:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>... -f <Dockerfile> .
     ```

     for MULTI:
     ```
      docker build --rm -t onnxruntime-MULTI --build-arg DEVICE=MULTI:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>... -f <Dockerfile> .
     ```

     for AUTO:
     ```
      docker build --rm -t onnxruntime-AUTO --build-arg DEVICE=AUTO:<DEVICE_TYPE_1>,<DEVICE_TYPE_2>,<DEVICE_TYPE_3>... -f <Dockerfile> .
     ```

2. Install the required rules, drivers and other packages as required from the steps above for each of the DEVICE_TYPE accordingly that would be added for the HETERO or MULTI or AUTO device build type.

3. Run the docker image as mentioned in the above steps

## ARM 32/64

The build instructions are similar to x86 CPU. But if you want to build them on a x86 machine, you need to install qemu-user-static system package (outside of docker instances) first. Then

1. Update submodules
```
git submodule update --init
```

2. Build the docker image from the Dockerfile in this repository.
  ```bash
  docker build -t onnxruntime-source -f Dockerfile.arm64 ..
  ```

3. Run the Docker image

  ```bash
  docker run -it onnxruntime-source
  ```

For ARM32, please use Dockerfile.arm32v7 instead of Dockerfile.arm64.

## NVIDIA Jetson TX1/TX2/Nano/Xavier:

These instructions are for [JetPack SDK 4.4](https://developer.nvidia.com/embedded/jetpack).
The Dockerfile.jetson is using [NVIDIA L4T 32.4.3](https://developer.nvidia.com/embedded/linux-tegra) as base image.
Versions different from these may require modifications to these instructions.
Instructions assume you are on Jetson host in the root of onnxruntime git project clone(`https://github.com/microsoft/onnxruntime`)

Two-step installation is required:

1. Build Python 'wheel' for ONNX Runtime on host Jetson system; Pre-built Python wheels are also available at [Nvidia Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime).
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

## MIGraphX
**Ubuntu 20.04, ROCm5.4, AMDMIGraphX v1.2**

1. Build the docker image from the Dockerfile in this repository.
  ```
  docker build -t onnxruntime-migraphx -f Dockerfile.migraphx .
  ```

2. Run the Docker image

  ```
  docker run -it --device=/dev/kfd --device=/dev/dri --group-add video onnxruntime-migraphx
  ```

   ## ROCm
**Ubuntu 20.04, ROCm5.4**

1. Build the docker image from the Dockerfile in this repository.
  ```
  docker build -t onnxruntime-rocm -f Dockerfile.rocm .
  ```

2. Run the Docker image

  ```
  docker run -it --device=/dev/kfd --device=/dev/dri --group-add video onnxruntime-rocm
  ```
