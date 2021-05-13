---
title: Vitis AI
parent: Execution Providers
grand_parent: Reference
nav_order: 12
---

# Vitis-AI Execution Provider
{: .no_toc }

[Vitis-AI](https://github.com/Xilinx/Vitis-AI) is Xilinx's development stack for hardware-accelerated AI inference on Xilinx platforms, including both edge devices and Alveo cards. It consists of optimized IP, tools, libraries, models, and example designs. It is designed with high efficiency and ease of use in mind, unleashing the full potential of AI acceleration on Xilinx FPGA and ACAP.

The current Vitis-AI execution provider inside ONNXRuntime enables acceleration of Neural Network model inference using DPUv1. DPUv1 is a hardware accelerator for Convolutional Neural Networks (CNN) on top of the Xilinx [Alveo](https://www.xilinx.com/products/boards-and-kits/alveo.html) platform and targets U200 and U250 accelerator cards.


## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

The following table lists system requirements for running docker containers as well as Alveo cards.  


| **Component**                                       | **Requirement**                                            |
|-----------------------------------------------------|------------------------------------------------------------|
| Motherboard                                         | PCI Express 3\.0\-compliant with one dual\-width x16 slot  |
| System Power Supply                                 | 225W                                                       |
| Operating System                                    | Ubuntu 16\.04, 18\.04                                      |
|                                                     | CentOS 7\.4, 7\.5                                          |
|                                                     | RHEL 7\.4, 7\.5                                            |
| CPU                                                 | Intel i3/i5/i7/i9/Xeon 64-bit CPU                          |
| GPU \(Optional to accelerate quantization\)         | NVIDIA GPU with a compute capability > 3.0                 |
| CUDA Driver \(Optional to accelerate quantization\) | nvidia\-410                                                |
| FPGA                                                | Xilinx Alveo U200 or U250                                  |
| Docker Version                                      | 19\.03\.1                                                  |

## Build
See [Build instructions](../../how-to/build/eps.md#vitis-ai).

**Hardware setup and docker build**

1. Clone the Vitis AI repository:
    ```
    git clone https://github.com/xilinx/vitis-ai
    ```
2. Install the Docker, and add the user to the docker group. Link the user to docker installation instructions from the following docker's website:
    * https://docs.docker.com/install/linux/docker-ce/ubuntu/
    * https://docs.docker.com/install/linux/docker-ce/centos/
    * https://docs.docker.com/install/linux/linux-postinstall/
3. Any GPU instructions will have to be separated from Vitis AI.
4. Set up Vitis AI to target Alveo cards. To target Alveo cards with Vitis AI for machine learning workloads, you must install the following software components:
    * Xilinx Runtime (XRT)
    * Alveo Deployment Shells (DSAs)
    * Xilinx Resource Manager (XRM) (xbutler)
    * Xilinx Overlaybins (Accelerators to Dynamically Load - binary programming files)

    While it is possible to install all of these software components individually, a script has been provided to automatically install them at once. To do so:
      * Run the following commands:
        ```
        cd Vitis-AI/alveo/packages
        sudo su
        ./install.sh
        ```
      * Power cycle the system.
5. Build and start the ONNXRuntime Vitis-AI Docker Container.
   ```
   cd {onnxruntime-root}/dockerfiles
   docker build -t onnxruntime-vitisai -f Dockerfile.vitisai .
   ./scripts/docker_run_vitisai.sh
   ```
   
   Setup inside container
   ```
   source /opt/xilinx/xrt/setup.sh
   conda activate vitis-ai-tensorflow
   ```

## Usage

### On-the-fly quantization

Usually, to be able to accelerate inference of Neural Network models with Vitis-AI DPU accelerators, those models need to quantized upfront. In the ONNXRuntime Vitis-AI execution provider we make use of on-the-fly quantization to remove this additional preprocessing step. In this flow, one doesn't need to quantize his/her model upfront but can make use of the typical inference execution calls (InferenceSession.run) to quantize the model on-the-fly using the first N inputs that are provided (see more information below). This will set up and calibrate the Vitis-AI DPU and from that point onwards inference will be accelerated for all next inputs.

## Configuration Options

A couple of environment variables can be used to customize the Vitis-AI execution provider.

| **Environment Variable**   | **Default if unset**      | **Explanation**                                         |
|----------------------------|---------------------------|---------------------------------------------------------|
| PX_QUANT_SIZE              | 128                    | The number of inputs that will be used for quantization (necessary for Vitis-AI acceleration) |
| PX_BUILD_DIR               | Use the on-the-fly quantization flow | Loads the quantization and compilation information from the provided build directory and immediately starts Vitis-AI hardware acceleration. This configuration can be used if the model has been executed before using on-the-fly quantization during which the quantization and comilation information was cached in a build directory. |

### Samples

When using python, you can base yourself on the following example:

```
# Import pyxir before onnxruntime
import pyxir
import pyxir.frontend.onnx
import pyxir.contrib.dpuv1.dpuv1

import onnxruntime

# Add other imports 
# ...

# Load inputs and do preprocessing
# ...

# Create an inference session using the Vitis-AI execution provider
session = onnxruntime.InferenceSession('[model_file].onnx', None,["VitisAIExecutionProvider"])

# First N (default = 128) inputs are used for quantization calibration and will
#   be executed on the CPU
# This config can be changed by setting the 'PX_QUANT_SIZE' (e.g. export PX_QUANT_SIZE=64)
imput_name = [...]
outputs = [session.run([], {input_name: calib_inputs[i]})[0] for i in range(128)]

# Afterwards, computations will be accelerated on the FPGA
input_data = [...]
result = session.run([], {input_name: input_data})
```
