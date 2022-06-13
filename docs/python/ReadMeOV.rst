OpenVINO™ Execution Provider for ONNX Runtime
===============================================

`OpenVINO™ Execution Provider for ONNX Runtime <https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html>`_ is a product designed for ONNX Runtime developers who want to get started with OpenVINO™ in their inferencing applications. This product delivers  `OpenVINO™ <https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html>`_ inline optimizations which enhance inferencing performance with minimal code modifications. 

OpenVINO™ Execution Provider for ONNX Runtime accelerates inference across many  `AI models <https://github.com/onnx/models>`_ on a variety of Intel® hardware such as:
 - Intel® CPUs
 - Intel® integrated GPUs
 - Intel® Movidius™ Vision Processing Units - referred to as VPU.


Installation
------------

Requirements
^^^^^^^^^^^^

- Ubuntu 18.04, 20.04, RHEL(CPU only) or Windows 10 - 64 bit
- Python 3.7, 3.8 or 3.9

This package supports:
 - Intel® CPUs
 - Intel® integrated GPUs
 - Intel® Movidius™ Vision Processing Units (VPUs).

Please Note for VAD-M use Docker installation / Build from Source for Linux. 

``pip3 install onnxruntime-openvino==1.12.0``

Windows release supports only Python 3.9. Please install OpenVINO™ PyPi Package separately for Windows. 
For installation instructions on Windows please refer to  `OpenVINO™ Execution Provider for ONNX Runtime for Windows <https://github.com/intel/onnxruntime/releases/>`_.

This **OpenVINO™ Execution Provider for ONNX Runtime** Linux Wheels comes with pre-built libraries of OpenVINO™ version 2022.1.0 meaning you do not have to install OpenVINO™ separately. CXX11_ABI flag for pre built OpenVINO™ libraries is 0. The package also comes with `ONNX Runtime Training module <https://github.com/intel/onnxruntime/tree/master/orttraining/>`_ to enable inferencing of torch models using `ORT <https://github.com/pytorch/ort>`_. 

For more details on build and installation please refer to `Build <https://onnxruntime.ai/docs/build/eps.html#openvino>`_.

Usage
^^^^^

By default, Intel® CPU is used to run inference. However, you can change the default option to either Intel® integrated GPU or Intel® VPU for AI inferencing. Invoke the following function to change the hardware on which inferencing is done.

For more API calls and environment variables, see  `Usage <https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#configuration-options>`_.

Samples
^^^^^^^^

To see what you can do with **OpenVINO™ Execution Provider for ONNX Runtime**, explore the demos located in the  `Examples <https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/OpenVINO_EP>`_.

Docker Support
^^^^^^^^^^^^^^

The latest OpenVINO™ EP docker image can be downloaded from DockerHub. 
For more details see  `Docker ReadMe <https://hub.docker.com/r/openvino/onnxruntime_ep_ubuntu18>`_.


Prebuilt Images
^^^^^^^^^^^^^^^^

- Please find prebuilt docker images for Intel® CPU and Intel® iGPU on OpenVINO™ Execution Provider `Release Page <https://github.com/intel/onnxruntime/releases/>`_. 

License
^^^^^^^^

**OpenVINO™ Execution Provider for ONNX Runtime** is licensed under `MIT <https://github.com/microsoft/onnxruntime/blob/master/LICENSE>`_.
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.  

Support
^^^^^^^^

Please submit your questions, feature requests and bug reports via   `GitHub Issues <https://github.com/microsoft/onnxruntime/issues>`_.

How to Contribute
^^^^^^^^^^^^^^^^^^

We welcome community contributions to **OpenVINO™ Execution Provider for ONNX Runtime**. If you have an idea for improvement:

* Share your proposal via  `GitHub Issues <https://github.com/microsoft/onnxruntime/issues>`_.
* Submit a  `Pull Request <https://github.com/microsoft/onnxruntime/pulls>`_.