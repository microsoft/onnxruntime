OpenVINO™ Execution Provider for ONNX Runtime
===============================================

`OpenVINO™ Execution Provider for ONNX Runtime <https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html>`_ is a product designed for ONNX Runtime developers who want to get started with OpenVINO™ in their inferencing applications. This product delivers  `OpenVINO™ <https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html>`_ inline optimizations which enhance inferencing performance with minimal code modifications.

OpenVINO™ Execution Provider for ONNX Runtime accelerates inference across many  `AI models <https://github.com/onnx/models>`_ on a variety of Intel® hardware such as:
 - Intel® CPUs
 - Intel® integrated GPUs
 - Intel® discrete GPUs
 - Intel® integrated NPUs

Installation
------------

Requirements
^^^^^^^^^^^^

- Ubuntu 18.04, 20.04 or Windows 10 - 64 bit
- Python 3.10, 3.11, 3.12 and 3.13 for Windows and Linux

This package supports:
 - Intel® CPUs
 - Intel® integrated GPUs
 - Intel® discrete GPUs
 - Intel® integrated NPUs

``pip3 install onnxruntime-openvino``

Please install OpenVINO™ PyPi Package separately for Windows.
For installation instructions on Windows please refer to  `OpenVINO™ Execution Provider for ONNX Runtime for Windows <https://github.com/intel/onnxruntime/releases/>`_.

**OpenVINO™ Execution Provider for ONNX Runtime** Linux Wheels comes with pre-built libraries of OpenVINO™ version 2025.1.0 eliminating the need to install OpenVINO™ separately.

For more details on build and installation please refer to `Build <https://onnxruntime.ai/docs/build/eps.html#openvino>`_.

Usage
^^^^^

By default, Intel® CPU is used to run inference. However, you can change the default option to either Intel® integrated GPU, discrete GPU, integrated NPU.
Invoke `the provider config device type argument <https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#summary-of-options>`_ to change the hardware on which inferencing is done.

For more API calls and environment variables, see  `Usage <https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#configuration-options>`_.

Samples
^^^^^^^^

To see what you can do with **OpenVINO™ Execution Provider for ONNX Runtime**, explore the demos located in the  `Examples <https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/OpenVINO_EP>`_.

License
^^^^^^^^

**OpenVINO™ Execution Provider for ONNX Runtime** is licensed under `MIT <https://github.com/microsoft/onnxruntime/blob/main/LICENSE>`_.
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
