
===========
API Summary
===========

Summary of public functions and classes exposed
in *ONNX Runtime*.

.. contents::
    :local:

Device
======

The package is compiled for a specific device, GPU or CPU.
The CPU implementation includes optimizations
such as MKL (Math Kernel Libary). The following function
indicates the chosen option:

.. autofunction:: onnxruntime.get_device

*onnxruntime* is compiled with a specific version of
`numpy <http://www.numpy.org/>`_. The following function
returns which version was used.

.. autofunction:: onnxruntime.compiled_with_numpy

Examples and datasets
=====================

The package contains a few models stored in ONNX format
used in the documentation. These don't need to be downloaded
as they are installed with the package.

.. autofunction:: onnxruntime.datasets.get_example

Load and run a model
====================

*ONNX Runtime* reads a model saved in ONNX format.
The main class *InferenceSession* wraps these functionalities
in a single place.

.. autoclass:: onnxruntime.ModelMetadata
    :members:

.. autoclass:: onnxruntime.InferenceSession
    :members:

.. autoclass:: onnxruntime.NodeArg
    :members:

.. autoclass:: onnxruntime.RunOptions
    :members:

.. autoclass:: onnxruntime.SessionOptions
    :members:

Backend
=======

In addition to the regular API which is optimized for performance and usability, 
*ONNX Runtime* also implements the
`ONNX backend API <https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md>`_
for verification of *ONNX* specification conformance.
The following functions are supported:

.. autofunction:: onnxruntime.backend.is_compatible

.. autofunction:: onnxruntime.backend.prepare

.. autofunction:: onnxruntime.backend.run

.. autofunction:: onnxruntime.backend.supports_device
