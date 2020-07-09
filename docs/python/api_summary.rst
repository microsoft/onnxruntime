
===========
API Summary
===========

Summary of public functions and classes exposed
in *ONNX Runtime*.

.. contents::
    :local:

IOBinding
=========

By default, *ONNX Runtime* always places input(s) and output(s) on CPU, which 
is not optimal if the input or output is consumed and produced on a device
other than CPU because it introduces data copy between CPU and the device. 
*ONNX Runtime* provides a feature, *IO Binding*, which addresses this issue by
enabling users to specify which device to place input(s) and output(s) on. 
Here are scenarios to use this feature. 

(In the following code snippets, *model.onnx* is the model to execute, 
*X* is the input data to feed, and *Y* is the output data.)

Scenario 1:

A graph is executed on a deivce other than CPU, for instance CUDA. Users can 
use IOBinding to put input on CUDA as the follows.

.. code-block:: python

	#X is numpy array on cpu 
	session = onnxruntime.InferenceSession('model.onnx')
	io_binding = session.io_binding()
	io_binding.bind_cpu_input('input', X)
	io_binding.bind_output('output')
	session.run_with_iobinding(io_binding)
	Y = io_binding.copy_outputs_to_cpu()[0]

Scenario 2:

The input data is on a device, users direclty use the input. The output data is on CPU.

.. code-block:: python

	session = onnxruntime.InferenceSession('model.onnx')
	io_binding = session.io_binding()
	io_binding.bind_input(name='input', device_type=X.device.type, device_id=0, element_type=np.float32, shape=list(X.size()), buffer_ptr=X.data_ptr())
	io_binding.bind_output('output')
	session.run_with_iobinding(io_binding)
	Y = io_binding.copy_outputs_to_cpu()[0]

Scenario 3:

The input data on a dveice, users directly use the input and also place output on the device:

.. code-block:: python

	session = onnxruntime.InferenceSession('model.onnx')
	io_binding = session.io_binding()
	io_binding.bind_input(name='input', device_type=X.device.type, device_id=0, element_type=np.float32, shape=list(X.size()), buffer_ptr=X.data_ptr())
	io_binding.bind_output(name='output', device_type=Y.device.type, device_id=0, element_type=np.float32, shape=list(Y.size()), buffer_ptr=Y.data_ptr())
	session.run_with_iobinding(io_binding)

Device
======

The package is compiled for a specific device, GPU or CPU.
The CPU implementation includes optimizations
such as MKL (Math Kernel Libary). The following function
indicates the chosen option:

.. autofunction:: onnxruntime.get_device

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
