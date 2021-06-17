
===========
API Summary
===========

Summary of public functions and classes exposed
in *ONNX Runtime*.

.. contents::
    :local:

OrtValue
=========
*ONNX Runtime* works with native Python data structures which are mapped into ONNX data formats :
Numpy arrays (tensors), dictionaries (maps), and a list of Numpy arrays (sequences).
The data backing these are on CPU.

*ONNX Runtime* supports a custom data structure that supports all ONNX data formats that allows users
to place the data backing these on a device, for example, on a CUDA supported device. This allows for
interesting *IOBinding* scenarios (discussed below). In addition, *ONNX Runtime* supports directly
working with *OrtValue* (s) while inferencing a model if provided as part of the input feed.

Below is an example showing creation of an *OrtValue* from a Numpy array while placing its backing memory
on a CUDA device:

.. code-block:: python

	#X is numpy array on cpu, create an OrtValue and place it on cuda device id = 0
	ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
	ortvalue.device_name()  # 'cuda'
	ortvalue.shape()  # shape of the numpy array X
	ortvalue.data_type()  # 'tensor(float)'
	ortvalue.is_tensor()  # 'True'
	np.array_equal(ortvalue.numpy(), X)  # 'True'

	#ortvalue can be provided as part of the input feed to a model
	ses = onnxruntime.InferenceSession('model.onnx')
	res = sess.run(["Y"], {"X": ortvalue})

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

A graph is executed on a device other than CPU, for instance CUDA. Users can 
use IOBinding to put input on CUDA as the follows.

.. code-block:: python

	#X is numpy array on cpu 
	session = onnxruntime.InferenceSession('model.onnx')
	io_binding = session.io_binding()
	# OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device 
	io_binding.bind_cpu_input('input', X)
	io_binding.bind_output('output')
	session.run_with_iobinding(io_binding)
	Y = io_binding.copy_outputs_to_cpu()[0]

Scenario 2:

The input data is on a device, users directly use the input. The output data is on CPU.

.. code-block:: python

	#X is numpy array on cpu
	X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
	session = onnxruntime.InferenceSession('model.onnx')
	io_binding = session.io_binding()
	io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
	io_binding.bind_output('output')
	session.run_with_iobinding(io_binding)
	Y = io_binding.copy_outputs_to_cpu()[0]

Scenario 3:

The input data and output data are both on a device, users directly use the input and also place output on the device.

.. code-block:: python

	#X is numpy array on cpu
	X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
	Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([3, 2], np.float32, 'cuda', 0)  # Change the shape to the actual shape of the output being bound
	session = onnxruntime.InferenceSession('model.onnx')
	io_binding = session.io_binding()
	io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
	io_binding.bind_output(name='output', device_type=Y_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=Y_ortvalue.shape(), buffer_ptr=Y_ortvalue.data_ptr())
	session.run_with_iobinding(io_binding)

Scenario 4:

Users can request *ONNX Runtime* to allocate an output on a device. This is particularly useful for dynamic shaped outputs.
Users can use the *get_outputs()* API to get access to the *OrtValue* (s) corresponding to the allocated output(s).
Users can thus consume the *ONNX Runtime* allocated memory for the output as an *OrtValue*.

.. code-block:: python

	#X is numpy array on cpu
	X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
	session = onnxruntime.InferenceSession('model.onnx')
	io_binding = session.io_binding()
	io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
	#Request ONNX Runtime to bind and allocate memory on CUDA for 'output'
	io_binding.bind_output('output', 'cuda')
	session.run_with_iobinding(io_binding)
	# The following call returns an OrtValue which has data allocated by ONNX Runtime on CUDA
	ort_output = io_binding.get_outputs()[0]


Scenario 5:

Users can bind *OrtValue* (s) directly.

.. code-block:: python

	#X is numpy array on cpu
	#X is numpy array on cpu
	X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
	Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([3, 2], np.float32, 'cuda', 0)  # Change the shape to the actual shape of the output being bound
	session = onnxruntime.InferenceSession('model.onnx')
	io_binding = session.io_binding()
	io_binding.bind_ortvalue_input('input', X_ortvalue)
	io_binding.bind_ortvalue_output('output', Y_ortvalue)
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
