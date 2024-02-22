
===
API
===

API Overview
============

*ONNX Runtime* loads and runs inference on a model in ONNX graph format, or ORT format (for memory and disk constrained environments).

The data consumed and produced by the model can be specified and accessed in the way that best matches your scenario.

Load and run a model
--------------------

InferenceSession is the main class of ONNX Runtime. It is used to load and run an ONNX model,
as well as specify environment and application configuration options.

.. code-block:: python

	session = onnxruntime.InferenceSession('model.onnx')

	outputs = session.run([output names], inputs)

ONNX and ORT format models consist of a graph of computations, modeled as operators,
and implemented as optimized operator kernels for different hardware targets.
ONNX Runtime orchestrates the execution of operator kernels via `execution providers`.
An execution provider contains the set of kernels for a specific execution target (CPU, GPU, IoT etc).
Execution provides are configured using the `providers` parameter. Kernels from different execution
providers are chosen in the priority order given in the list of providers. In the example below
if there is a kernel in the CUDA execution provider ONNX Runtime executes that on GPU. If not
the kernel is executed on CPU.

.. code-block:: python

	session = onnxruntime.InferenceSession(
		model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
	)

The list of available execution providers can be found here: `Execution Providers <https://onnxruntime.ai/docs/execution-providers>`_.

Since ONNX Runtime 1.10, you must explicitly specify the execution provider for your target.
Running on CPU is the only time the API allows no explicit setting of the `provider` parameter.
In the examples that follow, the `CUDAExecutionProvider` and `CPUExecutionProvider` are used, assuming the application is running on NVIDIA GPUs.
Replace these with the execution provider specific to your environment.

You can supply other session configurations via the `session options` parameter. For example, to enable
profiling on the session:

.. code-block:: python

	options = onnxruntime.SessionOptions()
	options.enable_profiling=True
	session = onnxruntime.InferenceSession(
		'model.onnx',
		sess_options=options,
		providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
	)


Data inputs and outputs
-----------------------

The ONNX Runtime Inference Session consumes and produces data using its OrtValue class.

Data on CPU
^^^^^^^^^^^

On CPU (the default), OrtValues can be mapped to and from native Python data structures: numpy arrays, dictionaries and lists of
numpy arrays.

.. code-block:: python

	# X is numpy array on cpu
	ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X)
	ortvalue.device_name()  # 'cpu'
	ortvalue.shape()        # shape of the numpy array X
	ortvalue.data_type()    # 'tensor(float)'
	ortvalue.is_tensor()    # 'True'
	np.array_equal(ortvalue.numpy(), X)  # 'True'

	# ortvalue can be provided as part of the input feed to a model
	session = onnxruntime.InferenceSession(
		'model.onnx',
		providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
	)
	results = session.run(["Y"], {"X": ortvalue})

By default, *ONNX Runtime* always places input(s) and output(s) on CPU. Having the data on CPU
may not optimal if the input or output is consumed and produced on a device
other than CPU because it introduces data copy between CPU and the device.


Data on device
^^^^^^^^^^^^^^

*ONNX Runtime* supports a custom data structure that supports all ONNX data formats that allows users
to place the data backing these on a device, for example, on a CUDA supported device. In ONNX Runtime,
this called `IOBinding`.

To use the `IOBinding` feature, replace `InferenceSession.run()` with `InferenceSession.run_with_iobinding()`.

A graph is executed on a device other than CPU, for instance CUDA. Users can
use IOBinding to copy the data onto the GPU.

.. code-block:: python

	# X is numpy array on cpu
	session = onnxruntime.InferenceSession(
		'model.onnx',
		providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
	)
	io_binding = session.io_binding()
	# OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device
	io_binding.bind_cpu_input('input', X)
	io_binding.bind_output('output')
	session.run_with_iobinding(io_binding)
	Y = io_binding.copy_outputs_to_cpu()[0]

The input data is on a device, users directly use the input. The output data is on CPU.

.. code-block:: python

	# X is numpy array on cpu
	X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
	session = onnxruntime.InferenceSession(
		'model.onnx',
		providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
	)
	io_binding = session.io_binding()
	io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
	io_binding.bind_output('output')
	session.run_with_iobinding(io_binding)
	Y = io_binding.copy_outputs_to_cpu()[0]

The input data and output data are both on a device, users directly use the input and also place output on the device.

.. code-block:: python

	#X is numpy array on cpu
	X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
	Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([3, 2], np.float32, 'cuda', 0)  # Change the shape to the actual shape of the output being bound
	session = onnxruntime.InferenceSession(
		'model.onnx',
		providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
	)
	io_binding = session.io_binding()
	io_binding.bind_input(
		name='input',
		device_type=X_ortvalue.device_name(),
		device_id=0,
		element_type=np.float32,
		shape=X_ortvalue.shape(),
		buffer_ptr=X_ortvalue.data_ptr()
	)
	io_binding.bind_output(
		name='output',
		device_type=Y_ortvalue.device_name(),
		device_id=0,
		element_type=np.float32,
		shape=Y_ortvalue.shape(),
		buffer_ptr=Y_ortvalue.data_ptr()
	)
	session.run_with_iobinding(io_binding)


Users can request *ONNX Runtime* to allocate an output on a device. This is particularly useful for dynamic shaped outputs.
Users can use the *get_outputs()* API to get access to the *OrtValue* (s) corresponding to the allocated output(s).
Users can thus consume the *ONNX Runtime* allocated memory for the output as an *OrtValue*.

.. code-block:: python

	#X is numpy array on cpu
	X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
	session = onnxruntime.InferenceSession(
		'model.onnx',
		providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
	)
	io_binding = session.io_binding()
	io_binding.bind_input(
		name='input',
		device_type=X_ortvalue.device_name(),
		device_id=0,
		element_type=np.float32,
		shape=X_ortvalue.shape(),
		buffer_ptr=X_ortvalue.data_ptr()
	)
	#Request ONNX Runtime to bind and allocate memory on CUDA for 'output'
	io_binding.bind_output('output', 'cuda')
	session.run_with_iobinding(io_binding)
	# The following call returns an OrtValue which has data allocated by ONNX Runtime on CUDA
	ort_output = io_binding.get_outputs()[0]


In addition, *ONNX Runtime* supports directly working with *OrtValue* (s) while inferencing a model if provided as part of the input feed.

Users can bind *OrtValue* (s) directly.

.. code-block:: python

	#X is numpy array on cpu
	#X is numpy array on cpu
	X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
	Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([3, 2], np.float32, 'cuda', 0)  # Change the shape to the actual shape of the output being bound
	session = onnxruntime.InferenceSession(
		'model.onnx',
		providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
	)
	io_binding = session.io_binding()
	io_binding.bind_ortvalue_input('input', X_ortvalue)
	io_binding.bind_ortvalue_output('output', Y_ortvalue)
	session.run_with_iobinding(io_binding)


You can also bind inputs and outputs directly to a PyTorch tensor.

.. code-block:: python

    # X is a PyTorch tensor on device
    session = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
    binding = session.io_binding()

    X_tensor = X.contiguous()

    binding.bind_input(
        name='X',
        device_type='cuda',
        device_id=0,
        element_type=np.float32,
        shape=tuple(x_tensor.shape),
        buffer_ptr=x_tensor.data_ptr(),
        )

    ## Allocate the PyTorch tensor for the model output
    Y_shape = ... # You need to specify the output PyTorch tensor shape
    Y_tensor = torch.empty(Y_shape, dtype=torch.float32, device='cuda:0').contiguous()
    binding.bind_output(
        name='Y',
        device_type='cuda',
        device_id=0,
        element_type=np.float32,
        shape=tuple(Y_tensor.shape),
        buffer_ptr=Y_tensor.data_ptr(),
    )

    session.run_with_iobinding(binding)
    
You can also see code examples of this API in in the `ONNX Runtime inferences examples <https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py>`_.


API Details
===========


InferenceSession
----------------

.. autoclass:: onnxruntime.InferenceSession
    :members:
    :inherited-members:

Options
-------

RunOptions
^^^^^^^^^^

.. autoclass:: onnxruntime.RunOptions
    :members:

SessionOptions
^^^^^^^^^^^^^^

.. autoclass:: onnxruntime.SessionOptions
    :members:

.. autoclass:: onnxruntime.ExecutionMode
    :members:

.. autoclass:: onnxruntime.ExecutionOrder
    :members:

.. autoclass:: onnxruntime.GraphOptimizationLevel
    :members:

.. autoclass:: onnxruntime.OrtAllocatorType
    :members:

.. autoclass:: onnxruntime.OrtArenaCfg
    :members:

.. autoclass:: onnxruntime.OrtMemoryInfo
    :members:

.. autoclass:: onnxruntime.OrtMemType
    :members:

Functions
---------

Allocators
^^^^^^^^^^

.. autofunction:: onnxruntime.create_and_register_allocator

.. autofunction:: onnxruntime.create_and_register_allocator_v2

Telemetry events
^^^^^^^^^^^^^^^^

.. autofunction:: onnxruntime.disable_telemetry_events

.. autofunction:: onnxruntime.enable_telemetry_events

Providers
^^^^^^^^^

.. autofunction:: onnxruntime.get_all_providers

.. autofunction:: onnxruntime.get_available_providers

Build, Version
^^^^^^^^^^^^^^

.. autofunction:: onnxruntime.get_build_info

.. autofunction:: onnxruntime.get_version_string

.. autofunction:: onnxruntime.has_collective_ops

Device
^^^^^^

.. autofunction:: onnxruntime.get_device

Logging
^^^^^^^

.. autofunction:: onnxruntime.set_default_logger_severity

.. autofunction:: onnxruntime.set_default_logger_verbosity

Random
^^^^^^

.. autofunction:: onnxruntime.set_seed

Data
----

OrtValue
^^^^^^^^

.. autoclass:: onnxruntime.OrtValue
    :members:

SparseTensor
^^^^^^^^^^^^

.. autoclass:: onnxruntime.SparseTensor
    :members:

Devices
-------

IOBinding
^^^^^^^^^

.. autoclass:: onnxruntime.IOBinding
    :members:

.. autoclass:: onnxruntime.SessionIOBinding
    :members:

OrtDevice
^^^^^^^^^

.. autoclass:: onnxruntime.OrtDevice
    :members:

Internal classes
----------------

These classes cannot be instantiated by users but they are returned
by methods or functions of this library.

ModelMetadata
^^^^^^^^^^^^^

.. autoclass:: onnxruntime.ModelMetadata
    :members:

NodeArg
^^^^^^^

.. autoclass:: onnxruntime.NodeArg
    :members:

Backend
=======

In addition to the regular API which is optimized for performance and usability, 
*ONNX Runtime* also implements the
`ONNX backend API <https://github.com/onnx/onnx/blob/main/docs/ImplementingAnOnnxBackend.md>`_
for verification of *ONNX* specification conformance.
The following functions are supported:

.. autofunction:: onnxruntime.backend.is_compatible

.. autofunction:: onnxruntime.backend.prepare

.. autofunction:: onnxruntime.backend.run

.. autofunction:: onnxruntime.backend.supports_device
