
===
API
===

.. contents::
    :local:

API Overview
============

*ONNX Runtime* loads and runs inference on a model in ONNX format, or ORT format (for memory and disk constrained environments).

The main class *InferenceSession* wraps model loading and running, as well as user specified configuration.

The data consumed by the model and the outputs that the model produces can be provided in a number of different ways.

Data on CPU
-----------

*ONNX Runtime* works with native Python data structures which are mapped into ONNX data formats:
Numpy arrays (tensors), dictionaries (maps), and a list of Numpy arrays (sequences).
The data backing these are on CPU.

Below is an example showing creation of an *OrtValue* from a Numpy array while placing its backing memory
on a CUDA device:

Scenario 1:

.. code-block:: python

	# X is numpy array on cpu, create an OrtValue and place it on cuda device id = 0
	ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
	ortvalue.device_name()  # 'cuda'
	ortvalue.shape()  # shape of the numpy array X
	ortvalue.data_type()  # 'tensor(float)'
	ortvalue.is_tensor()  # 'True'
	np.array_equal(ortvalue.numpy(), X)  # 'True'

	# ortvalue can be provided as part of the input feed to a model
	ses = onnxruntime.InferenceSession('model.onnx')
	res = sess.run(["Y"], {"X": ortvalue})

By default, *ONNX Runtime* always places input(s) and output(s) on CPU, which 
may not optimal if the input or output is consumed and produced on a device
other than CPU because it introduces data copy between CPU and the device.
See the sections below for way to minimize data copying and maximize I/O throughput.

Data on device
--------------

*ONNX Runtime* supports a custom data structure that supports all ONNX data formats that allows users
to place the data backing these on a device, for example, on a CUDA supported device. In ONNX Runtime, this called `IOBinding`.

To use the `IOBinding` feature the `InferenceSession.run()` is replaced by `InferenceSession.run_with_iobinding()`.


(In the following code snippets, *model.onnx* is the model to execute, 
*X* is the input data to feed, and *Y* is the output data.)

Scenario 2:

A graph is executed on a device other than CPU, for instance CUDA. Users can 
use IOBinding to put input on CUDA as the follows.

.. code-block:: python

	# X is numpy array on cpu 
	session = onnxruntime.InferenceSession('model.onnx')
	io_binding = session.io_binding()
	# OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device 
	io_binding.bind_cpu_input('input', X)
	io_binding.bind_output('output')
	session.run_with_iobinding(io_binding)
	Y = io_binding.copy_outputs_to_cpu()[0]

Scenario 3:

The input data is on a device, users directly use the input. The output data is on CPU.

.. code-block:: python

	# X is numpy array on cpu
	X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
	session = onnxruntime.InferenceSession('model.onnx')
	io_binding = session.io_binding()
	io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
	io_binding.bind_output('output')
	session.run_with_iobinding(io_binding)
	Y = io_binding.copy_outputs_to_cpu()[0]

Scenario 4:

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

Scenario 5:

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


Access data directly
--------------------

In addition, *ONNX Runtime* supports directly working with *OrtValue* (s) while inferencing a model if provided as part of the input feed.

but you can also provide pointers to Pytorch tensor storage

Scenario 6:

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


Scenario 7:

You can also bind inputs and outputs directly to a PyTorch tensor.

.. code-block:: python

    io_binding = session.io_binding()
    for input_onnx in session.get_inputs():
        tensor: torch.Tensor = inputs[input_onnx.name]
        tensor = tensor.contiguous()
        if tensor.dtype in [torch.int64, torch.long]:
            # int32 mandatory as input of bindings, int64 not supported
            tensor = tensor.type(dtype=torch.int32).to(device)
        io_binding.bind_input(
            name=input_onnx.name,
            device_type=device,
            device_id=device_id,
            element_type=torch_to_numpy_dtype_dict[tensor.dtype],
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )
        inputs[input_onnx.name] = tensor
    outputs = dict()
    output_shapes = ...
    for axis_name, shape in output_shapes.items():
        tensor = torch.empty(shape, dtype=torch.float32, device=device).contiguous()
        outputs[axis_name] = tensor
        binding.bind_output(
            name=axis_name,
            device_type=device,
            device_id=device_id,
            element_type=np.float32,  # hard coded output type
            shape=tuple(shape),
            buffer_ptr=tensor.data_ptr(),
        )
    session.run_with_iobinding(binding)

API Details
===========


Main class
----------

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

OrtDevice
^^^^^^^^^

.. autoclass:: onnxruntime.OrtDevice
    :members:

Internal classes
----------------

These classes cannot be instantiated by users but they are returned
by methods or functions of this libary.

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
`ONNX backend API <https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md>`_
for verification of *ONNX* specification conformance.
The following functions are supported:

.. autofunction:: onnxruntime.backend.is_compatible

.. autofunction:: onnxruntime.backend.prepare

.. autofunction:: onnxruntime.backend.run

.. autofunction:: onnxruntime.backend.supports_device
