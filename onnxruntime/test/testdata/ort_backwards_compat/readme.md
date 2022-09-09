This directory contains ORT format models to test for backwards compatibility when we are forced to make an update that invalidates a kernel hash.

When this happens, first create a directory for the currently released ORT version if one doesn't already exist.

Find a model that uses the operator with the kernel hash change and copy it to the directory for the currently released ORT version.
The ONNX test data is generally a good place to do this. See cmake/external/onnx/onnx/backend/test/data/node.

Convert the model to ORT format using the currently released ORT version. This model will contain the original hash.

e.g.
Setting environment variable ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL=basic
and then running `python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_style=Fixed ORTv1.10/not1.onnx`
will create the ORT format model `not1.basic.ort`

Add both the ONNX and the ORT format models to the repository.

See onnxruntime/test/providers/kernel_def_hash_test.cc for information on updating the backwards compatibility hash
information and unit tests to use the new model.
