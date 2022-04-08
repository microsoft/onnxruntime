This directory contains ORT format models to test for backwards compatibility when we are forced to make an update that invalidates a kernel hash.

When this happens, first create a new directory for the currently released ORT version.

Find a model that uses the operator with the kernel hash change. The ONNX test data is generally a good place to do this. See cmake/external/onnx/onnx/backend/test/data/node

Convert the model to ORT format using the currently released ORT version. This model will contain the original hash.

e.g. 
Running `python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_level=basic ORTv1.10/not1.onnx` 
will create the ORT format model `not1.basic.ort`

Add both the ONNX and the ORT format models to the repository. 

See onnxruntime/test/providers/kernel_def_hash_test.cc for information on updating the backwards compatibility hash
information and unit tests to use the new model.