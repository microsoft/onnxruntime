# ORT Format Update in 1.13

In ONNX Runtime 1.13, there was a breaking change to the
[ORT format](https://onnxruntime.ai/docs/reference/ort-format-models.html) in order to enable additional execution
providers with statically registered kernels in a minimal build.
More details can be found [here](../onnxruntime/core/flatbuffers/schema/README.md#version-5).

Unfortunately, this means that any older models (prior to ORT format version 5) will no longer work with ONNX Runtime
1.13 or later and must be re-converted.
Please refer
[here](https://onnxruntime.ai/docs/reference/ort-format-models.html#convert-onnx-models-to-ort-format) for instructions
on how to convert an ONNX model to ORT format.
