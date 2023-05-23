# ORT Format Update in 1.13

In ONNX Runtime 1.13, there was a breaking change to the
[ORT format](https://onnxruntime.ai/docs/reference/ort-format-models.html) (version 5) in order to enable additional
execution providers with statically registered kernels in a minimal build.
More details can be found [here](../onnxruntime/core/flatbuffers/schema/README.md#version-5).

## Backwards Compatibility

### ONNX Runtime 1.13
Any older models (prior to ORT format version 5) will no longer work with ONNX Runtime 1.13 and must be re-converted.

### ONNX Runtime 1.14+
ONNX Runtime 1.14+ provides limited backwards compatibility for loading older models (prior to ORT format version 5).
- In a full build, older models may be loaded but any saved runtime optimizations will be ignored.
- In a minimal build, older models cannot be loaded.

An older model may be re-converted.

It is also possible to load an older ORT format model in a full build and then save it back out as an ORT format model.
This process may be used to upgrade an ORT format model. However, any saved runtime optimizations from the older model
will be ignored.

## Re-converting an ORT format model
Please refer
[here](https://onnxruntime.ai/docs/reference/ort-format-models.html#convert-onnx-models-to-ort-format) for instructions
on how to convert an ONNX model to ORT format.
