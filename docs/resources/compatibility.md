---
title: Operator compatibility
parent: Resources
nav_order: 1
---

# ONNX and operator compatibility

Supporting models based on the standard [ONNX](https://onnx.ai) format, the runtime is compatible with PyTorch, scikit-learn, TensorFlow, Keras, and all other frameworks and tools that support the interoperable format.

* [Getting ONNX models - tutorials](https://github.com/onnx/tutorials#getting-onnx-models)

ONNX Runtime is up to date and backwards compatible with all operators (both DNN and traditional ML) since ONNX v1.2.1+. [(ONNX compatibility details)](docs/Versioning.md). Newer versions of ONNX Runtime support all models that worked with prior versions, so updates should not break integrations. 

* [Supported operators/types](resources/operators/OperatorKernels.md)
  * *Operators not supported in the current ONNX spec may be available as a [Contrib Operator](resource/operators/ContribOperators.md)*
* [Extensibility: Add a custom operator/kernel](docs/AddingCustomOp.md)
