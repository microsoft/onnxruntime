---
title: Operators
parent: Reference
has_children: true
nav_order: 3
---
# ONNX Runtime Operators

ONNX Runtime implements operators from the [ONNX specification](https://onnx.ai/onnx/operators/) and provides [contrib ops](ContribOperators.md) for additional functionality.

## Supported ONNX Operators

The operators available at runtime depend on the ONNX Runtime version and build configuration:

- **Full builds** include the standard and contrib operator kernels registered for that release.
- **Reduced builds** include only the operators and types selected by their configuration. See [Reduced operator config file](reduced-operator-config-file.md).

The full list of kernels registered in the current build is available in [OperatorKernels.md](OperatorKernels.md).

## Contrib Operators

ONNX Runtime provides additional operators beyond the ONNX specification via the `com.microsoft` domain. These are documented in [ContribOperators.md](ContribOperators.md).

## Mobile Operators

Current pre-built mobile packages include full operator support. To minimize a custom mobile or web build, generate a reduced operator configuration for the models you plan to run. [MobileOps.md](MobileOps.md) retains the operator lists for older reduced mobile packages.

## Custom Operators

You can extend ONNX Runtime with your own operators. See:

- [Add a custom operator](add-custom-op.md)
- [Custom Python operators](custom-python-operator.md)
