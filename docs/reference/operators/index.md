---
title: Operators
parent: Reference
has_children: true
nav_order: 3
---
# ONNX Runtime Operators

ONNX Runtime supports the operators defined in the [ONNX specification](https://onnx.ai/onnx/operators/), as well as additional operators for mobile and web targets, and [contrib ops](ContribOperators.md) for extended functionality.

## Supported ONNX Operators

ONNX Runtime aims to support all standard ONNX operators. The specific set of operators available depends on the build configuration:

- **Full builds**: All standard ONNX opset operators plus optional contrib ops
- **Reduced builds** (mobile/web): A subset of operators; see [mobile package operator support](mobile_package_op_type_support_1.17.md) for the latest list

The full list of kernels registered in the current build is available in [OperatorKernels.md](OperatorKernels.md).

## Contrib Operators

ONNX Runtime provides additional operators beyond the ONNX specification via the `com.microsoft` domain. These are documented in [ContribOperators.md](ContribOperators.md).

## Mobile Operators

A reduced set of operators is available for mobile and web targets. These are documented in [MobileOps.md](MobileOps.md).

## Custom Operators

You can extend ONNX Runtime with your own operators. See:
- [Add a custom operator](add-custom-op.md)
- [Custom Python operators](custom-python-operator.md)
