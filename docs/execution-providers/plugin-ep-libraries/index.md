---
title: Plugin Execution Provider Libraries
parent: Execution Providers
has_children: true
nav_order: 17
redirect_from: /docs/reference/execution-providers/Plugin-EP-Libraries
---

# Plugin Execution Provider Libraries

An ONNX Runtime Execution Provider (EP) executes model operations on one or more hardware accelerators (e.g., GPU, NPU, etc.). ONNX Runtime provides a variety of built-in EPs, such as the default CPU EP. To enable further extensibility, ONNX Runtime supports user-defined plugin EP libraries that an application can register with ONNX Runtime for use in an ONNX Runtime inference session.

## Available Plugin Execution Providers

These are some of the available plugin EP implementations.

### QNN Execution Provider

A plugin execution provider that brings Qualcomm hardware acceleration to ONNX Runtime — enabling high-performance AI inference on Qualcomm Snapdragon SoCs via the Qualcomm AI Runtime SDK (QAIRT).

References:
- [Documentation](https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md)
- [Repository](https://github.com/onnxruntime/onnxruntime-qnn)

## Plugin Execution Provider Libraries Reference

The other pages in this section provide information about using and implementing plugin EP libraries.
