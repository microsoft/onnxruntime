---
title: Limitations
parent: Deploy ONNX Runtime Mobile
grand_parent: How to
nav_order: 7
---

## Limitations

A minimal build has the following limitations:
  - No support for ONNX format models
    - Model must be converted to ORT format
  - No support for runtime optimizations
    - Optimizations are performed during conversion to ORT format
  - Limited support for runtime partitioning (assigning nodes in a model to specific execution providers)
    - Execution providers that statically register kernels (e.g. ONNX Runtime CPU Execution Provider) are supported by default
      - All execution providers that will be used at runtime MUST be registered when creating the ORT format model
    - Execution providers that compile nodes are optionally supported
      - currently this is limited to the NNAPI and CoreML Execution Providers
        - see [here](using-nnapi-coreml-with-ort-mobile) for details on using the NNAPI or CoreML Execution Providers with ONNX Runtime Mobile.

We do not currently offer backwards compatibility guarantees for ORT format models, as we will be expanding the capabilities in the short term and may need to update the internal format in an incompatible manner to accommodate these changes. You may need to regenerate the ORT format models to use with a future version of ONNX Runtime. Once the feature set stabilizes we will provide backwards compatibility guarantees.