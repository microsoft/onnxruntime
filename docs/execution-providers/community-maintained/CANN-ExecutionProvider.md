---
title: Huawei - CANN
description: Instructions to execute ONNX Runtime with the Huawei CANN execution provider
grand_parent: Execution Providers
parent: Community-maintained
nav_order: 7
redirect_from: /docs/reference/execution-providers/CANN-ExecutionProvider
---

# CANN Execution Provider
{: .no_toc }

Huawei Compute Architecture for Neural Networks (CANN) is a heterogeneous computing architecture for AI scenarios and provides multi-layer programming interfaces to help users quickly build AI applications and services based on the Ascend platform.

Using CANN Excution Provider for ONNX Runtime can help you accelerate ONNX models on Huawei Ascend hardware.

The CANN Execution Provider (EP) for ONNX Runtime is developed by Huawei.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

Please reference table below for official CANN packages dependencies for the ONNX Runtime inferencing package.

|ONNX Runtime|CANN|
|---|---|
|v1.12.1|6.0.0|
|v1.13.1|6.0.0|
|v1.14.0|6.0.0|

## Build

For build instructions, please see the [BUILD page](../../build/eps.md#cann).

## Install

Pre-built binaries of ONNX Runtime with CANN EP are published for most language bindings. Please reference [Install ORT](../../install).

## Configuration Options

The CANN Execution Provider supports the following configuration options.

### device_id

The device ID.

Default value: 0

### npu_mem_limit

The size limit of the device memory arena in bytes. This size limit is only for the execution provider's arena. The total device memory usage may be higher.

### arena_extend_strategy

The strategy for extending the device memory arena.

Value                   | Description
-|-
kNextPowerOfTwo     | subsequent extensions extend by larger amounts (multiplied by powers of two)
kSameAsRequested    | extend by the requested amount

Default value: kNextPowerOfTwo

### do_copy_in_default_stream

Whether to do copies in the default stream or use separate streams. The recommended setting is true. If false, there are race conditions and possibly better performance.

Default value: true

### enable_cann_graph

Whether to use the graph inference engine to speed up performance. The recommended setting is true. If false, it will fall back to the single-operator inference engine.

Default value: true

## Samples

Currently, users can use C/C++ and Python API on CANN EP.

### Python

```python
import onnxruntime as ort

model_path = '<path to model>'

options = ort.SessionOptions()

providers = [
    (
        "CANNExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "npu_mem_limit": 2 * 1024 * 1024 * 1024,
            "do_copy_in_default_stream": True,
            "enable_cann_graph": True
        },
    ),
    "CPUExecutionProvider",
]

session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
```

### C/C++

```c
const static OrtApi *g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

OrtSessionOptions *session_options;
g_ort->CreateSessionOptions(&session_options);

OrtCANNProviderOptions *cann_options = nullptr;
g_ort->CreateCANNProviderOptions(&cann_options);

std::vector<const char *> keys{"device_id", "npu_mem_limit", "arena_extend_strategy", "do_copy_in_default_stream", "enable_cann_graph"};
std::vector<const char *> values{"1", "2147483648", "kSameAsRequested", "1", "1"};

g_ort->UpdateCANNProviderOptions(cann_options, keys.data(), values.data(), keys.size());

g_ort->SessionOptionsAppendExecutionProvider_CANN(session_options, cann_options);

// Finally, don't forget to release the provider options and session options
g_ort->ReleaseCANNProviderOptions(cann_options);
g_ort->ReleaseSessionOptions(session_options);
```

## Supported ops

Following ops are supported by the CANN Execution Provider in single-operator Inference mode.

|Operator|Note|
|--------|------|
|ai.onnx:Abs||
|ai.onnx:Add||
|ai.onnx:AveragePool|Only 2D Pool is supported.|
|ai.onnx:BatchNormalization||
|ai.onnx:Cast||
|ai.onnx:Ceil||
|ai.onnx:Conv|Only 2D Conv is supported.<br/>Weights and bias should be constant.|
|ai.onnx:Cos||
|ai.onnx:Div||
|ai.onnx:Dropout||
|ai.onnx:Exp||
|ai.onnx:Erf||
|ai.onnx:Flatten||
|ai.onnx:Floor||
|ai.onnx:Gemm||
|ai.onnx:GlobalAveragePool||
|ai.onnx:GlobalMaxPool||
|ai.onnx:Identity||
|ai.onnx:Log||
|ai.onnx:MatMul||
|ai.onnx:MaxPool|Only 2D Pool is supported.|
|ai.onnx:Mul||
|ai.onnx:Neg||
|ai.onnx:Reciprocal||
|ai.onnx:Relu||
|ai.onnx:Reshape||
|ai.onnx:Round||
|ai.onnx:Sin||
|ai.onnx:Sqrt||
|ai.onnx:Sub||
|ai.onnx:Transpose||

## Additional Resources

Additional operator support and performance tuning will be added soon.

* [Ascend](https://www.hiascend.com/en/)
* [CANN](https://www.hiascend.com/en/software/cann)
