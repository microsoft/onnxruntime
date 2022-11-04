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

|ONNX Runtime|CANN|OS|
|---|---|---|---|
|v1.12.1|6.0|Ubuntu 18.04<br/>Ubuntu 20.04<br/>CentOS 7.8|
|v1.13.1|6.0|Ubuntu 18.04<br/>Ubuntu 20.04<br/>CentOS 7.8|

## Build

For build instructions, please see the [BUILD page](../../build/eps.md#CANN).

## Install

Pre-built binaries of ONNX Runtime with CANN EP are published for most language bindings. Please reference [Install ORT](../../install).

## Samples

Currently, users can use C/C++ and Python API on CANN EP.

### C/C++

```c
const static OrtApi *g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

OrtSessionOptions *session_options;
g_ort->CreateSessionOptions(&session_options);

OrtCANNProviderOptions *cann_options = nullptr;
g_ort->CreateCANNProviderOptions(&cann_options);

std::vector<const char *> keys{"device_id", "max_opqueue_num", "npu_mem_limit", "arena_extend_strategy", "do_copy_in_default_stream"};
std::vector<const char *> values{"1", "10000", "2147483648", "kSameAsRequested", "1"};

g_ort->UpdateCANNProviderOptions(cann_options, keys.data(), values.data(), keys.size());

g_ort->SessionOptionsAppendExecutionProvider_CANN(session_options, cann_options);

// Finally, don't forget to release the provider options and session options
g_ort->ReleaseSessionOptions(session_options);
g_ort->ReleaseCANNProviderOptions(cann_options);
```

### Python

```python
import onnxruntime as ort

model_path = '<path to model>'

options = ort.SessionOptions()

providers = [
    ('CANNExecutionProvider', {
        'device_id': 0,
        'max_opqueue_num': 10000,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'npu_mem_limit': 2 * 1024 * 1024 * 1024,
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
```

## Supported ops

Following ops are supported by the CANN Execution Provider,

|Operator|Note|
|--------|------|
|ai.onnx:Add||
|ai.onnx:AveragePool|Only 2D Pool is supported.|
|ai.onnx:BatchNormalization||
|ai.onnx:Conv|Only 1D/2D Conv is supported.<br/>Weights and bias should be constant.|
|ai.onnx:Div||
|ai.onnx:Dropout||
|ai.onnx:Flatten||
|ai.onnx:Gemm|Input B should be constant.|
|ai.onnx:GlobalAveragePool|Only 2D Pool is supported.|
|ai.onnx:GlobalMaxPool|Only 2D Pool is supported.|
|ai.onnx:Identity||
|ai.onnx:MatMul|Input B should be constant.|
|ai.onnx:MaxPool|Only 2D Pool is supported.|
|ai.onnx:Mul||
|ai.onnx:Relu||
|ai.onnx:Sub||

## Additional Resources

Additional operator support and performance tuning will be added soon.

* [Ascend](https://www.hiascend.com/en/)
* [CANN](https://www.hiascend.com/en/software/cann)
