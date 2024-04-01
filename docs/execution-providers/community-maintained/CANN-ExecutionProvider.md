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

## Install

Pre-built binaries of ONNX Runtime with CANN EP are published, but only for python currently, please refer to [onnxruntime-cann](https://pypi.org/project/onnxruntime-cann/).

## Requirements

Please reference table below for official CANN packages dependencies for the ONNX Runtime inferencing package.

|ONNX Runtime|CANN|
|---|---|
|v1.12.1|6.0.0|
|v1.13.1|6.0.0|
|v1.14.0|6.0.0|
|v1.15.0|6.0.0|

## Build

For build instructions, please see the [BUILD page](../../build/eps.md#cann).

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

### enable_cann_graph

Whether to use the graph inference engine to speed up performance. The recommended setting is true. If false, it will fall back to the single-operator inference engine.

Default value: true

### dump_graphs

Whether to dump the subgraph into onnx format for analysis of subgraph segmentation.

Default value: false

### dump_om_model

Whether to dump the offline model for Ascend AI Processor to an .om file.

Default value: true

### precision_mode

The precision mode of the operator.

Value                   | Description
-|-
force_fp32/cube_fp16in_fp32out | convert to float32 first according to operator implementation
force_fp16 | convert to float16 when float16 and float32 are both supported
allow_fp32_to_fp16 | convert to float16 when float32 is not supported
must_keep_origin_dtype | keep it as it is
allow_mix_precision/allow_mix_precision_fp16 | mix precision mode

Default value: force_fp16

### op_select_impl_mode

Some built-in operators in CANN have high-precision and high-performance implementation.

Value                   | Description
-|-
high_precision | aim for high precision
high_performance | aim for high preformance

Default value: high_performance

### optypelist_for_implmode

Enumerate the list of operators which use the mode specified by the op_select_impl_mode parameter.

The supported operators are as follows:

* Pooling
* SoftmaxV2
* LRN
* ROIAlign

Default value: None

## Performance tuning

### IO Binding

The [I/O Binding feature](../../performance/tune-performance/iobinding.html) should be utilized to avoid overhead resulting from copies on inputs and outputs.

* Python

```python
import numpy as np
import onnxruntime as ort

providers = [
    (
        "CANNExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "npu_mem_limit": 2 * 1024 * 1024 * 1024,
            "enable_cann_graph": True,
        },
    ),
    "CPUExecutionProvider",
]

model_path = '<path to model>'

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

session = ort.InferenceSession(model_path, sess_options=options, providers=providers)

x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.int64)
x_ortvalue = ort.OrtValue.ortvalue_from_numpy(x, "cann", 0)

io_binding = sess.io_binding()
io_binding.bind_ortvalue_input(name="input", ortvalue=x_ortvalue)
io_binding.bind_output("output", "cann")

sess.run_with_iobinding(io_binding)

return io_binding.get_outputs()[0].numpy()
```

* C/C++(future)

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
            "op_select_impl_mode": "high_performance",
            "optypelist_for_implmode": "Gelu",
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

std::vector<const char *> keys{"device_id", "npu_mem_limit", "arena_extend_strategy", "enable_cann_graph"};
std::vector<const char *> values{"0", "2147483648", "kSameAsRequested", "1"};

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
