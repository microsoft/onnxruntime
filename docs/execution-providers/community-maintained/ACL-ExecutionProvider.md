---
title: Arm - ACL
description: Instructions to execute ONNX Runtime with the ACL Execution Provider
grand_parent: Execution Providers
parent: Community-maintained
nav_order: 2
redirect_from: /docs/reference/execution-providers/ACL-ExecutionProvider
---

# ACL Execution Provider
{: .no_toc }

The ACL Execution Provider enables accelerated performance on Arm®-based CPUs through [Arm Compute Library](https://github.com/ARM-software/ComputeLibrary){:target="_blank"}.


## Build
For build instructions, please see the [build page](../../build/eps.md#arm-compute-library).

## Usage
### C/C++
{: .no_toc }

```
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
bool enable_fast_math = true;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ACL(sf, enable_fast_math));
```
The C API details are [here](../../get-started/with-c.html).

### Python
{: .no_toc }

```
import onnxruntime

providers = [("ACLExecutionProvider", {"enable_fast_math": "true"})]
sess = onnxruntime.InferenceSession("model.onnx", providers=providers)
```

## Performance Tuning
Arm Compute Library has a fast math mode that can increase performance with some potential decrease in accuracy for MatMul and Conv operators. It is disabled by default.

When using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/perftest){:target="_blank"}, use the flag `-e acl` to enable the ACL Execution Provider.  You can additionally use `-i 'enable_fast_math|true'` to enable fast math.

Arm Compute Library uses the ONNX Runtime intra-operator thread pool when running via the execution provider. You can control the size of this thread pool using the `-x` option.

## Supported Operators

|Operator|Supported types|
|---|---|
|AveragePool|float|
|BatchNormalization|float|
|Concat|float|
|Conv|float, float16|
|FusedConv|float|
|FusedMatMul|float, float16|
|Gemm|float|
|GlobalAveragePool|float|
|GlobalMaxPool|float|
|MatMul|float, float16|
|MatMulIntegerToFloat|uint8, int8, uint8+int8|
|MaxPool|float|
|NhwcConv|float|
|Relu|float|
|QLinearConv|uint8, int8, uint8+int8|
