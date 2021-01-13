---
title: ARM Compute Library (ACL)
parent: Execution Providers
grand_parent: Reference
nav_order: 8
---

# ACL Execution Provider

[Arm Compute Library](https://github.com/ARM-software/ComputeLibrary) is an open source inference engine maintained by Arm and Linaro companies. The integration of ACL as an execution provider (EP) into ONNX Runtime accelerates performance of ONNX model workloads across Armv8 cores.

## Build ACL execution provider
For build instructions, please see the [BUILD page](../../how-to/build.md#ARM-Compute-Library).

## Using the ACL execution provider
### C/C++
```
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
bool enable_cpu_mem_arena = true;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ACL(sf, enable_cpu_mem_arena));
```
The C API details are [here](../api/c-api.md).

## Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../../how-to/tune-performance.md)

When/if using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/perftest), use the flag -e acl
