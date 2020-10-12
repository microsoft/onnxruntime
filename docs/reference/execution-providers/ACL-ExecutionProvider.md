---
title: ARM Compute Library (ACL)
parent: Execution Providers
grand_parent: Reference
nav_order: 1
---

# ACL Execution Provider

[Arm Compute Library](https://github.com/ARM-software/ComputeLibrary) is an open source inference engine maintained by Arm and Linaro companies. The integration of ACL as an execution provider (EP) into ONNX Runtime accelerates performance of ONNX model workloads across Armv8 cores.

## Build ACL execution provider

For build instructions, please see the [BUILD page](../../how-to/build.md#ARM-Compute-Library).

## Using the ACL execution provider

### C/C++

To use ACL as execution provider for inference, please register it as below.

```c++
string log_id = "Foo";
auto logging_manager = std::make_unique<LoggingManager>
(std::unique_ptr<ISink>{new CLogSink{}},
                                  static_cast<Severity>(lm_info.default_warning_level),
                                  false,
                                  LoggingManager::InstanceType::Default,
                                  &log_id)
Environment::Create(std::move(logging_manager), env)
InferenceSession session_object{so, env};
session_object.RegisterExecutionProvider(std::make_unique<::onnxruntime::ACLExecutionProvider>());
status = session_object.Load(model_file_name);
```

The C API details are [here](../api/c-api.md).

## Performance Tuning

For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../../how-to/tune-performance.md)

When/if using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/perftest), use the flag -e acl
