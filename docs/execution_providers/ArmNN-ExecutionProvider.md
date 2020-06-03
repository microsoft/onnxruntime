## ArmNN Execution Provider

[ArmNN](https://github.com/ARM-software/armnn) is an open source inference engine maintained by Arm and Linaro companies. The integration of ArmNN as an execution provider (EP) into ONNX Runtime accelerates performance of ONNX model workloads across Armv8 cores.

### Build ArmNN execution provider
For build instructions, please see the [BUILD page](../../BUILD.md#ArmNN).

### Using the ArmNN execution provider
#### C/C++
To use ArmNN as execution provider for inferencing, please register it as below.
```
string log_id = "Foo";
auto logging_manager = std::make_unique<LoggingManager>
(std::unique_ptr<ISink>{new CLogSink{}},
                                  static_cast<Severity>(lm_info.default_warning_level),
                                  false,
                                  LoggingManager::InstanceType::Default,
                                  &log_id)
Environment::Create(std::move(logging_manager), env)
InferenceSession session_object{so, env};
session_object.RegisterExecutionProvider(std::make_unique<::onnxruntime::ArmNNExecutionProvider>());
status = session_object.Load(model_file_name);
```
The C API details are [here](../C_API.md#c-api).

### Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../ONNX_Runtime_Perf_Tuning.md)

When/if using [onnxruntime_perf_test](../../onnxruntime/test/perftest), use the flag -e armnn
