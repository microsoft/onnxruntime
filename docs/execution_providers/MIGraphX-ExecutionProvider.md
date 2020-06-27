# MIGraphX Execution Provider

ONNX Runtime's [MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX/) execution provider uses AMD's Deep Learning graph optimization engine to accelerate ONNX model on AMD GPUs. 

## Build
For build instructions, please see the [BUILD page](../../BUILD.md#AMD-MIGraphX). 

## Using the MIGraphX execution provider
### C/C++
The MIGraphX execution provider needs to be registered with ONNX Runtime to enable in the inference session. 
```
string log_id = "Foo";
auto logging_manager = std::make_unique<LoggingManager>
(std::unique_ptr<ISink>{new CLogSink{}},
                                  static_cast<Severity>(lm_info.default_warning_level),
                                  false,
                                  LoggingManager::InstanceType::Default,
                                  &log_id)
Environment::Create(std::move(logging_manager), env)
InferenceSession session_object{so,env};
session_object.RegisterExecutionProvider(std::make_unique<::onnxruntime::MIGraphXExecutionProvider>());
status = session_object.Load(model_file_name);
```
You can check [here](https://github.com/scxiao/ort_test/tree/master/char_rnn) for a specific c/c++ program.

The C API details are [here](../C_API.md#c-api).

### Python
When using the Python wheel from the ONNX Runtime build with MIGraphX execution provider, it will be automatically
prioritized over the default GPU or CPU execution providers. There is no need to separately register the execution
provider. Python APIs details are [here](../python/api_summary.rst#api-summary).

You can check [here](https://github.com/scxiao/ort_test/tree/master/python/run_onnx) for a python script to run an
model on either the CPU or MIGraphX Execution Provider.

## Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../ONNX_Runtime_Perf_Tuning.md)

When/if using [onnxruntime_perf_test](../../onnxruntime/test/perftest#onnxruntime-performance-test), use the flag `-e migraphx` 

## Configuring environment variables
MIGraphX providers an environment variable ORT_MIGRAPHX_FP16_ENABLE to enable the FP16 mode.

