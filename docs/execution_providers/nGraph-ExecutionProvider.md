<p align="center">
  <img src="../images/ngraph-logo.png">
</p>

# nGraph Execution Provider

[nGraph](https://github.com/NervanaSystems/ngraph) is a deep learning compiler from Intel®. The integration of nGraph as an execution provider (EP) into ONNX Runtime accelerates performance of ONNX model workloads across wide range of hardware offerings. Microsoft and Intel worked closely to integrate the nGraph EP with ONNX Runtime to showcase the benefits of quantization (int8). The nGraph EP leverages Intel® DL Boost and delivers performance increase with minimal loss of accuracy relative to FP32 ONNX models. With the nGraph EP, the ONNX Runtime delivers better inference performance across range of Intel hardware including Intel® Xeon® Processors compared to a generic CPU execution provider.

## Build
For build instructions, please see the [BUILD page](../../BUILD.md#nGraph).

## Supported OS
While the nGraph Compiler stack supports various operating systems and backends ([full list available here](https://www.ngraph.ai/ecosystem)), the nGraph execution provider for ONNX Runtime is validated for the following:  

*	Ubuntu 16.04
* Windows 10 (`DEX_ONLY` mode is only one supported for the moment, codegen mode is work-in-progress.)
* More to be added soon!

## Supported backend
*	CPU

## Using the nGraph execution provider
### C/C++
To use nGraph as execution provider for inferencing, please register it as below.
```
InferenceSession session_object{so};
session_object.RegisterExecutionProvider(std::make_unique<::onnxruntime::NGRAPHExecutionProvider>());
status = session_object.Load(model_file_name);
```
The C API details are [here](../C_API.md#c-api).

### Python
When using the python wheel from the ONNX Runtime built with nGraph execution provider, it will be automatically prioritized over the CPU execution provider. Python APIs details are [here](../python/api_summary.rst#api-summary).

## Performance Tuning
For performance tuning, please see guidance on this page: [ONNX Runtime Perf Tuning](../ONNX_Runtime_Perf_Tuning.md)

When/if using [onnxruntime_perf_test](../../onnxruntime/test/perftest), use the flag -e ngraph
