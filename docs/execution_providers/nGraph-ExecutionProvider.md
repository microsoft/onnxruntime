<p align="center">
  <img src="docs/images/ngraph-logo.png">
</p>

## nGraph Execution Provider (preview)

[nGraph](https://github.com/NervanaSystems/ngraph) is a deep learning compiler from Intel®. The integration of nGraph as an execution provider (EP) into ONNX Runtime accelerates performance of ONNX model workloads across wide range of hardware offerings. Microsoft and Intel worked closely to integrate the nGraph EP with ONNX Runtime to showcase the benefits of quantization (int8). The nGraph EP leverages Intel® DL Boost and delivers performance increase with minimal loss of accuracy relative to FP32 ONNX models. With the nGraph EP, the ONNX Runtime delivers better inference performance across range of Intel hardware including Intel® Xeon® Processors compared to a generic CPU execution provider.

### Build nGraph execution provider
Developers can now tap into the power of nGraph through ONNX Runtime to accelerate inference performance of ONNX models. Instructions for building the nGraph execution provider from the source is available [here](https://github.com/Microsoft/onnxruntime/blob/master/BUILD.md#nGraph).

While the nGraph Compiler stack supports various operating systems and backends ([full list available here](https://www.ngraph.ai/ecosystem)), the nGraph execution provider for ONNX Runtime is validated for the following:  
### Supported OS
*	Ubuntu 16.04
* Windows 10 (`DEX_ONLY` mode is only one supported for the moment, codegen mode is work-in-progress.)
* More to be added soon!

### Supported backend
*	CPU
* More to be added soon!

### Using the nGraph execution provider
#### C/C++
To use nGraph as execution provider for inferencing, please register it as below.
```
InferenceSession session_object{so};
session_object.RegisterExecutionProvider(std::make_unique<::onnxruntime::NGRAPHExecutionProvider>());
status = session_object.Load(model_file_name);
```
The C API details are [here](https://github.com/Microsoft/onnxruntime/blob/master/docs/C_API.md#c-api).

### Python
When using the python wheel from the ONNX Runtime built with nGraph execution provider, it will be automatically prioritized over the CPU execution provider. Python APIs details are [here](https://github.com/Microsoft/onnxruntime/blob/master/docs/python/api_summary.rst#api-summary).

### Using onnxruntime_perf_test
You can test the performance of your ONNX Model with the nGraph execution provider. Use the flag -e ngraph in [onnxruntime_perf_test](https://github.com/Microsoft/onnxruntime/tree/master/onnxruntime/test/perftest#onnxruntime-performance-test).
