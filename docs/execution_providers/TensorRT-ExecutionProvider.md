## TensortRT Execution Provider

The TensorRT execution provider in the ONNX Runtime makes use of NVIDIA's [TensortRT](https://developer.nvidia.com/tensorrt) Deep Learning inferencing engine to accelerate ONNX model in their family of GPUs. Microsoft and NVIDIA worked closely to integrate the TensorRT execution provider with ONNX Runtime.

With the TensorRT execution provider, the ONNX Runtime delivers better inferencing performance on the same hardware compared to generic GPU acceleration. 

### Build TensorRT execution provider
Developers can now tap into the power of TensorRT through ONNX Runtime to accelerate inferencing of ONNX models. Instructions to build the TensorRT execution provider from source are available [here](https://github.com/Microsoft/onnxruntime/blob/master/BUILD.md#build). [Dockerfiles](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles#tensorrt-version-preview) are available for convenience.

### Using the TensorRT execution provider
#### C/C++
The TensortRT execution provider needs to be registered with ONNX Runtime to enable in the inference session. 
```
InferenceSession session_object{so};
session_object.RegisterExecutionProvider(std::make_unique<::onnxruntime::TensorrtExecutionProvider>());
status = session_object.Load(model_file_name);
```
The C API details are [here](https://github.com/Microsoft/onnxruntime/blob/master/docs/C_API.md#c-api).

### Python
When using the Python wheel from the ONNX Runtime build with TensorRT execution provider, it will be automatically prioritized over the default GPU or CPU execution providers. There is no need to separately register the execution provider. Python APIs details are [here](https://microsoft.github.io/onnxruntime/api_summary.html).

### Performance Tuning
To test the performance of your ONNX Model with the TensorRT execution provider, use the flag `-e tensorrt` in [onnxruntime_perf_test](https://github.com/Microsoft/onnxruntime/tree/master/onnxruntime/test/perftest#onnxruntime-performance-test).

### Sample
Please see [this Notebook](https://github.com/microsoft/onnxruntime/blob/master/docs/python/notebooks/onnx-inference-byoc-gpu-cpu-aks.ipynb) for an example of running a model on GPU using ONNX Runtime through Azure Machine Learning Services.

### Using onnxruntime_perf_test
You can test the performance for your ONNX Model with the TensorRT execution provider. Use the flag `-e tensorrt` in [onnxruntime_perf_test](https://github.com/Microsoft/onnxruntime/tree/master/onnxruntime/test/perftest#onnxruntime-performance-test).

### Configuring Engine Max Batch Size and Workspace Size
By default TensorRT execution provider builds an ICudaEngine with max batch size = 1 and max workspace size = 1 GB
One can override these defaults by setting environment variables ORT_TENSORRT_MAX_BATCH_SIZE and ORT_TENSORRT_MAX_WORKSPACE_SIZE.
e.g. on Linux
#### override default batch size to 10
export ORT_TENSORRT_MAX_BATCH_SIZE=10
#### override default max workspace size to 2GB
export ORT_TENSORRT_MAX_WORKSPACE_SIZE=2147483648

