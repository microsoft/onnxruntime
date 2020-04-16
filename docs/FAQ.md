# FAQ 

## Do the GPU builds support quantized models?
The default CUDA build does not support any quantized operators right now. The TensorRT EP has limited support for INT8 quantized ops. In general, support of quantized models through ORT is continuing to expand on a model-driven basis. For performance improvements, quantization is not always required, and we suggest trying alternative strategies to [performance tune](./ONNX_Runtime_Perf_Tuning.md) before determining that quantization is necessary.

## How do I turn on verbose mode logging in Python?
```
import onnxruntime as ort
ort.set_default_logger_severity(0)
```

## How do I load and run models that have multiple inputs and outputs using the C/C++ API?
See [this example](./../onnxruntime/test/shared_lib/test_inference.cc#L395)

## How do I force single threaded execution mode in ORT? 
This will be supported in ONNX Runtime v1.3.0+

By default, session.run() uses all the computer's cores. To limit use to a single thread only, please do both of the following:
* Build with openmp disabled or set environment variable OMP_NUM_THREADS to 1.
* Set the session options intra_op_num_threads and inter_op_num_threads to 1 each.

**Python example:**
```
#!/usr/bin/python3
os.environ["OMP_NUM_THREADS"] = "1"
import onnxruntime

opts = onnxruntime.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
ort_session = onnxruntime.InferenceSession('/path/to/model.onnx', sess_options=opts)
```

**C++ example:**
```
// initialize  enviroment...one enviroment per process
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

// initialize session options if needed
Ort::SessionOptions session_options;
session_options.SetIntraOpNumThreads(1);
session_options.SetInterOpNumThreads(1);
#ifdef _WIN32
  const wchar_t* model_path = L"squeezenet.onnx";
#else
  const char* model_path = "squeezenet.onnx";
#endif

Ort::Session session(env, model_path, session_options);
```



