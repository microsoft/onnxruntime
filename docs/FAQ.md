# FAQ
Here are some commonly raised questions from users of ONNX Runtime and brought up in [Issues](https://github.com/microsoft/onnxruntime/issues).

## Do the GPU builds support quantized models?
The default CUDA build supports 3 standard quantization operators: QuantizeLinear, DequantizeLinear, and MatMulInteger. The TensorRT EP has limited support for INT8 quantized ops. In general, support of quantized models through ORT is continuing to expand on a model-driven basis. For performance improvements, quantization is not always required, and we suggest trying alternative strategies to [performance tune](./ONNX_Runtime_Perf_Tuning.md) before determining that quantization is necessary.

## How do I change the severity level of the default logger to something other than the default (WARNING)?
Setting the severity level to VERBOSE is most useful when debugging errors.

Refer to the API documentation:
* Python - [RunOptions.log_severity_level](https://microsoft.github.io/onnxruntime/python/api_summary.html#onnxruntime.RunOptions.log_severity_level)
```
import onnxruntime as ort
ort.set_default_logger_severity(0)
```
* C - [SetSessionLogSeverityLevel](./../include/onnxruntime/core/session/onnxruntime_c_api.h)

## How do I load and run models that have multiple inputs and outputs using the C/C++ API?
See an example from the 'override initializer' test in [test_inference.cc](./../onnxruntime/test/shared_lib/test_inference.cc) that has 3 inputs and 3 outputs.
```
std::vector<Ort::Value> ort_inputs;
ort_inputs.push_back(std::move(label_input_tensor));
ort_inputs.push_back(std::move(f2_input_tensor));
ort_inputs.push_back(std::move(f11_input_tensor));
std::vector<const char*> input_names = {"Label", "F2", "F1"};
const char* const output_names[] = {"Label0", "F20", "F11"};
std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
ort_inputs.data(), ort_inputs.size(), output_names, countof(output_names));
```

## How do I force single threaded execution mode in ORT? By default, session.run() uses all the computer's cores. 

To limit use to a single thread only:
* If built with OpenMP, set the environment variable OMP_NUM_THREADS to 1. The default inter_op_num_threads in session options is already 1.  
* If not built with OpenMP, set the session options intra_op_num_threads to 1. Do not change the default inter_op_num_threads (1).

It's recommended to build onnxruntime without openmp if you only need single threaded execution. 

This is supported in ONNX Runtime v1.3.0+

**Python example:**
```
#!/usr/bin/python3
os.environ["OMP_NUM_THREADS"] = "1"
import onnxruntime

opts = onnxruntime.SessionOptions()
opts.inter_op_num_threads = 1
opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
ort_session = onnxruntime.InferenceSession('/path/to/model.onnx', sess_options=opts)
```

**C++ example:**
```
// initialize  environment...one environment per process
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

// initialize session options if needed
Ort::SessionOptions session_options;
session_options.SetInterOpNumThreads(1);
#ifdef _WIN32
  const wchar_t* model_path = L"squeezenet.onnx";
#else
  const char* model_path = "squeezenet.onnx";
#endif

Ort::Session session(env, model_path, session_options);
```
