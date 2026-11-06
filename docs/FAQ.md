# FAQ
Here are some commonly raised questions from users of ONNX Runtime and brought up in [Issues](https://github.com/microsoft/onnxruntime/issues).

## Do the GPU builds support quantized models?
The default CUDA build supports 3 standard quantization operators: QuantizeLinear, DequantizeLinear, and MatMulInteger. The TensorRT EP has limited support for INT8 quantized ops. In general, support of quantized models through ORT is continuing to expand on a model-driven basis. For performance improvements, quantization is not always required, and we suggest trying alternative strategies to [performance tune](https://onnxruntime.ai/docs/performance/tune-performance/) before determining that quantization is necessary.

## Can I use a prebuilt TensorRT execution provider with TensorRT 11?

TensorRT source compatibility and prebuilt binary compatibility are different. On Linux, a
`libonnxruntime_providers_tensorrt.so` that declares `libnvinfer.so.10` and `libnvonnxparser.so.10` as dependencies
requires TensorRT 10 libraries. Installing the TensorRT 11 Python package, changing `LD_LIBRARY_PATH` to a
TensorRT 11 directory, or upgrading ONNX Runtime to another wheel with the same dependencies cannot satisfy them.
Do not symlink TensorRT 11 libraries to TensorRT 10 names: this does not make the major-version ABIs compatible.

| Provider build | TensorRT libraries required at runtime | Works with only TensorRT 11 installed? |
| --- | --- | --- |
| Prebuilt `onnxruntime-gpu` wheel linked to TensorRT 10 | The TensorRT 10 version supported by that wheel | No |
| Source build against a TensorRT 10 SDK | Compatible TensorRT 10 libraries | No |
| Source build against a TensorRT 11 SDK, using source with TensorRT 11 adaptations | Compatible TensorRT 11 libraries | Yes, subject to the SDK's CUDA, driver, GPU, and platform requirements |

The current [TensorRT CI configuration](../.github/workflows/linux_tensorrt_ci.yml) still selects TensorRT 10.
This repository does not configure a separate TensorRT 11 wheel. TensorRT 11 source conditionals are not an
announcement of a TensorRT 11-targeted prebuilt package, package version, or release date. Check the
[TensorRT EP requirements](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements)
and [release notes](https://github.com/microsoft/onnxruntime/releases) for published binary compatibility;
the `onnxruntime-gpu` package name alone does not select a TensorRT major version.

### Build a TensorRT 11 wheel on Linux

Follow the [TensorRT build prerequisites](https://onnxruntime.ai/docs/build/eps.html#tensorrt), including the
Python build dependencies in an activated virtual environment. Use a source revision containing the TensorRT 11
adaptations; do not assume an older release tag contains them. Install the TensorRT 11 **development SDK** (headers,
libraries, and ONNX parser), not just Python bindings. Select CUDA, cuDNN, and a driver compatible with both that
SDK and the [CUDA EP requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

Set the following absolute paths to your installations. Use a new build directory so CMake does not reuse
TensorRT 10 headers or library paths from an earlier build:

```bash
ORT_SOURCE_DIR=/absolute/path/to/onnxruntime
ORT_BUILD_DIR=/absolute/path/to/ort-trt11-build
TENSORRT_HOME=/absolute/path/to/TensorRT-11
CUDA_HOME=/absolute/path/to/cuda
CUDNN_HOME=/absolute/path/to/cudnn

python "$ORT_SOURCE_DIR/tools/ci_build/build.py" \
  --build_dir "$ORT_BUILD_DIR" --config Release --update --build --parallel \
  --build_wheel --use_cuda --use_tensorrt --use_tensorrt_builtin_parser \
  --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" --tensorrt_home "$TENSORRT_HOME"
```

`--use_tensorrt_builtin_parser` selects the ONNX parser shipped with the TensorRT SDK, rather than building
the separately pinned OSS parser. Confirm that CMake reports `NV_TENSORRT_MAJOR is 11` and finds the libraries
under your TensorRT 11 installation.

Install the resulting wheel from `$ORT_BUILD_DIR/Release/dist/` into a clean virtual environment. Do not install
CPU and GPU ONNX Runtime distributions together, or mix a provider library from this build with another ORT wheel.
Set `ORT_WHEEL` to the actual wheel filename. Expose the SDK libraries before starting Python; adjust `lib`/`lib64`
to match your SDK layouts:

```bash
ORT_WHEEL=/absolute/path/to/the/produced.whl
python -m pip install "$ORT_WHEEL"
export LD_LIBRARY_PATH="$TENSORRT_HOME/lib:$CUDA_HOME/lib64:$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Import the installed wheel, not the source checkout.
cd /tmp
PROVIDER_LIB="$(python -c 'from pathlib import Path; import onnxruntime; print(Path(onnxruntime.__file__).parent / "capi" / "libonnxruntime_providers_tensorrt.so")')"
readelf -d "$PROVIDER_LIB" | grep NEEDED
ldd "$PROVIDER_LIB"
```

The TensorRT dependencies should be `libnvinfer.so.11` and `libnvonnxparser.so.11`, not `.so.10`, and `ldd` should
show no missing libraries. Run your model with `TensorrtExecutionProvider` explicitly selected, verify it remains
in the session's `get_providers()`, and check profiling output for actual TensorRT execution. A successful import,
`get_available_providers()`, or CUDA/CPU fallback alone does not validate TensorRT initialization or execution.

For custom TensorRT plugins, rebuild them against TensorRT 11 and load them using `trt_extra_plugin_lib_paths`;
CUDA/CPU fallback cannot execute TensorRT-only plugins. Regenerate TensorRT engine caches after changing
TensorRT versions. TensorRT 11 uses strongly typed networks: the `trt_fp16_enable`, `trt_bf16_enable`, and
`trt_int8_enable` options no longer control precision; represent the intended types and quantization in the model.

## How can I reduce model loading time for large CPU or CUDA models?

See [Accelerate model loading](Model_Loading_Performance.md) for parallel CPU weight prepacking and CUDA external-data
loading through pinned host buffers, including the session and execution provider options that control them.

## How do I change the severity level of the default logger to something other than the default (WARNING)?
Setting the severity level to VERBOSE is most useful when debugging errors.

Refer to the API documentation:
* Python - [RunOptions.log_severity_level](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.RunOptions.log_severity_level)
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
