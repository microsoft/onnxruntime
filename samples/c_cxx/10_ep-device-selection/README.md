# EP Device Selection

ONNX Runtime provides since version 1.23.0 an execution provider independent way of querying and selecting
inference devices. This involves typically 3 steps. 

- 1. Registration of execution provider libraries
```cpp
  auto env = Ort::Env(ORT_LOGGING_LEVEL_WARNING);
  env.RegisterExecutionProviderLibrary("openvino", ORT_TSTR("onnxruntime_providers_openvino.dll"));
  env.RegisterExecutionProviderLibrary("qnn", ORT_TSTR("onnxruntime_providers_qnn.dll"));
  env.RegisterExecutionProviderLibrary("nv_tensorrt_rtx", ORT_TSTR("onnxruntime_providers_nv_tensorrt_rtx.dll"));
```

- 2. Querying and selecting Execution Provider (EP) Devices

```cpp
  auto ep_devices = env.GetEpDevices();
  auto selected_devices = my_ep_selection_function(ep_devices);

  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_V2(env, selected_devices, ep_options);
  // Optionally, set device policy. E.g. OrtExecutionProviderDevicePolicy_PREFER_GPU, OrtExecutionProviderDevicePolicy_PREFER_NPU, OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE
  session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);
```
- 3. Use the session options to create a inference session

```cpp
  Ort::Session session(env, ORT_TSTR("path/to/model.onnx"), session_options);
```


## Building the sample

`cmake -B build -S . -DONNX_RUNTIME_PATH=path/to/onnxruntime> -DTRTRTX_RUNTIME_PATH=<path/to/TRTRTX/libs> && cmake --build build --config Release`

Then run
```
./build/Release/ep-device-selection -i ./Input.png -o ./output.png
```

Run 

```
./build/Release/ep-device-selection -h
```
to know about more available command line options that influence device selection.

## Model

The ONNX file in this folder was generated using code from https://github.com/DepthAnything/Depth-Anything-V2 (Apache 2.0)
with weights from https://huggingface.co/depth-anything/Depth-Anything-V2-Small/ (Apache 2.0).

## Dependencies

This sample vendors a copy of https://github.com/lvandeve/lodepng (Zlib license)
