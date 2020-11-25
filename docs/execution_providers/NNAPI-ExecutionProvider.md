# NNAPI Execution Provider

[Android Neural Networks API (NNAPI)](https://developer.android.com/ndk/guides/neuralnetworks) is a unified interface to CPU, GPU, and NN accelerators on Android.

## Minimum requirements

The NNAPI EP requires Android devices with Android 8.1 or higher, it is recommended to use Android devices with Android 9 or higher to achieve optimal performance.

## Build NNAPI EP

For build instructions, please see the [BUILD page](../../BUILD.md#Android-NNAPI-Execution-Provider).

## Using NNAPI EP in C/C++

```
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sf));
Ort::Session session(env, model_path, sf);
```
The C API details are [here](../C_API.md#c-api).
