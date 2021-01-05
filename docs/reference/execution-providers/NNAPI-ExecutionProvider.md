---
title: NNAPI
parent: Execution Providers
grand_parent: Reference
nav_order: 6
---


# NNAPI Execution Provider
{: .no_toc }

[Android Neural Networks API (NNAPI)](https://developer.android.com/ndk/guides/neuralnetworks) is a unified interface to CPU, GPU, and NN accelerators on Android.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Minimum requirements

The NNAPI EP requires Android devices with Android 8.1 or higher, it is recommended to use Android devices with Android 9 or higher to achieve optimal performance.

## Build NNAPI EP

For build instructions, please see the [BUILD page](../../how-to/build.md#Android-NNAPI-Execution-Provider).

## Using NNAPI EP in C/C++

```c++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sf));
Ort::Session session(env, model_path, sf);
```

The C API details are [here](../api/c-api.md).
