# ONNX Runtime Java API for Android

This directory contains JNI (Java Native Interface) code and Java API of ONNX Runtime

## Getting Started

The Java API is like C++ API and C# API. 

1. Most Java API can throw an OrtException

2. For efficiency, tensor is a [direct byte buffer](https://docs.oracle.com/javase/7/docs/api/java/nio/ByteBuffer.html) in Java

3. The native object will be released if the corresponding Java object is released in GC. In addition, every Java object has a dispose() method that releases the native object manually.

4. `SessionOptions` has a method `appendNnapiExecutionProvider` for NNAPI EP on Android.

## Build AAR Package

Firstly, build libonnxruntime-jni.so and copy it into the directory of the corresponding architecture (`onnxruntime/src/main/jniLibs/arm64-v8a/` or `onnxruntime/src/main/jniLibs/armeabi-v7a/`), then build the AAR Package by running

```bash
./gradlew :onnxruntime:build
```

The Android AAR package will be generated at `onnxruntime/build/outputs/aar/`

## Example

An Android app demo is located in `sample/`.
