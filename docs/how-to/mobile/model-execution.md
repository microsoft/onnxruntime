---
title: Model Execution
parent: Deploy ONNX Runtime Mobile
grand_parent: How to
has_children: false
nav_order: 5
---
{::options toc_levels="2..4" /}

# Executing an ORT format model

The API for executing ORT format models is the same as for ONNX models.

See the [ONNX Runtime API documentation](../../reference/api) for details on individual API usage.

## APIs by platform


| Platform | Available APIs |
|----------|----------------|
| Android | C, C++, Java |
| iOS | C, C++, Objective-C (Swift via bridge) |

## ORT format model loading

If you provide a filename for the ORT format model, a file extension of '.ort' will be inferred to be an ORT format model.

If you provide in-memory bytes for the ORT format model, a marker in those bytes will be checked to infer if it's an ORT format model.

If you wish to explicitly say that the InferenceSession input is an ORT format model you can do so via SessionOptions, although this generally should not be necessary.

C++ API
```c++
Ort::SessionOptions session_options;
session_options.AddConfigEntry('session.load_model_format', 'ORT');

Ort::Env env;
```
```
Ort::Session session(env, <path to model>, session_options);
```
or
```
Ort::Session session(env, <byte array representing an ONNX model>, <byte size of the byte array>, session_options);
```

Java API
```java
try(OrtEnvironment env = OrtEnvironment.getEnvironment();
    SessionOptions session_options = new SessionOptions()) {
    session_options.addConfigEntry("session.load_model_format", "ORT");
    OrtSession session = env.createSession(<path to model> or <byte array representing an ONNX model>, session_options);
  ...
}
```

------

Next: [Using NNAPI and CoreML with ORT Mobile](using-nnapi-coreml-with-ort-mobile)
