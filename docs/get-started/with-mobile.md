---
title: Mobile
parent: Get Started
toc: true
nav_order: 7
---
# Get Started with ORT for Mobile
{: .no_toc }

The API for executing ORT format models is the same as for ONNX models. 

See the [ONNX Runtime API documentation](../api) for details on individual API usage.

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
Ort::Session session(env, <path to model>, session_options);
```

Java API
```java
SessionOptions session_options = new SessionOptions();
session_options.addConfigEntry("session.load_model_format", "ORT");

OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession(<path to model>, opsession_optionstions);
```

## Learn More
- [Deploy on mobile device and web](../tutorials/ort-format-model/)