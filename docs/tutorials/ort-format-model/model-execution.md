---
title: Model Execution
parent: Deploy on mobile device and web
grand_parent: Tutorials
has_children: false
nav_order: 5
---
{::options toc_levels="2..4" /}

# Executing an ORT format model

The API for executing ORT format models is the same as for ONNX models.

See the [ONNX Runtime API documentation](../../api) for details on individual API usage.

## APIs by platform


| Platform | Available APIs |
|----------|----------------|
| Android | C, C++, Java, Kotlin |
| iOS | C, C++, Objective-C (Swift via bridge) |
| Web | JavaScript |

## ORT format model loading

If you provide a filename for the ORT format model, a file extension of '.ort' will be inferred to be an ORT format model.

If you provide in-memory bytes for the ORT format model, a marker in those bytes will be checked to infer if it's an ORT format model.

If you wish to explicitly say that the InferenceSession input is an ORT format model you can do so via SessionOptions, although this generally should not be necessary.

### Load ORT format model from a file path

C++ API
```c++
Ort::SessionOptions session_options;
session_options.AddConfigEntry("session.load_model_format", "ORT");

Ort::Env env;
Ort::Session session(env, <path to model>, session_options);
```

Java API
```java
SessionOptions session_options = new SessionOptions();
session_options.addConfigEntry("session.load_model_format", "ORT");

OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession(<path to model>, session_options);
```

JavaScript API
```js
import * as ort from "onnxruntime-web";

const session = await ort.InferenceSession.create("<path to model>");
```

### Load ORT format model from an in-memory byte array

If a session is created using an input byte array containing the ORT format model data, by default we will copy the model bytes at the time of session creation to ensure the model bytes buffer is valid.

You may also enable the option to use the model bytes directly by setting the Session Options config `session.use_ort_model_bytes_directly` to `1`, this may reduce the peak memory usage of ONNX Runtime Mobile, you will need to guarantee that the model bytes are valid throughout the lifespan of the ORT session using the model bytes. For ONNX Runtime Web, this option is set by default.

C++ API
```c++
Ort::SessionOptions session_options;
session_options.AddConfigEntry("session.load_model_format", "ORT");
session_options.AddConfigEntry("session.use_ort_model_bytes_directly", "1");

std::ifstream stream(<path to model>, std::ios::in | std::ios::binary);
std::vector<uint8_t> model_bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

Ort::Env env;
Ort::Session session(env, model_bytes.data(), model_bytes.size(), session_options);
```

Java API
```java
SessionOptions session_options = new SessionOptions();
session_options.addConfigEntry("session.load_model_format", "ORT");
session_options.addConfigEntry("session.use_ort_model_bytes_directly", "1");

byte[] model_bytes = Files.readAllBytes(Paths.get(<path to model>));

OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession(model_bytes, session_options);
```

JavaScript API
```js
import * as ort from "onnxruntime-web";

const response = await fetch(modelUrl);
const arrayBuffer = await response.arrayBuffer();
model_bytes = new Uint8Array(arrayBuffer);

const session = await ort.InferenceSession.create(model_bytes);
```

------

Next: [Using platform specific execution providers](./using-platform-specific-ep.md)
