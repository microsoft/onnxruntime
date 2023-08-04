---
title: Extensions
has_children: true
nav_order: 7
---

# ONNXRuntime-Extensions

[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status%2Fmicrosoft.onnxruntime-extensions?branchName=main)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=209&branchName=main)

## What's ONNXRuntime-Extensions

Introduction: ONNXRuntime-Extensions is a library that extends the capability of the ONNX models and inference with ONNX Runtime, via ONNX Runtime Custom Operator ABIs. It includes a set of [ONNX Runtime Custom Operator](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html) to support the common pre- and post-processing operators for vision, text, and nlp models. And it supports multiple languages and platforms, like Python on Windows/Linux/macOS, some mobile platforms like Android and iOS, and Web-Assembly etc. The basic workflow is to enhance a ONNX model firstly and then do the model inference with ONNX Runtime and ONNXRuntime-Extensions package.


<img src="../../images/combine-ai-extensions-img.png" alt="Pre and post-processing custom operators for vision, text, and NLP models" width="100%"/>
<sub>This image was created using <a href="https://github.com/sayanshaw24/combine" target="_blank">Combine.AI</a>, which is powered by Bing Chat, Bing Image Creator, and EdgeGPT.</sub>

## Quickstart

### **Python installation**
```bash
pip install onnxruntime-extensions
````


### **Nightly Build**

#### <strong>on Windows</strong>
```cmd
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-extensions
```
Please ensure that you have met the prerequisites of onnxruntime-extensions (e.g., onnx and onnxruntime) in your Python environment.
#### <strong>on Linux/macOS</strong>
Please make sure the compiler toolkit like gcc(later than g++ 8.0) or clang are installed before the following command
```bash
python -m pip install git+https://github.com/microsoft/onnxruntime-extensions.git
```


## Usage

## 1. Generate the pre-/post- processing ONNX model
With onnxruntime-extensions Python package, you can easily get the ONNX processing graph by converting them from Huggingface transformer data processing classes, check the following API for details.
```python
help(onnxruntime_extensions.gen_processing_models)
```
### NOTE: These data processing model can be merged into other model [onnx.compose](https://onnx.ai/onnx/api/compose.html) if needed.
## 2. Using Extensions for ONNX Runtime inference

### Python
There are individual packages for the following languages, please install it for the build.
```python
import onnxruntime as _ort
from onnxruntime_extensions import get_library_path as _lib_path

so = _ort.SessionOptions()
so.register_custom_ops_library(_lib_path())

# Run the ONNXRuntime Session, as ONNXRuntime docs suggested.
# sess = _ort.InferenceSession(model, so)
# sess.run (...)
```
### C++

```c++
  // The line loads the customop library into ONNXRuntime engine to load the ONNX model with the custom op
  Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary((OrtSessionOptions*)session_options, custom_op_library_filename, &handle));

  // The regular ONNXRuntime invoking to run the model.
  Ort::Session session(env, model_uri, session_options);
  RunSession(session, inputs, outputs);
```
### Java
```java
var env = OrtEnvironment.getEnvironment();
var sess_opt = new OrtSession.SessionOptions();

/* Register the custom ops from onnxruntime-extensions */
sess_opt.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
```

### C#
```C#
SessionOptions options = new SessionOptions()
options.RegisterOrtExtensions()
session = new InferenceSession(model, options)
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

[MIT License](https://github.com/microsoft/onnxruntime-extensions/blob/main/LICENSE)