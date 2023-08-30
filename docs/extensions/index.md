---
title: Extensions
has_children: true
nav_order: 7
---

# ONNXRuntime-Extensions

[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status%2Fmicrosoft.onnxruntime-extensions?branchName=main)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=209&branchName=main)

## What is ONNXRuntime-Extensions?

ONNXRuntime-Extensions is a library that extends the capability of the ONNX models and inference with ONNX Runtime, via ONNX Runtime Custom Operator ABIs. It includes a set of Custom Operators to support common model pre- and post-processing for audio, vision, text, and language models. As with ONNX Runtime, Extensions also supports multiple languages and platforms (Python on Windows/Linux/macOS, Android and iOS mobile platforms and Web-Assembly for web.

The basic workflow is to add the custom operators to an ONNX model and then to perform inference on the enhanced model with ONNX Runtime and ONNXRuntime-Extensions packages.


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

## Prepare the pre/post-processing ONNX model
With onnxruntime-extensions Python package, you can easily get the ONNX processing graph by converting them from Huggingface transformer data processing classes such as in the following example:
```python
import onnxruntime as _ort
from transformers import AutoTokenizer
from onnxruntime_extensions import OrtPyFunction, gen_processing_models

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = OrtPyFunction(gen_processing_models(tokenizer, pre_kwargs={})[0])
```

For more information, you can check API using the following:
```python
help(onnxruntime_extensions.gen_processing_models)
```
### NOTE: These data processing model can be merged into other model [onnx.compose](https://onnx.ai/onnx/api/compose.html) if needed.
## Inference with ONNX Runtime Extensions

### Python
There are individual packages for the following languages, please install it for the build.
```python
import onnxruntime as _ort
from onnxruntime_extensions import get_library_path as _lib_path

so = _ort.SessionOptions()
so.register_custom_ops_library(_lib_path())

# Run the ONNXRuntime Session as per ONNXRuntime docs suggestions.
sess = _ort.InferenceSession(model, so)
sess.run (...)
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

## Tutorials

Check out some end to end tutorials with our custom operators:
- NLP: [An end-to-end BERT tutorial](https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/bert_e2e.py)
- Audio: [Using audio encoding and decoding for Whisper](https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/whisper_e2e.py)
- Vision: [The YOLO model with our DrawBoundingBoxes operator](https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/yolo_e2e.py)

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
