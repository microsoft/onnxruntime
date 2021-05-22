# ONNX Runtime Web

ONNX Runtime Web is a Javascript library for running ONNX models on browsers and on Node.js.

ONNX Runtime Web has adopted WebAssembly and WebGL technologies for providing an optimized ONNX model inference runtime for both CPUs and GPUs.

### Why ONNX models

The [Open Neural Network Exchange](http://onnx.ai/) (ONNX) is an open standard for representing machine learning models. The biggest advantage of ONNX is that it allows interoperability across different open source AI frameworks, which itself offers more flexibility for AI frameworks adoption. See [Getting ONNX Models](#Getting-ONNX-models).

### Why ONNX Runtime Web

With ONNX Runtime Web, web developers can score pre-trained ONNX models directly on browsers with various benefits of reducing server-client communication and protecting user privacy, as well as offering install-free and cross-platform in-browser ML experience.

ONNX Runtime Web can run on both CPU and GPU. For running on CPU, [WebAssembly](https://developer.mozilla.org/en-US/docs/WebAssembly) is adopted to execute the model at near-native speed. Furthermore, ONNX Runtime Web utilizes [Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers) to provide a "multi-threaded" environment to parallelize data processing. Empirical evaluation shows very promising performance gains on CPU by taking full advantage of WebAssembly and Web Workers. For running on GPUs, a popular standard for accessing GPU capabilities - WebGL is adopted. ONNX Runtime Web has further adopted several novel optimization techniques for reducing data transfer between CPU and GPU, as well as some techniques to reduce GPU processing cycles to further push the performance to the maximum.

See [Compatibility](#Compatibility) and [Operators Supported](#Operators) for a list of platforms and operators ONNX Runtime Web currently supports.

## Usage

Refer to [ONNX Runtime JavaScript examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js) for samples and tutorials.

## Documents

### Developers

Refer to [Using VSCode](../README.md#Using-VSCode) for setting up development environment.

For information about building ONNX Runtime Web development, please check [Build](../README.md#build-2).

### Getting ONNX models

You can get ONNX models easily in multiple ways:

- Choose a pre-trained ONNX model from the [ONNX Model Zoo](https://github.com/onnx/models)
- Convert models from mainstream frameworks, e.g. PyTorch, TensorFlow and Keras, by following [ONNX tutorials](https://github.com/onnx/tutorials)
- Use your data to generate a customized ONNX model from [Azure Custom Vision service](https://docs.microsoft.com/en-us/azure/cognitive-services/Custom-Vision-Service/home)
- [Train a custom model in AzureML](https://github.com/Azure/MachineLearningNotebooks/tree/master/training) and save it in the ONNX format

Learn more about ONNX

- [ONNX website](http://onnx.ai/)
- [ONNX on GitHub](https://github.com/onnx/onnx)

### Compatibility

|    OS/Browser    |       Chrome       |        Edge        |       Safari       |      Electron      |
| :--------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|    Windows 10    | :heavy_check_mark: | :heavy_check_mark: |         -          | :heavy_check_mark: |
|      macOS       | :heavy_check_mark: |         -          | :heavy_check_mark: | :heavy_check_mark: |
| Ubuntu LTS 18.04 | :heavy_check_mark: |         -          |         -          | :heavy_check_mark: |
|       iOS        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |         -          |
|     Android      | :heavy_check_mark: |         -          |         -          |         -          |

### Operators

#### WebAssembly backend

ONNX Runtime Web currently support all operators in [ai.onnx](https://github.com/onnx/onnx/blob/master/docs/Operators.md) and [ai.onnx.ml](https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md).

#### WebGL backend

ONNX Runtime Web currently supports most operators in [ai.onnx](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md) operator set v7 (opset v7). See [operators.md](./docs/operators.md) for a complete, detailed list of which ONNX operators are supported by WebGL backend.

## License

License information can be found [here](https://github.com/microsoft/onnxruntime/blob/master/README.md#license).
