# ONNX Runtime Web

ONNX Runtime Web is a Javascript library for running ONNX models on browsers and on Node.js.

ONNX Runtime Web has adopted WebAssembly and WebGL technologies for providing an optimized ONNX model inference runtime for both CPUs and GPUs.

### Why ONNX models

The [Open Neural Network Exchange](http://onnx.ai/) (ONNX) is an open standard for representing machine learning models. The biggest advantage of ONNX is that it allows interoperability across different open source AI frameworks, which itself offers more flexibility for AI frameworks adoption.

### Why ONNX Runtime Web

With ONNX Runtime Web, web developers can score pre-trained ONNX models directly on browsers with various benefits of reducing server-client communication and protecting user privacy, as well as offering install-free and cross-platform in-browser ML experience.

ONNX Runtime Web can run on both CPU and GPU. For running on CPU, [WebAssembly](https://developer.mozilla.org/en-US/docs/WebAssembly) is adopted to execute the model at near-native speed. Furthermore, ONNX Runtime Web utilizes [Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers) to provide a "multi-threaded" environment to parallelize data processing. Empirical evaluation shows very promising performance gains on CPU by taking full advantage of WebAssembly and Web Workers. For running on GPUs, a popular standard for accessing GPU capabilities - WebGL is adopted. ONNX Runtime Web has further adopted several novel optimization techniques for reducing data transfer between CPU and GPU, as well as some techniques to reduce GPU processing cycles to further push the performance to the maximum.

See [Compatibility](#Compatibility) and [Operators Supported](#Operators) for a list of platforms and operators ONNX Runtime Web currently supports.

## Usage

Refer to [ONNX Runtime JavaScript examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js) for samples and tutorials.

## Documents

### Developement

Refer to the following links for development information:

- [Development](../README.md#Development)
- [Build](../README.md#Build-2)
- [Test](../README.md#Test)
- [Debugging](../README.md#Debugging)
- [Generating Document](../README.md#Generating-Document)

### Compatibility

|    OS/Browser    |   Chrome    |    Edge     |   Safari    |  Electron   | Node.js |
| :--------------: | :---------: | :---------: | :---------: | :---------: | :-----: |
|    Windows 10    | wasm, webgl | wasm, webgl |      -      | wasm, webgl |  wasm   |
|      macOS       | wasm, webgl |      -      | wasm, webgl | wasm, webgl |  wasm   |
| Ubuntu LTS 18.04 |    wasm     |      -      |      -      |    wasm     |  wasm   |
|       iOS        |    wasm     |    wasm     |  wasm       |      -      |    -    |
|     Android      |    wasm     |      -      |      -      |      -      |    -    |

### Operators

#### WebAssembly backend

ONNX Runtime Web currently support all operators in [ai.onnx](https://github.com/onnx/onnx/blob/master/docs/Operators.md) and [ai.onnx.ml](https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md).

#### WebGL backend

ONNX Runtime Web currently supports a subset of operators in [ai.onnx](https://github.com/onnx/onnx/blob/master/docs/Operators.md) operator set. See [operators.md](./docs/operators.md) for a complete, detailed list of which ONNX operators are supported by WebGL backend.

## License

License information can be found [here](https://github.com/microsoft/onnxruntime/blob/master/README.md#license).
