# ONNX Runtime Web

ONNX Runtime Web is a Javascript library for running ONNX models on browsers and on Node.js.

ONNX Runtime Web has adopted WebAssembly and WebGL technologies for providing an optimized ONNX model inference runtime for both CPUs and GPUs.

### Why ONNX models

The [Open Neural Network Exchange](http://onnx.ai/) (ONNX) is an open standard for representing machine learning models. The biggest advantage of ONNX is that it allows interoperability across different open source AI frameworks, which itself offers more flexibility for AI frameworks adoption. See [Getting ONNX Models](#Getting-ONNX-models).

### Why ONNX Runtime Web

With ONNX Runtime Web, web developers can score pre-trained ONNX models directly on browsers with various benefits of reducing server-client communication and protecting user privacy, as well as offering install-free and cross-platform in-browser ML experience.

ONNX Runtime Web can run on both CPU and GPU. For running on CPU, [WebAssembly](https://developer.mozilla.org/en-US/docs/WebAssembly) is adopted to execute the model at near-native speed. Furthermore, ONNX Runtime Web utilizes [Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers) to provide a "multi-threaded" environment to parallelize data processing. Empirical evaluation shows very promising performance gains on CPU by taking full advantage of WebAssembly and Web Workers. For running on GPUs, a popular standard for accessing GPU capabilities - WebGL is adopted. ONNX Runtime Web has further adopted several novel optimization techniques for reducing data transfer between CPU and GPU, as well as some techniques to reduce GPU processing cycles to further push the performance to the maximum.

See [Compatibility](#Compatibility) and [Operators Supported](#Operators) for a list of platforms and operators ONNX Runtime Web currently supports.

## Getting Started

There are multiple ways to use ONNX Runtime Web in a project:

### Using `<script>` tag

This is the most straightforward way to use ONNX Runtime Web. The following HTML example shows how to use it:

```html
<html>
  <head> </head>

  <body>
    <!-- Load ONNX Runtime Web -->
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <!-- Code that consume ONNX Runtime Web -->
    <script>
      async function runMyModel() {
        // create a session
        const myOrtSession = await ort.InferenceSession.create(
          "./my-model.onnx"
        );
        // generate model input
        const input0 = new ort.Tensor(
          new Float32Array([1.0, 2.0, 3.0, 4.0]) /* data */,
          [2, 2] /* dims */
        );
        // execute the model
        const outputs = await myOrtSession.run({ input_0: input0 });
        // consume the output
        const outputTensor = outputs["output_0"];
        console.log(`model output tensor: ${outputTensor.data}.`);
      }
      runMyModel();
    </script>
  </body>
</html>
```

<!-- TODO: Refer to [browser/Add](./examples/browser/add) for an example. -->

### Using NPM and bundling tools

Modern browser based applications are usually built by frameworks like [Angular](https://angular.io/), [React](https://reactjs.org/), [Vue.js](https://vuejs.org/) and so on. This solution usually builds the source code into one or more bundle file(s). The following TypeScript example shows how to use ONNX Runtime Web in an async context:

1. Import `Tensor` and `InferenceSession`.

```ts
import { Tensor, InferenceSession } from "onnxruntime-web";
```

2. Create an instance of `InferenceSession` and load ONNX model.

```ts
// use the following in an async method
const url = "./data/models/resnet/model.onnx";
const session = await InferenceSession.create(url);
```

3. Create your input Tensor(s) similar to the example below. You need to do any pre-processing required by
   your model at this stage. For that refer to the documentation of the model you have:

```javascript
// creating an array of input Tensors is the easiest way. For other options see the API documentation
const input0 = new Tensor(new Float32Array([1.0, 2.0, 3.0, 4.0]), [2, 2]);
```

4. Run the model with the input Tensors. The output Tensor(s) are available once the run operation is complete:

```javascript
// run this in an async method:
// assume model's input name is 'input_0' and output name is 'output_0'
const outputs = await session.run({ input_0: input0 });
const outputTensor = outputs.output_0;
```

5. Bundle your code. All web application frameworks offer bundling tools and instructions. Specifically, you can specify onnxruntime-web as an external dependency:

```js
  // a webpack example
  externals: {
    'onnxruntime-web': 'ort', // add this line in your webpack.config.js
    // ...
  }
```

so that you can consume the file `ort.min.js` from a CDN provider demonstrated as above.

<!-- TODO More verbose examples on how to use ONNX Runtime Web are located under the `examples` folder. For further info see [Examples](./examples/README.md) -->

## Documents

### Developers

<!-- TODO development documents and API -->

For information on ONNX.js development, please check [Development](./docs/development.md)

For API reference, please check [API](./docs/api.md).

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
