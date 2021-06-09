# onnxruntime-react-native

ONNX Runtime React Native provides a JavaScript library for running ONNX models on React Native app.

### Why ONNX models

The [Open Neural Network Exchange](http://onnx.ai/) (ONNX) is an open standard for representing machine learning models. The biggest advantage of ONNX is that it allows interoperability across different open source AI frameworks, which itself offers more flexibility for AI frameworks adoption.

### Why ONNX Runtime React Native

With ONNX Runtime React Native, React Native developers can score pre-trained ONNX models directy on React Native apps by leveraging the native [ONNX Runtime](http://www.onnxruntime.ai/docs/) CPU engine, so it supports most functionalities native ONNX Runtime offers, including full ONNX operator coverage, multi-threading, [ONNX Runtime Quantization](https://www.onnxruntime.ai/docs/how-to/quantization.html) as well as [ONNX Runtime Mobile](http://www.onnxruntime.ai/docs/how-to/deploy-on-mobile.html).

### Installation

```sh
yarn add onnxruntime-react-native
```

### Usage

```js
import { InferenceSession } from "onnxruntime-react-native";

// load a model
const session: InferenceSession = await InferenceSession.create(modelPath);
// input as InferenceSession.OnnxValueMapType
const result = session.run(input, ['num_detection:0', 'detection_classes:0'])
```

Refer to [ONNX Runtime JavaScript examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js) for samples and tutorials. Different from other JavaScript frameworks like node.js and web, React Native library doesn't support these features.

- Unsigned data type at Tensor
- Model loading using ArrayBuffer

### Operator

ONNX Runtime React Native currently support all operators in [ai.onnx](https://github.com/onnx/onnx/blob/master/docs/Operators.md) and [ai.onnx.ml](https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md).

### License

License information can be found [here](https://github.com/microsoft/onnxruntime/blob/master/README.md#license).
