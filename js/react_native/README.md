# onnxruntime-react-native

ONNX Runtime React Native provides a JavaScript library for running ONNX models on React Native app.

### Why ONNX models

The [Open Neural Network Exchange](http://onnx.ai/) (ONNX) is an open standard for representing machine learning models. The biggest advantage of ONNX is that it allows interoperability across different open source AI frameworks, which itself offers more flexibility for AI frameworks adoption.

### Why ONNX Runtime React Native

With ONNX Runtime React Native, React Native developers can score pre-trained ONNX models directy on React Native apps by leveraging [ONNX Runtime Mobile](https://www.onnxruntime.ai/docs/reference/mobile/prebuilt-package/), so it provides a light-weight inference solution for Android and iOS.

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

### Operator and type support

ONNX Runtime React Native currently supports most operators used by popular models. Refer to [ONNX Runtime Mobile Pacakge Operator and Type](https://www.onnxruntime.ai/docs/reference/mobile/prebuilt-package/1.8%20ORTMobilePackageOperatorTypeSupport).

### License

License information can be found [here](https://github.com/microsoft/onnxruntime/blob/master/README.md#license).
