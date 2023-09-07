# onnxruntime-react-native

ONNX Runtime React Native provides a JavaScript library for running ONNX models in a React Native app.

### Why ONNX models

The [Open Neural Network Exchange](http://onnx.ai/) (ONNX) is an open standard for representing machine learning models. The biggest advantage of ONNX is that it allows interoperability across different open source AI frameworks, which itself offers more flexibility for AI frameworks adoption.

### Why ONNX Runtime React Native

With ONNX Runtime React Native, React Native developers can score pre-trained ONNX models directly in React Native apps by leveraging [ONNX Runtime](https://onnxruntime.ai/docs/), so it provides a light-weight inference solution for Android and iOS.

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

Refer to [ONNX Runtime JavaScript examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js) for samples and tutorials. The ONNX Runtime React Native library does not currently support the following features:

- Tensors with unsigned data types, with the exception of uint8 on Android devices
- Model loading using ArrayBuffer

### Operator and type support

ONNX Runtime React Native version 1.13 supports both ONNX and ORT format models, and includes all operators and types.

Previous ONNX Runtime React Native packages use the ONNX Runtime Mobile package, and support operators and types used in popular mobile models.
See [here](https://onnxruntime.ai/docs/reference/operators/MobileOps.html) for the list of supported operators and types.

### License

License information can be found [here](https://github.com/microsoft/onnxruntime/blob/main/README.md#license).
