# onnxruntime-react-native

ONNX Runtime React Native provides a JavaScript library for running ONNX models on React Native app.

## Installation

```sh
yarn add onnxruntime-react-native
```

## Usage

```js
import { InferenceSession } from "onnxruntime-react-native";

// load a model
const session: InferenceSession = await InferenceSession.create(modelPath);
// input as InferenceSession.OnnxValueMapType
const result = session.run(input, ['num_detection:0', 'detection_classes:0'])
```

Refer to [ONNX Runtime JavaScript examples](https://github.com/microsoft/onnxruntime-inference-examples/js) for samples and tutorials. Different from other JavaScript frameworks like node.js and web, React Native library doesn't support these features

- Unsigned data type at Tensor
- Model loading using ArrayBuffer

## License

License information can be found [here](https://github.com/microsoft/onnxruntime/blob/master/README.md#license).
