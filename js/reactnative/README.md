# -onnxruntime-reactnative

Onnxruntime bridge for react native

## Installation

```sh
npm install onnxruntime-common
npm install onnxruntime-reactnative
```

## Usage

```js
import { InferenceSesion } from "onnxruntime-reactnative";

// ...

const session: InferenceSession = await InferenceSession.create(modelPath);
// input as InferenceSession.OnnxValueMapType
const result = sesion.run(input, ['num_detection:0', 'detection_classes:0'])
```

## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

MIT
