# onnxruntime-react-native

ONNX Runtime React Native provides a JavaScript library for running ONNX models in a React Native app.

### Why ONNX models

The [Open Neural Network Exchange](http://onnx.ai/) (ONNX) is an open standard for representing machine learning models. The biggest advantage of ONNX is that it allows interoperability across different open source AI frameworks, which itself offers more flexibility for AI frameworks adoption.

### Why ONNX Runtime React Native

With ONNX Runtime React Native, React Native developers can score pre-trained ONNX models directly in React Native apps by leveraging [ONNX Runtime](https://onnxruntime.ai/docs/), so it provides a light-weight inference solution for Android and iOS.

### Installation

```sh
npm install onnxruntime-react-native
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

### Bundling ONNX Models (Expo)

When using Expo, the config plugin can automatically bundle `.onnx` and `.ort` model files into your native app. Specify model paths in your `app.json` (or `app.config.js`):

```json
{
  "expo": {
    "plugins": [
      [
        "onnxruntime-react-native",
        {
          "models": [
            "./assets/models/my_model.onnx",
            "./assets/models/my_model_quantized.ort"
          ]
        }
      ]
    ]
  }
}
```

Paths are relative to your project root. Running `expo prebuild` will:

- **iOS**: Copy each model into the Xcode project and add it to the "Copy Bundle Resources" build phase.
- **Android**: Copy each model into `android/app/src/main/assets/`.

If no `models` config is provided, the plugin behaves as before (native linking only).

#### Loading Bundled Models at Runtime

Add `onnx` and `ort` to your `metro.config.js` asset extensions:

```js
// metro.config.js
const { getDefaultConfig } = require("expo/metro-config");
const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push("onnx", "ort");
module.exports = config;
```

Then load the model at runtime. The example below tries the native bundle first (production builds where the plugin has copied the model), then falls back to Metro's asset system (dev server / simulator):

```js
import { Asset } from "expo-asset";
import * as FileSystem from "expo-file-system";
import { Platform } from "react-native";
import { InferenceSession } from "onnxruntime-react-native";

async function resolveModelPath() {
  // Production iOS: plugin bundles the model into the native app bundle
  if (Platform.OS === "ios" && FileSystem.bundleDirectory) {
    const bundlePath = `${FileSystem.bundleDirectory}my_model.onnx`;
    const info = await FileSystem.getInfoAsync(bundlePath);
    if (info.exists) return bundlePath;
  }

  // Fallback: Metro asset system (dev server / simulator).
  // On Android this works in both dev and production — the plugin copies
  // models to assets/ which is where the Android native code already looks.
  const [asset] = await Asset.loadAsync(
    require("./assets/models/my_model.onnx")
  );
  if (!asset.localUri) throw new Error("Failed to resolve model URI");
  return asset.localUri;
}

const session = await InferenceSession.create(await resolveModelPath());
```

### Operator and type support

ONNX Runtime React Native version 1.13 supports both ONNX and ORT format models, and includes all operators and types.

Previous ONNX Runtime React Native packages use the ONNX Runtime Mobile package, and support operators and types used in popular mobile models.
See [here](https://onnxruntime.ai/docs/reference/operators/MobileOps.html) for the list of supported operators and types.

### License

License information can be found [here](https://github.com/microsoft/onnxruntime/blob/main/README.md#license).
