`js/react_native/e2e/src/mnist.onnx` is `onnxruntime/test/testdata/mnist.onnx` updated to opset 15.

```bash
cd <repo root>/js
python -m onnxruntime.tools.update_onnx_opset --opset 15 ../onnxruntime/test/testdata/mnist.onnx ./react_native/e2e/src/mnist.onnx
```

`js/react_native/e2e/src/mnist.ort` and `js/react_native/e2e/android/app/src/main/assets/mnist.ort` are converted from `js/react_native/e2e/src/mnist.onnx`.

```bash
cd <repo root>/js
python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_style=Fixed --output_dir ./react_native/e2e/android/app/src/main/assets ./react_native/e2e/src/mnist.onnx
python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_style=Fixed --output_dir ./react_native/e2e/src ./react_native/e2e/src/mnist.onnx
```
