ONNX format models are used for types that are only supported in the full build.
ORT format models are used for types that are supported in both the full and mobile builds.

`test_types_*.ort` ORT format models are converted from
`js/node/test/testdata/test_types_*.onnx` ONNX models.

run:

```bash
cd <repo root>/js
python -m onnxruntime.tools.convert_onnx_models_to_ort \
  --optimization_style Fixed \
  --output_dir ./react_native/e2e/android/app/src/main/assets \
  ./node/test/testdata
python -m onnxruntime.tools.convert_onnx_models_to_ort \
  --optimization_style Fixed \
  --output_dir ./react_native/e2e/src \
  ./node/test/testdata
```

Some additional files will be generated. They can be removed.
