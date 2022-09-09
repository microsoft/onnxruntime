`js/react_native/android/src/androidTest/res/raw/test_types_*.ort` and
`js/react_native/ios/OnnxruntimeModuleTest/Resources/test_types_*.ort` ORT format models are converted from
`js/node/test/testdata/test_types_*.onnx` ONNX models.

For example, to generate `js/react_native/android/src/androidTest/res/raw/test_types_*.ort`, from the `js` directory,
run:

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort \
  --optimization_style Fixed \
  --output_dir ./react_native/android/src/androidTest/res/raw \
  ./node/test/testdata
```

Some additional files will be generated. They can be removed.
