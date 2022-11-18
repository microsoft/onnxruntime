The resources in 'android/src/androidTest/res/raw' and 'ios/OnnxruntimeModuleTest/Resources' contain a mix of .ORT and .ONNX format models.

Originally the ORT React Native package used the ORT Mobile build, which only supports .ORT format models, and does not support the double or bool types.

We're now using the 'full' ORT build in the React Native package, so both model formats and all types are supported.

ONNX format models are used for types that are only supported in the full build.
ORT format models are used for types that are supported in both the full and mobile builds.

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
