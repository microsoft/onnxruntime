# Quantization Tool
This tool can be used to quantize selected ONNX models. Support is based on operators in the model. Please refer to https://onnxruntime.ai/docs/performance/quantization.html for usage details and https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization for examples.

## Static Quantization Tool

### Build
Please add `--enable_pybind` and `--build_wheel` to the build command to acquire the python tools.

```bash
cd onnxruntime
.\build.bat --config RelWithDebInfo --build_shared_lib --parallel --cmake_generator "Visual Studio 17 2022" --enable_pybind --build_wheel
```

### Model and Data
The static quantization tool expects the directory structure of model and data.

```ps1
work_dir\resnet18-v1-7
├───model.onnx
├───test_data_set_0
├───test_data_set_1
├───test_data_set_2
├───test_data_set_3
├───test_data_set_4
├───test_data_set_5
├───test_data_set_6
├───test_data_set_7
├───test_data_set_8
└───test_data_set_9
```

### Usage
Install the python tools built in onnxruntime
```ps1
cd work_dir
python -m venv ort_env
ort_env\Scripts\activate
python -m pip install <path-to-built-folder>\RelWithDebInfo\RelWithDebInfo\dist\<name-of-the-wheel>.whl

# The following command yields model_quant.onnx under the same directory "resnet18-v1-7"
python -m onnxruntime.quantization.static_quantize_runner -i resnet18-v1-7\model.onnx -o resnet18-v1-7\model_quant.onnx

work_dir\resnet18-v1-7
├───model.onnx
├───model_quant.onnx
├───test_data_set_0
│   ...
└───test_data_set_9
```

### Quantization Arguments
Please refer to `static_quantize_runner.py` for more detailed arguments.

```ps1
python -m onnxruntime.quantization.static_quantize_runner -i resnet18-v1-7\model.onnx -o resnet18-v1-7\model_quant.onnx --activation_type qint8 --weight_type qint16
python -m onnxruntime.quantization.static_quantize_runner -i resnet18-v1-7\model.onnx -o resnet18-v1-7\model_quant.onnx --activation_type qint16 --weight_type qint16 --quantize_bias
python -m onnxruntime.quantization.static_quantize_runner -i resnet18-v1-7\model.onnx -o resnet18-v1-7\model_quant.onnx --activation_type qint16 --weight_type qint8 --per_channel
```

### Tensor Quant Overrides Json Format
With `--tensor_quant_overrides`, the tool can consume the json file with quantization override information.
```ps1
python -m onnxruntime.quantization.static_quantize_runner -i resnet18-v1-7\model.onnx -o resnet18-v1-7\model_quant.onnx --tensor_quant_overrides <path-to-json>\encoding.json
```

The tool expects the encoding.json with the format:
```json
{
    "conv1_1": [
        {
            "scale": 0.005,
            "zero_point": 12
        }
    ]
}
```
- Each key is the name of a tensor in the onnx model.
    - e.g. "conv1_1"
- For each tensor, a list of dictionary should be provided
    - For per-tensor quantization, the list contains a single dictionary.
    - For per-channel quantization, the list contains a dictionary for each channel in the tensor.
    - Each dictionary contain the information required for quantization including:
        - scale (float)
        - zero_point (int)
