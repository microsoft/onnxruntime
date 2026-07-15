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
│   ├───input_0.pb
│   └───input_1.pb
├───test_data_set_1
│   ├───input_0.pb
│   └───input_1.pb
├───test_data_set_2
│   ├───input_0.pb
│   └───input_1.pb
├───test_data_set_3
│   ├───input_0.pb
│   └───input_1.pb
├───test_data_set_4
│   ├───input_0.pb
│   └───input_1.pb
├───test_data_set_5
│   ├───input_0.pb
│   └───input_1.pb
├───test_data_set_6
│   ├───input_0.pb
│   └───input_1.pb
├───test_data_set_7
│   ├───input_0.pb
│   └───input_1.pb
├───test_data_set_8
│   ├───input_0.pb
│   └───input_1.pb
└───test_data_set_9
    ├───input_0.pb
    └───input_1.pb
```

Note that the indexing must fully align the order of model inputs (i.e., `input_0.pb` is expected to be the data for the 1st model input, `input_1.pb` for the 2nd, and so on).

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

### Choosing parameters for CPU inference

The right combination of quantization parameters depends on the target CPU architecture.
Choosing the wrong combination is a common cause of quantized models running *slower* than FP32.

**Format**

- Use `quant_format=QuantFormat.QDQ` (the default since ORT 1.11). ORT's CPU kernels are optimized
  for the QDQ representation. `QOperator` format is mainly useful for specific hardware back-ends.

**Activation type and weight type by platform**

- x86/x64 without VNNI (most pre-Skylake-SP desktop/laptop CPUs):
    - `activation_type=QuantType.QUInt8`, `weight_type=QuantType.QInt8`
    - Set `reduce_range=True` to quantize weights to 7-bit to reduce the risk of integer saturation
      on CPUs that typically lack the VNNI dot-product instruction.

- x86/x64 with VNNI (e.g., Intel Skylake-SP/Cascade Lake/Ice Lake/Sapphire Rapids or AMD Zen4 and
  later, though exact support varies by SKU):
    - `activation_type=QuantType.QUInt8`, `weight_type=QuantType.QInt8`
    - `reduce_range=False` — VNNI-capable cores typically accumulate 8-bit products without
      saturation, so range reduction is often unnecessary.

- ARM (Cortex-A, Apple Silicon, Graviton):
    - `activation_type=QuantType.QInt8`, `weight_type=QuantType.QInt8`
    - `reduce_range=False` — ARM NEON/SVE generally handles signed 8-bit arithmetic without
      saturation issues.

**per_channel**

- `per_channel=False` (default) gives better CPU throughput. `per_channel=True` can improve accuracy
  for models whose weight distributions differ substantially across output channels.

**Example: x64 non-VNNI**

```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

# reader: CalibrationDataReader = MyCalibrationDataReader(...)  # user-supplied

quantize_static(
    model_input="model.onnx",
    model_output="model_quant.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    reduce_range=True,   # recommended on non-VNNI x64
    per_channel=False,
)
```

Note: this guidance applies to models produced by `quantize_static`. The separate
`convert_onnx_models_to_ort` tool's `--target_platform` flag only affects ORT format conversion
and does not change the quantization parameters above.

For the full quantization guide see
https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

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
