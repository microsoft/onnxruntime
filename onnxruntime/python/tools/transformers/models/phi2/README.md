# Phi2 Optimizations
## Prerequisites
```bash
pip install -r requirements.txt
```
_From source:_
```bash
pip install onnxruntime-gpu==1.17.0 # or onnxruntime==1.17.0 if using cpu
git clone git@github.com:microsoft/onnxruntime.git
cd onnxruntime/onnxruntime/python/tools/transformers
python -m models.phi2.convert_to_onnx -h
```
_From wheel:_ \
Install [ORT nightly package]([https://onnxruntime.ai/docs/install/](https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages))
```bash
python -m onnxruntime.transformers.models.phi2.convert_to_onnx -h
```

## Export optimized phi2 onnx model for different scenarios
**Export FP32 ONNX model for Nvidia GPUs** \
_From source:_
```
python -m models.phi2.convert_to_onnx --fp32_gpu
```
_From wheel:_
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp32_gpu
```
\
**Export FP16 ONNX model for Nvidia GPUs** \
_From source:_
```
python -m models.phi2.convert_to_onnx --fp32_gpu
```
_From wheel:_
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp32_gpu
```
\
**Export INT4 ONNX model for Nvidia GPUs** \
_From source:_
```
python -m models.phi2.convert_to_onnx --fp32_gpu
```
_From wheel:_
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp32_gpu
```
\
**Export FP16 ONNX model for Nvidia GPUs with CUDA architecture SM=80~89** \
_From source:_
```
python -m models.phi2.convert_to_onnx --fp16_gpu_sm8x
```
_From wheel:_
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp16_gpu_sm8x
```
\
**Export INT4 ONNX model for Nvidia GPUs with CUDA architecture SM=80~89** \
_From source:_
```
python -m models.phi2.convert_to_onnx --int4_gpu_sm8x
```
_From wheel:_
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --int4_gpu_sm8x
```
\
**Export FP32 ONNX model for CPU** \
_From source:_
```
python -m models.phi2.convert_to_onnx --fp32_cpu
```
_From wheel:_
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp32_cpu
```
\
**Export INT4 ONNX model for CPU** \
_From source:_
```
python -m models.phi2.convert_to_onnx --int4_cpu
```
_From wheel:_
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --int4_cpu
```
\
**Export all at once** \
_From source:_
```
python -m models.phi2.convert_to_onnx --fp32_cpu --int4_cpu --fp32_gpu --fp16_gpu --int4_gpu --fp16_gpu_sm8x --int4_gpu_sm8x
```
_From wheel:_
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp32_cpu --int4_cpu --fp32_gpu --fp16_gpu --int4_gpu --fp16_gpu_sm8x --int4_gpu_sm8x
```
## Run example with ORT and benchmark
**(e.g) Export FP16 and INT4 ONNX models for Nvidia GPUs with CUDA architecture SM=80~89 and run examples.** \
_From source:_
```
python -m models.phi2.convert_to_onnx --fp16_gpu_sm8x --int4_gpu_sm8x --run_example
```
_From wheel:_
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp16_gpu_sm8x --int4_gpu_sm8x --run_example
```
The inference example currently supports all models running on CUDA.

## Limitations
- It's a known issue that symbolic shape inference fails. It can be ignored at the moment as it won't affect the optimized model's inference.
- Torch dynamo export only support Linux. The model export cannot be run on Windows as of now.
