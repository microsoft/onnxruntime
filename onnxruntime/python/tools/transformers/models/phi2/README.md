# Phi2 Optimizations
## Prerequisites
```
pip install -r requirements.txt
```
- From source: \
pip install onnxruntime-gpu==1.17.0
```
git clone git@github.com:microsoft/onnxruntime.git
cd onnxruntime/onnxruntime/python/tools/transformers
python -m models.phi2.convert_to_onnx -h
```
- From wheel: \
pip install [ort-nightly-gpu](https://onnxruntime.ai/docs/install/)
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx -h
```

## Export optimized phi2 onnx model for different senarios
- Export FP32 ONNX model for Nvidia GPUs \
From source:
```
python -m models.phi2.convert_to_onnx --fp32_gpu
```
From wheel:
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp32_gpu
```
- Export FP16 ONNX model for Nvidia GPUs 
From source:
```
python -m models.phi2.convert_to_onnx --fp32_gpu
```
From wheel:
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp32_gpu
```
- Export INT4 ONNX model for Nvidia GPUs 
From source:
```
python -m models.phi2.convert_to_onnx --fp32_gpu
```
From wheel:
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp32_gpu
```
- Export FP16 ONNX model for Nvidia A100 
From source:
```
python -m models.phi2.convert_to_onnx --fp16_a100
```
From wheel:
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp16_a100
```
- Export INT4 ONNX model for Nvidia A100 
From source:
```
python -m models.phi2.convert_to_onnx --int4_a100
```
From wheel:
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --int4_a100
```
- Export FP32 ONNX model for CPU 
From source:
```
python -m models.phi2.convert_to_onnx --fp32_cpu
```
From wheel:
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp32_cpu
```
- Export INT4 ONNX model for CPU 
From source:
```
python -m models.phi2.convert_to_onnx --int4_cpu
```
From wheel:
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --int4_cpu
```
- Export all of them at once
From source:
```
python -m models.phi2.convert_to_onnx --fp32_cpu --int4_cpu --fp32_gpu --fp16_gpu --int4_gpu --fp16_a100 --int4_a100
```
From wheel:
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp32_cpu --int4_cpu --fp32_gpu --fp16_gpu --int4_gpu --fp16_a100 --int4_a100
```
## Run example with ORT and benchmark
- (e.g) Export FP16 and INT4 ONNX models for Nvidia A100 and run examples.
From source:
```
python -m models.phi2.convert_to_onnx --fp16_a100 --int4_a100 --run_example
```
From wheel:
```
python -m onnxruntime.transformers.models.phi2.convert_to_onnx --fp16_a100 --int4_a100 --run_example
```
The inference example currently supports all models running on CUDA.

## Limitations
There's a known issue that symbolic shape inference will fail. It can be ignored at the moment as it won't affect the optimized model's inference.



