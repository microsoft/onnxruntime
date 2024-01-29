# Phi2 Optimizations
## Prerequisites
```
git clone git@github.com:microsoft/onnxruntime.git
cd onnxruntime/onnxruntime/python/tools/transformers/models/phi2
pip install -r requirements.txt
```

## Export optimized onnx model for different senarios

- Export FP32 ONNX model for CPU 
```
python convert_to_onnx.py --fp32_cpu
```
- Export INT4 ONNX model for CPU 
```
python convert_to_onnx.py --int4_cpu
```
- Export FP32 ONNX model for Nvidia GPUs 
```
python convert_to_onnx.py --fp32_gpu
```
- Export FP16 ONNX model for Nvidia GPUs 
```
- python convert_to_onnx.py --fp16_gpu
```
- Export INT4 ONNX model for Nvidia GPUs 
```
python convert_to_onnx.py --int4_gpu
```
- Export FP16 ONNX model for Nvidia A100 
```
python convert_to_onnx.py --fp16_a100
```
- Export INT4 ONNX model for Nvidia A100 
```
python convert_to_onnx.py --int4_a100
```
- Export all of them 
```
python convert_to_onnx.py --fp32_cpu --int4_cpu --fp32_gpu --fp16_gpu --int4_gpu --fp16_a100 --int4_a100
```
## Run example with ORT and benchmark
- Export FP16 ONNX model for Nvidia A100 and run example
```
python convert_to_onnx.py --fp16_a100 --run_example
```

## Limitations
There's a known issue that symbolic shape inference will fail. It can be ignored at the moment as it won't affect the optimized model's inference.



