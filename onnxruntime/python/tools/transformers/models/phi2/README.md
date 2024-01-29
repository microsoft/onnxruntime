# Phi2 Optimizations
### Prerequisites
`$ pip install -r requirements.txt`

### Export optimized onnx model for different senarios
#### Generate fp32 ONNX model for CPU
`$ python convert_to_onnx.py --fp32_cpu`
#### Generate int4 ONNX model for CPU
`$ python convert_to_onnx.py --int4_cpu`
#### Generate fp32 ONNX model for Nvidia GPUs
`$ python convert_to_onnx.py --fp32_gpu`
#### Generate fp16 ONNX model for Nvidia GPUs
`$ python convert_to_onnx.py --fp16_gpu`
#### Generate int4 ONNX model for Nvidia GPUs
`$ python convert_to_onnx.py --int4_gpu`
#### Generate fp16 ONNX model for Nvidia A100
`$ python convert_to_onnx.py --fp16_a100`
#### Generate int4 ONNX model for Nvidia A100
`$ python convert_to_onnx.py --int4_a100`
#### Generate all of them
`$ python convert_to_onnx.py --fp32_cpu --int4_cpu --fp32_gpu --fp16_gpu --int4_gpu --fp16_a100 --int4_a100`

### Example run with ORT and benchmark
```
# For example: Export fp16 ONNX model for Nvidia A100 and run example:
$ python convert_to_onnx.py --fp16_a100 --run_example
```



