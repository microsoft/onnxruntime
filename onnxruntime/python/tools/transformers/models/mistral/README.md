# Mistral

## Introduction

This README will cover the steps to export and optimize Mistral in ORT, which is largely shared with the LLAMA2 files located in the `models/llama` directory. 

## Prerequisites

See the Prerequisites section in `models/llama/README.md`

## Exporting Mistral

There is currently one supported way to export Mistral to ONNX format:

### [Hugging Face Optimum](https://github.com/huggingface/optimum)

Note that this may produce two ONNX models with older Optimum versions. If this occurs, you will need to update your Optimum version so only one merged model is produced.

The following command will export Mistral in full precision:
```
python -m optimum.exporters.onnx -m mistralai/Mistral-7B-v0.1 --library-name transformers /path/to/model/directory
```

## Optimizing & Quantizing Mistral

To quantize Mistral to FP16 and apply fusion optimizations, you can run the following command:
```
python optimize.py --onnx-model-path /path/to/mistral/model.onnx --output-path /path/to/optimized/model.opt.onnx -hf mistralai/Mistral-7B-v0.1
```

## Benchmark Mistral
The benchmarking script in the llama directory (`models/llama/benchmark.py`) directory support Mistral benchmarking by including the flags `-bt mistral-ort` for benchmarking the ONNX model produced by the above, or `-bt mistral-hf`, which will benchmark the HuggingFace implementation of Mistral. For example, to benchmark the ORT and HF versions respectively, you can run: 

```
python models/llama/benchmark.py -bt mistral-ort -p fp16 -m mistralai/Mistral-7B-v0.1 --ort-model-path /dev_data/petermca/mistral/model_opt/model.quant.onnx
python models/llama/benchmark.py -bt mistral-hf -p fp16 -m mistralai/Mistral-7B-v0.1
```

