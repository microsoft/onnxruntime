# LLaMA

## Exporting LLaMA

There are two ways to export LLaMA models such as LLaMA and LLaMA2 (using LLaMA2 7B as an example).

Option 1: from source
```
$ git clone https://github.com/microsoft/onnxruntime
$ cd onnxruntime/onnxruntime/python/tools/transformers/models/llama
$ python3 convert_to_onnx.py -m meta-llama/Llama-2-7b-hf --output llama2-7b
```

Option 2: from wheel
```
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b
```

# Examples of Exporting LLaMA

Here are some additional examples for exporting LLaMA.

## Export Saved Model on Disk
```
# From source:
$ python3 convert_to_onnx.py -m meta-llama/Llama-2-7b-hf --input ./Llama-2-7b-hf --output ./llama2-7b

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --input ./Llama-2-7b-hf --output ./llama2-7b
```

## Export with Different Precision

FP16:
```
# From source:
$ python3 convert_to_onnx.py -m meta-llama/Llama-2-7b-hf --output llama2-7b --precision fp16

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b --precision fp16
```

INT8:
```
# From source:
$ python3 convert_to_onnx.py -m meta-llama/Llama-2-7b-hf --output llama2-7b --precision int8 --quantize_embedding_layer

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b --precision int8 --quantize_embedding_layer
```

## Benchmark LLaMA

Here are some examples of how you can benchmark LLaMA.

Note: In the below examples, `PyTorch` refers to running in PyTorch without `torch.compile` and `PyTorch 2.0` refers to running in PyTorch with `torch.compile`.

### Variants

1. PyTorch (without `torch.compile`), FP32, Hugging Face `generate()` and `decode()` API
```
python3 benchmark.py \
    --benchmark-type hf-pt \
    --model-name meta-llama/Llama-2-7b-hf \
    --model-size 7b \
    --precision fp32 \
    --batch-size 2 \
    --device cpu \
    --prompt "Hey, are you conscious? Can you talk to me?" \
    --auth
```

2. PyTorch 2.0 (with `torch.compile`), FP16, Hugging Face `generate()` and `decode()` API
```
python3 benchmark.py \
    --benchmark-type hf-pt2 \
    --model-name meta-llama/Llama-2-7b-hf \
    --model-size 7b \
    --precision fp16 \
    --batch-size 2 \
    --device cuda \
    --prompt "Hey, are you conscious? Can you talk to me?" \
    --auth
```

3. ONNX Runtime, FP32
```
python3 benchmark.py \
    --benchmark-type ort \
    --ort-model-path llama-2-onnx/7B_float32/ONNX/LlamaV2_7B_float32.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --model-size 7b \
    --precision fp32 \
    --batch-size 1 \
    --sequence-length 16 \
    --device cpu
```

4. ONNX Runtime, FP16
```
python3 benchmark.py \
    --benchmark-type ort \
    --ort-model-path llama-2-onnx/7B_float16/ONNX/LlamaV2_7B_float16.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --model-size 7b \
    --precision fp16 \
    --batch-size 1 \
    --sequence-length 16 \
    --device cuda
```
