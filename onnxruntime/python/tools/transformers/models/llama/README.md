# LLaMA-2

## Exporting LLaMA-2

There are several ways to export LLaMA-2 models (using LLaMA-2 7B as an example).

### Option 1: from convert_to_onnx
```
# From source:
$ git clone https://github.com/microsoft/onnxruntime
$ cd onnxruntime/onnxruntime/python/tools/transformers/
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b
```

To make this option compatible with [Hugging Face's Optimum](https://github.com/huggingface/optimum), you will need to create `config.json` and `generation_config.json` for your model and store them in the same directory as your ONNX models. For example, you can find those JSON files for LLaMA-2 7B on Hugging Face [here](https://huggingface.co/meta-llama/Llama-2-7b-hf).

### Option 2: from [Microsoft's custom export](https://github.com/microsoft/Llama-2-Onnx)

Please follow the [README instructions](https://github.com/microsoft/Llama-2-Onnx#before-you-start) in the custom export of LLaMA-2.

### Option 3: from [Hugging Face Optimum](https://github.com/huggingface/optimum)

First, log into the Hugging Face CLI in your terminal:

```
$ huggingface-cli login
```

Once authenticated, run the following Python code to export:

```
from optimum.onnxruntime import ORTModelForCausalLM

name = "meta-llama/Llama-2-7b-hf"
model = ORTModelForCausalLM.from_pretrained(
    name,
    export=True,
    use_auth_token=True,
)
model.save_pretrained(name.split("/")[-1] + "-onnx")
```

## Examples of Exporting LLaMA-2

Here are some additional examples for exporting LLaMA-2.

Export Saved Model on Disk
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --input ./Llama-2-7b-hf --output ./llama2-7b

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --input ./Llama-2-7b-hf --output ./llama2-7b
```

Export for FP16
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp16 --precision fp16

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp16 --precision fp16
```

Export for INT8
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int8 --precision int8 --quantization_method smooth_quant

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int8 --precision int8 --quantization_method smooth_quant
```

Note: [Intel's Neural Compressor](https://github.com/intel/neural-compressor) takes time to run the SmoothQuant quantization algorithm on LLMs. On an [Azure Standard_NC24s_v3 VM](https://learn.microsoft.com/en-us/azure/virtual-machines/ncv3-series), it takes about ~30-45 min for each of the exported ONNX models.

## Benchmark LLaMA-2

Here are some examples of how you can benchmark LLaMA-2.

Note: In the below examples, `PyTorch` refers to running in PyTorch without `torch.compile` and `PyTorch 2.0` refers to running in PyTorch with `torch.compile`.

### Variants

1. PyTorch (without `torch.compile`), FP32
```
python3 -m models.llama.benchmark \
    --benchmark-type hf-pt \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp32 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cpu \
    --auth
```

2. PyTorch 2.0 (with `torch.compile`), FP16
```
python3 -m models.llama.benchmark \
    --benchmark-type hf-pt2 \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp16 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cuda \
    --auth
```

3. Optimum + ONNX Runtime, FP32, export via Optimum or convert_to_onnx
```
python3 -m models.llama.benchmark \
    --benchmark-type hf-ort \
    --hf-ort-model-path ./Llama-2-7b-hf-onnx/ \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp32 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cpu \
    --auth
```

4. Optimum + ONNX Runtime, FP16, export via convert_to_onnx
```
python3 -m models.llama.benchmark \
    --benchmark-type hf-ort \
    --hf-ort-model-path ./llama2-7b-fp16/ \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp16 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cuda \
    --auth
```

5. Optimum + ONNX Runtime, INT8, export via convert_to_onnx
```
python3 -m models.llama.benchmark \
    --benchmark-type hf-ort \
    --hf-ort-model-path ./llama2-7b-int8/ \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision int8 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cpu \
    --auth
```

6. ONNX Runtime, FP32, Microsoft custom export
```
python3 -m models.llama.benchmark \
    --benchmark-type ort \
    --ort-model-path llama-2-onnx/7B_float32/ONNX/LlamaV2_7B_float32.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp32 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cpu
```

7. ONNX Runtime, FP16, Microsoft custom export
```
python3 -m models.llama.benchmark \
    --benchmark-type ort \
    --ort-model-path ./llama-2-onnx/7B_float16/ONNX/LlamaV2_7B_float16.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp16 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cuda
```

You can profile a variant by adding the `--profile` flag and providing one batch size and sequence length combination.

### Benchmark All
You can use `benchmark_all.py` to benchmark across various platforms and automatically store the results in a CSV file. Here is an example.
```
python3 -m models.llama.benchmark_all \
    --hf-ort-model-path ./llama2-7b-fp16/ \
    --ort-model-path ./llama-2-onnx/7B_float16/ONNX/LlamaV2_7B_float16.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp16 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cuda
```
