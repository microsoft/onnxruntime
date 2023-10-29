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

As indicated in `requirements.txt`, you will also need to install Optimum from source. Once installed, you will need to modify `ORTModelForCausalLM.forward` in `optimum/optimum/onnxruntime/modeling_decoder.py` as follows:

```
# Before
if self.use_cache:
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]
        # Flatten the past_key_values (no need to flatten for models using multi-query attn)


# After
if self.use_cache:
    if past_key_values is not None:
        input_ids = input_ids[:, -1:] if past_key_values[0][0].shape[2] != 0 else input_ids
        # Flatten the past_key_values (no need to flatten for models using multi-query attn)
```

### Option 2: from [Microsoft's custom export](https://github.com/microsoft/Llama-2-Onnx)

Please follow the [README instructions](https://github.com/microsoft/Llama-2-Onnx#before-you-start) in the custom export of LLaMA-2.

### Option 3: from [Hugging Face Optimum](https://github.com/huggingface/optimum)

Note that this will produce two ONNX models whereas the above two options produce one ONNX model. 

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

Export for FP32 CUDA
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp32-gpu --precision fp32 --execution_provider cuda

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp32 --precision fp32 --execution_provider cuda
```

Export for FP32 CPU
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp32-cpu --precision fp32 --execution_provider cpu

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp32 --precision fp32 --execution_provider cpu
```

Export for FP16 CUDA
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp16 --precision fp16 --execution_provider cuda

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp16 --precision fp16 --execution_provider cuda
```

Export for INT8 CPU (SmoothQuant)
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int8 --precision int8 --quantization_method smooth_quant --execution_provider cpu

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int8 --precision int8 --quantization_method smooth_quant --execution_provider cpu
```

Note: [Intel's Neural Compressor](https://github.com/intel/neural-compressor) takes time to run the SmoothQuant quantization algorithm on LLMs. On an [Azure Standard_NC24s_v3 VM](https://learn.microsoft.com/en-us/azure/virtual-machines/ncv3-series), it takes about ~30-45 min for each of the exported ONNX models.

Export for INT8 CPU (DynamicQuant)
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int8 --precision int8 --quantization_method quantize_dynamic --execution_provider cpu

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int8 --precision int8 --quantization_method quantize_dynamic --execution_provider cpu
```

Export for INT4 CUDA
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int4-gpu --precision int4 --quantization_method blockwise --execution_provider cuda

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int4 --precision int4 --quantization_method blockwise --execution_provider cuda
```

Export for INT4 CPU
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int4-cpu --precision int4 --quantization_method blockwise --execution_provider cpu

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int4 --precision int4 --quantization_method blockwise --execution_provider cpu
```

Export Sharded model, llama-70b into 4 partitions
```
# From source:
$ 1. Get OnnxRuntime code from https://github.com/frankdongms/transformers/tree/frdong/shard_llama or
$    wait until PR: https://github.com/huggingface/transformers/pull/27119 got merged into HF transformers
$ 2. Build OnnxRuntime from source with NCCL enabled, sample command: ./build.sh --config RelWithDebInfo --use_cuda --cuda_home /usr/local/cuda-12.2 --cudnn_home /usr/local/cuda-12.2 --build_wheel --cuda_version=12.2 --parallel --skip_tests --enable_nccl --nccl_home /usr/local/cuda-12.2 --use_mpi --mpi_home=/usr/lib/x86_64-linux-gnu/
$ 3. Shard and export llama-70b model: CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh 4 -m meta-llama/Llama-2-7b-hf --output llama2-7b-dis2 --precision fp16 --execution_provider cuda
```

## Benchmark LLaMA-2

Here are some examples of how you can benchmark LLaMA-2.

### Variants

1. PyTorch without `torch.compile`, FP32
```
python3 -m models.llama.benchmark \
    --benchmark-type hf-pt-eager \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp32 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cpu \
    --auth
```

2. PyTorch with `torch.compile`, FP16
```
python3 -m models.llama.benchmark \
    --benchmark-type hf-pt-compile \
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
    --hf-ort-dir-path ./Llama-2-7b-hf-onnx/ \
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
    --hf-ort-dir-path ./llama2-7b-fp16/ \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp16 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cuda \
    --auth
```

5. ONNX Runtime, FP32, Microsoft custom export
```
python3 -m models.llama.benchmark \
    --benchmark-type ort-msft \
    --ort-model-path ./llama-2-onnx/7B_float32/ONNX/LlamaV2_7B_float32.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp32 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cpu
```

6. ONNX Runtime, FP16, Microsoft custom export
```
python3 -m models.llama.benchmark \
    --benchmark-type ort-msft \
    --ort-model-path ./llama-2-onnx/7B_float16/ONNX/LlamaV2_7B_float16.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp16 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cuda
```

7. ONNX Runtime, FP32, convert_to_onnx
```
python3 -m models.llama.benchmark \
    --benchmark-type ort-convert-to-onnx \
    --ort-model-path ./llama2-7b/Llama-2-7b-hf_decoder_merged_model_fp32.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp32 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cpu
```

8. ONNX Runtime, FP16, convert_to_onnx
```
python3 -m models.llama.benchmark \
    --benchmark-type ort-convert-to-onnx \
    --ort-model-path ./llama2-7b/Llama-2-7b-hf_decoder_merged_model_fp16.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp16 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cuda
```

You can profile a variant by adding the `--profile` flag and providing one batch size and sequence length combination.

### Benchmark All
You can use `benchmark_all.py` to benchmark across various options and automatically store the results in a CSV file. Here is an example.
```
python3 -m models.llama.benchmark_all \
    --hf-pt-eager \
    --hf-pt-compile \
    --hf-ort-dir-path ./llama2-7b-fp16/ \
    --ort-convert-to-onnx-model-path ./llama2-7b-fp16/Llama-2-7b-hf_decoder_merged_model_fp16.onnx \
    --ort-msft-model-path ./llama-2-onnx/7B_float16/ONNX/LlamaV2_7B_float16.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp16 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cuda
```
