# Contents
 - [LLaMA-2](#llama-2)
   - [Exporting LLaMa-2](#exporting-llama-2)
   - [Benchmarking LLaMa-2](#benchmark-llama-2)
 - [Mistral](#mistral)
   - [Exporting Mistral](#exporting-mistral)
   - [Optimizing and Quantizing Mistral](#optimizing-and-quantizing-mistral)
   - [Benchmarking Mistral](#benchmark-mistral)


# LLaMA-2

## Prerequisites

Please note the package versions needed for using LLaMA-2 in the `requirements.txt` file that fits your scenario.
- `requirements-cpu.txt`
  - For running LLaMA-2 on CPU
- `requirements-cuda.txt`
  - For running LLaMA-2 on CUDA
  - Note that `torch` with CUDA enabled is not installed automatically. This is because `torch` should be installed with the CUDA version used on your machine. Please visit [the PyTorch website](https://pytorch.org/get-started/locally/) to download the `torch` version that is used with the CUDA version installed on your machine and satisfies the requirement listed in the file.
- `requirements-quant.txt`
  - For running the SmoothQuant algorithm using [Intel's Neural Compressor](https://github.com/intel/neural-compressor)
- `requirements-70b-model.txt`
  - For running the LLaMA-2 70B model on multiple GPUs
- `requirements.txt`
  - Package versions needed in each of the above files

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

Note that this may produce two ONNX models with older Optimum versions. The above two options produce one ONNX model and installing Optimum from source will now produce one ONNX model.

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

Export Model with Different GPU Device Ids
```
# From source using first GPU:
$ CUDA_VISIBLE_DEVICES=0 python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --input ./Llama-2-7b-hf --output ./llama2-7b

# From wheel using second GPU:
$ CUDA_VISIBLE_DEVICES=1 python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --input ./Llama-2-7b-hf --output ./llama2-7b
```

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
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp32-gpu --precision fp32 --execution_provider cuda
```

Export for FP32 CPU
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp32-cpu --precision fp32 --execution_provider cpu

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp32-cpu --precision fp32 --execution_provider cpu
```

Export for FP16 CUDA (with MultiHeadAttention)
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp16 --precision fp16 --execution_provider cuda

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp16 --precision fp16 --execution_provider cuda
```

Export for FP16 CUDA (with GroupQueryAttention)
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp16 --precision fp16 --execution_provider cuda --use_gqa

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-fp16 --precision fp16 --execution_provider cuda --use_gqa
```

Note: GroupQueryAttention currently works with the FP16 CUDA and INT4 CUDA models, and it can provide faster inference than MultiHeadAttention, especially for large sequence lengths (e.g. 1024 or larger). For the best performance, you should pre-allocate the KV cache buffers to have size `(batch_size, num_heads, max_sequence_length, head_size)` so that the past KV and present KV caches share the same memory. You also need to bind them with ONNX Runtime's [IO binding](https://onnxruntime.ai/docs/api/python/api_summary.html#iobinding).

Here is an example of how you can bind directly to `torch.tensor` objects:
```
# Assumes all inputs and outputs to the model are pre-allocated with the correct shapes in GPU memory

# Bind inputs
for k, v in inputs.items():
    io_binding.bind_input(
        name=k,
        device_type="cuda",
        device_id=0,
        element_type=np.float16,
        shape=tuple(v.shape),
        buffer_ptr=v.data_ptr()
    )

# Bind outputs
for output in model.get_outputs():
    name = output.name
    if "present" in name:
        # Bind KV cache outputs to KV cache inputs
        v = inputs[name.replace("present", "past_key_values")]
        io_binding.bind_output(
            name=name,
            device_type="cuda",
            device_id=0,
            element_type=np.float16,
            shape=tuple(v.shape),
            buffer_ptr=v.data_ptr()
        )
    else:
        # Bind other outputs as actual outputs
        v = outputs[name]
        io_binding.bind_output(
            name=name,
            device_type="cuda",
            device_id=0,
            element_type=np.float16,
            shape=tuple(v.shape),
            buffer_ptr=v.data_ptr()
        )

io_binding.synchronize_inputs()
sess.run_with_iobinding(io_binding)
io_binding.synchronize_outputs()
```

Export for INT8 CPU (SmoothQuant)
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int8 --precision int8 --quantization_method smooth_quant --execution_provider cpu --no_merged

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int8 --precision int8 --quantization_method smooth_quant --execution_provider cpu --no_merged
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
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int4-gpu --precision int4 --quantization_method blockwise --execution_provider cuda --use_gqa

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int4-gpu --precision int4 --quantization_method blockwise --execution_provider cuda --use_gqa
```

Note: See the FP16 CUDA notes about GroupQueryAttention. The `--use_gqa` flag is optional.

Export for INT4 CPU
```
# From source:
$ python3 -m models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int4-cpu --precision int4 --quantization_method blockwise --execution_provider cpu

# From wheel:
$ python3 -m onnxruntime.transformers.models.llama.convert_to_onnx -m meta-llama/Llama-2-7b-hf --output llama2-7b-int4-cpu --precision int4 --quantization_method blockwise --execution_provider cpu
```

Export LLaMA-2 70B sharded model into 4 partitions
```
# From source:
# 1. Install necessary packages from requirements-70b-model.txt
$ pip install -r requirements-70b-model.txt

# 2. Build ONNX Runtime from source with NCCL enabled. Here is a sample command:
$ ./build.sh --config Release --use_cuda --cuda_home /usr/local/cuda-12.2 --cudnn_home /usr/local/cuda-12.2 --build_wheel --cuda_version=12.2 --parallel --skip_tests --enable_nccl --nccl_home /usr/local/cuda-12.2 --use_mpi --mpi_home=/usr/lib/x86_64-linux-gnu/

# 3. Shard and export the LLaMA-2 70B model. With FP16, you will need at least 140GB of GPU memory to load the model. Therefore, you will need at least 4 40GB A100 GPUs or 2 80GB A100 GPUs to shard the PyTorch model and export each shard to ONNX. Here is an example command:
$ CUDA_VISIBLE_DEVICES=0,1,2,3 bash convert_70b_model.sh 4 -m meta-llama/Llama-2-70b-hf --output llama2-70b-distributed --precision fp16 --execution_provider cuda --use_gqa
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

4. Optimum + ONNX Runtime, FP16, export via Optimum or convert_to_onnx
```
python3 -m models.llama.benchmark \
    --benchmark-type hf-ort \
    --hf-ort-dir-path ./Llama-2-7b-hf-onnx/ \
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

7. ONNX Runtime, FP32, convert_to_onnx, use 2nd GPU
```
CUDA_VISIBLE_DEVICES=1 python3 -m models.llama.benchmark \
    --benchmark-type ort-convert-to-onnx \
    --ort-model-path ./llama2-7b/rank_0_Llama-2-7b-hf_decoder_merged_model_fp32.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp32 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cpu
```

8. ONNX Runtime, FP16, convert_to_onnx, use 5th GPU
```
CUDA_VISIBLE_DEVICES=4 python3 -m models.llama.benchmark \
    --benchmark-type ort-convert-to-onnx \
    --ort-model-path ./llama2-7b/rank_0_Llama-2-7b-hf_decoder_merged_model_fp16.onnx \
    --model-name meta-llama/Llama-2-7b-hf \
    --precision fp16 \
    --batch-sizes "1 2" \
    --sequence-lengths "8 16" \
    --device cuda
```

9. ONNX Runtime, FP16, convert_to_onnx, LLaMA-2 70B shard to 4 GPUs
```
CUDA_VISIBLE_DEVICES=4,5,6,7 bash benchmark_70b_model.sh 4 \
    --benchmark-type ort-convert-to-onnx \
    --ort-model-path ./llama2-70b-dis/rank_{}_Llama-2-70b-hf_decoder_merged_model_fp16.onnx \
    --model-name meta-llama/Llama-2-70b-hf \
    --precision fp16 \
    --device cuda \
    --warmup-runs 5 \
    --num-runs 100
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
    --device cuda \
    --warmup-runs 5 \
    --num-runs 1000 \
    --timeout 60  # number of minutes before moving to the next benchmark
```

# Mistral

## Introduction

These tools for LLaMA-2 also allow the quantization and optimization of Mistral in ORT. 

## Exporting Mistral

There is currently one supported way to export Mistral to ONNX format:

### [Hugging Face Optimum](https://github.com/huggingface/optimum)


The following command will export Mistral in full precision:
```
python -m optimum.exporters.onnx -m mistralai/Mistral-7B-v0.1 --library-name transformers /path/to/model/directory
```

## Optimizing and Quantizing Mistral

To quantize Mistral to FP16 and apply fusion optimizations, you can run the following command:
```
python -m models.llama.convert_to_onnx -i /path/to/model/directory -o /path/to/optimized_model/directory -p fp16 --optimize_optimum -m mistralai/Mistral-7B-v0.1
```

## Benchmark Mistral
The benchmarking scripts in the LLaMA directory support Mistral benchmarking. To benchmark the ORT and HF versions respectively, you can run: 

```
python -m models.llama.benchmark \
    -bt ort-convert-to-onnx \
    -p fp16 \
    -m mistralai/Mistral-7B-v0.1 \
    --ort-model-path /path/to/model.onnx
python -m models.llama.benchmark \
    -bt hf-pt-eager \
    -p fp16 \
    -m mistralai/Mistral-7B-v0.1
```

