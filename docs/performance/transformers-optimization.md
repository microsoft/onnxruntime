---
title: Transformers optimizer 
description: Transformer model optimization tool to use with ONNX Runtime
parent: Performance
nav_order: 4
---
# Transformer Model Optimization Tool Overview
{: .no_toc }

While ONNX Runtime automatically applies most optimizations while loading transformer models, some of the latest optimizations that have not yet been integrated into ONNX Runtime. These additional optimizations can be applied using the [transformer optimization tool](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers) to tune models for the best performance. This optimization tool provides an offline capability to optimize transformer models in scenarios where ONNX Runtime does not apply the optimization at load time.

This tool can be helpful when:

* ONNX Runtime does not yet have transformer-specific graph optimization enabled
* The model can be converted to use float16 to boost performance using mixed precision on GPUs with Tensor Cores (like V100 or T4)
* The model has inputs with dynamic axis, which blocks some optimizations from being applied by ONNX Runtime due to shape inference.
* Experimenting with disabling or enabling some fusions to evaluate impact on performance or accuracy.

**Usage:**

1. [Install ONNX Runtime](#1-install-onnx-runtime)
2. [Convert the transformer model to ONNX](#2-convert-a-transformer-model-to-onnx)
3. [Run the model optimizer tool](#3-run-the-model-optimizer-tool)
4. [Benchmark and profile the model](#4-benchmark-and-profile-the-model)

## Supported models

For the list of models that have been tested with the optimizer, please refer to [this page](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/README.md#supported-models).

Most optimizations require exact match of a subgraph. Any layout change in the subgraph might cause some optimization to not work. Note that different versions of training or export tool might lead to different graph layouts. It is recommended to use the latest released version of PyTorch and Transformers.

### Limitations

* Due to the CUDA implementation of the Attention kernel in ONNX Runtime, the maximum number of attention heads is 1024.
* Normally, due to GPU memory constraints, the maximum supported sequence length is 4096 for Longformer and 1024 for other types of models.

---

## 1. Install ONNX Runtime

First you need install onnxruntime or onnxruntime-gpu package for CPU or GPU inference. To use onnxruntime-gpu, it is required to install CUDA and cuDNN and add their bin directories to PATH environment variable. See [Python installation instructions](./../install/index.md#python-installs).

## 2. Convert a transformer model to ONNX

To convert the transformer model to ONNX, use [torch.onnx](https://pytorch.org/docs/stable/onnx.html) or [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx).

* Huggingface transformers has a [notebook](https://github.com/huggingface/notebooks/blob/master/examples/onnx-export.ipynb) shows an example of exporting a pretrained model to ONNX.

* For tf2onnx, please refer to this [BERT tutorial](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/BertTutorial.ipynb).

### GPT-2 Model conversion

Converting the GPT-2 model from PyTorch to ONNX is not straightforward when past state is used. The tool [convert_to_onnx](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/gpt2/convert_to_onnx.py) can help.

You can use commands like the following to convert a pre-trained PyTorch GPT-2 model to ONNX for given precision (float32, float16):

```bash
python -m onnxruntime.transformers.models.gpt2.convert_to_onnx -m gpt2 --model_class GPT2LMHeadModel --output gpt2.onnx -p fp32

python -m onnxruntime.transformers.models.gpt2.convert_to_onnx -m distilgpt2 --model_class GPT2LMHeadModel --output distilgpt2.onnx -p fp16 --use_gpu --optimize_onnx --auto_mixed_precision
```

The tool will also verify whether the ONNX model and corresponding PyTorch model generate the same outputs given the same random inputs.

### Longformer Model conversion

Requirement: Linux OS (e.g. Ubuntu 18.04 or 20.04) and a Python environment with PyTorch 1.9.* like the following:

```bash
conda create -n longformer python=3.8
conda activate longformer
pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install onnx transformers==4.18.0 onnxruntime numpy
```

Next, build the source of torch extensions:

```bash
cd onnxruntime/python/tools/transformers/models/longformer/torch_extensions
python setup.py install
```

It will generate a PyTorch extension file like "build/lib.linux-x86_64-3.8/longformer_attention.cpython-38-x86_64-linux-gnu.so" under the directory.

Finally, convert longformer model to ONNX model like the following:

```bash
cd ..
python convert_to_onnx.py -m longformer-base-4096
```

The exported ONNX model can only run on GPU right now.

## 3. Run the model optimizer tool

For all Optimizer options, please see [Github](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/README.md#optimizer-options).

In your Python code, you can use the optimizer like the following:

```python
from onnxruntime.transformers import optimizer
optimized_model = optimizer.optimize_model("bert.onnx", model_type='bert', num_heads=12, hidden_size=768)
optimized_model.convert_float_to_float16()
optimized_model.save_model_to_file("bert_fp16.onnx")
```

You can also use command line. Example of optimizing a BERT-large model to use mixed precision (float16):

```console
python -m onnxruntime.transformers.optimizer --input bert_large.onnx --output bert_large_fp16.onnx --num_heads 16 --hidden_size 1024 --float16
```

You can also download the latest script files from [here](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/). Then run it like the following:

```console
python optimizer.py --input bert.onnx --output bert_opt.onnx --model_type bert
```

### BERT Model Verification

If your BERT model has three inputs (like input_ids, token_type_ids and attention_mask), a script compare_bert_results.py can be used to do a quick verification. The tool will generate some fake input data, and compare results from both the original and optimized models. If outputs are all close, it is safe to use the optimized model.

Example of verifying models optimized for CPU:

```console
python -m onnxruntime.transformers.compare_bert_results --baseline_model original_model.onnx --optimized_model optimized_model_cpu.onnx --batch_size 1 --sequence_length 128 --samples 100
```

For GPU, please append --use_gpu to the command.

## 4. Benchmark and profile the model

### Benchmarking

The bash script [run_benchmark.sh](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/run_benchmark.sh) can be used for running benchmarks. You can modify the bash script to choose your options (models, batch sizes, sequence lengths, target device, etc) before running.

The bash script will call benchmark.py script to measure inference performance of OnnxRuntime, PyTorch or PyTorch+TorchScript on pretrained models of Huggingface Transformers.

#### Benchmark.py

If you use run_benchmark.sh, you need not use benchmark.py directly. You can skip this section if you do not want to know the details.

Below is example to run benchmark.py on pretrained model bert-base-cased on GPU.

```console
python -m onnxruntime.transformers.benchmark -g -m bert-base-cased -o -v -b 0
python -m onnxruntime.transformers.benchmark -g -m bert-base-cased -o
python -m onnxruntime.transformers.benchmark -g -m bert-base-cased -e torch
python -m onnxruntime.transformers.benchmark -g -m bert-base-cased -e torchscript
```

The first command will generate ONNX models (both before and after optimizations), but not run performance tests since batch size is 0. The other three commands will run performance test on each of three engines: OnnxRuntime, PyTorch and PyTorch+TorchScript.

If you remove -o parameter, optimizer script is not used in benchmark.

If your GPU (like V100 or T4) has TensorCore, you can append `-p fp16` to the above commands to enable mixed precision. In some decoder-only(e.g GPT2) based generative models, you can enable [strict mode](../execution-providers/CUDA-ExecutionProvider.md#enable_skip_layer_norm_strict_mode) for SkipLayerNormalization Op on CUDA EP to achieve better accuracy. However, the performance will drop a bit.

If you want to benchmark on CPU, you can remove -g option in the commands.

Note that our current benchmark on GPT2 and DistilGPT2 models has disabled past state from inputs and outputs.

By default, ONNX model has only one input (input_ids). You can use -i parameter to test models with multiple inputs. For example, we can add "-i 3" to command line to test a bert model with 3 inputs (input_ids, token_type_ids and attention_mask). This option only supports OnnxRuntime right now.

### Performance Test

bert_perf_test.py can be used to check the BERT model inference performance. Below are examples:

```console
python -m onnxruntime.transformers.bert_perf_test --model optimized_model_cpu.onnx --batch_size 1 --sequence_length 128
```

For GPU, please append --use_gpu to the command.

After test is finished, a file like perf_results_CPU_B1_S128_<date_time>.txt or perf_results_GPU_B1_S128_<date_time>.txt will be output to the model directory.

### Profiling

profiler.py can be used to run profiling on a transformer model. It can help figure out the bottleneck of a model, and CPU time spent on a node or subgraph.

Examples commands:

```console
python -m onnxruntime.transformers.profiler --model bert.onnx --batch_size 8 --sequence_length 128 --samples 1000 --dummy_inputs bert --thread_num 8 --kernel_time_only
python -m onnxruntime.transformers.profiler --model gpt2.onnx --batch_size 1 --sequence_length 1 --past_sequence_length 128 --samples 1000 --dummy_inputs gpt2 --use_gpu
python -m onnxruntime.transformers.profiler --model longformer.onnx --batch_size 1 --sequence_length 4096 --global_length 8 --samples 1000 --dummy_inputs longformer --use_gpu
```

Result file like onnxruntime_profile__<date_time>.json will be output to current directory. Summary of nodes, top expensive nodes and results grouped by operator type will be printed to console.

Benchmark results can be found [here](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/README.md#benchmark-results).
