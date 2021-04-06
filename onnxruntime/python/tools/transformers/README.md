# Transformer Model Optimization Tool Overview

ONNX Runtime automatically applies most optimizations while loading a transformer model. Some of the latest optimizations that have not yet been integrated into ONNX Runtime are available in this tool that tunes models for the best performance.

This tool can help in the following senarios:
* Model is exported by tf2onnx or keras2onnx, and ONNX Runtime does not have graph optimization for them right now.
* Convert model to use float16 to boost performance using mixed precision on GPUs with Tensor Cores (like V100 or T4).
* Model has inputs with dynamic axis, which blocks some optimizations to be applied in ONNX Runtime due to shape inference.
* Disable or enable some fusions to see its impact on performance or accuracy.

## Installation

First you need install onnxruntime or onnxruntime-gpu package for CPU or GPU inference. To use onnxruntime-gpu, it is required to install CUDA and cuDNN and add their bin directories to PATH environment variable.

## Limitations

Due to CUDA implementation of Attention kernel, maximum number of attention heads is 1024. Normally, maximum supported sequence length is 4096 for Longformer and 1024 for other types of models.

## Export a transformer model to ONNX

PyTorch could export model to ONNX. The tf2onnx and keras2onnx tools can be used to convert model that trained by Tensorflow.
Huggingface transformers has a [notebook](https://github.com/huggingface/transformers/blob/master/notebooks/04-onnx-export.ipynb) shows an example of exporting a pretrained model to ONNX.
For Keras2onnx, please refer to its [example script](https://github.com/onnx/keras-onnx/blob/master/applications/nightly_build/test_transformers.py).
For tf2onnx, please refer to its [BERT tutorial](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/BertTutorial.ipynb).

### GPT-2 Model conversion

Converting GPT-2 model from PyTorch to ONNX is not straightforward when past state is used. We add a tool [convert_to_onnx](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/convert_to_onnx.py) to help you.

You can use commands like the following to convert a pre-trained PyTorch GPT-2 model to ONNX for given precision (float32, float16 or int8):
```
python -m onnxruntime.transformers.convert_to_onnx -m gpt2 --model_class GPT2LMHeadModel --output gpt2.onnx -p fp32
python -m onnxruntime.transformers.convert_to_onnx -m distilgpt2 --model_class GPT2LMHeadModel --output distilgpt2.onnx -p fp16 --use_gpu --optimize_onnx
python -m onnxruntime.transformers.convert_to_onnx -m [path_to_gpt2_pytorch_model_directory] --output quantized.onnx -p fp32 --optimize_onnx
```

The tool will also verify whether the ONNX model and corresponding PyTorch model generate same outputs given same random inputs.

### Longformer Model conversion

Requirement: Linux OS (For example Ubuntu 18.04 or 20.04) and a python environment like the following:
```
conda create -n longformer python=3.6
conda activate longformer
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install onnx transformers onnxruntime
```
Next, get the source of [torch extensions for Longformer exporting](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers/torch_extensions), and run the following:
```
python setup.py install
```
It will generate file like "build/lib.linux-x86_64-3.6/longformer_attention.cpython-36m-x86_64-linux-gnu.so" under the directory.

Finally, use [convert_longformer_to_onnx](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/longformer/convert_longformer_to_onnx.py) to convert to ONNX model like the following:
```
python convert_longformer_to_onnx.py -m longformer-base-4096
```

The exported ONNX model can only run in GPU right now.

## Model Optimizer

In your python code, you can use the optimizer like the following:

```python
from onnxruntime.transformers import optimizer
optimized_model = optimizer.optimize_model("gpt2.onnx", model_type='gpt2', num_heads=12, hidden_size=768)
optimized_model.convert_model_float32_to_float16()
optimized_model.save_model_to_file("gpt2_fp16.onnx")
```

You can also use command line. Example of optimizing a BERT-large model to use mixed precision (float16):
```console
python -m onnxruntime.transformers.optimizer --input bert_large.onnx --output bert_large_fp16.onnx --num_heads 16 --hidden_size 1024 --float16
```

You can also download the latest script files from [here](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers/). Then run it like the following:
```console
python optimizer.py --input gpt2.onnx --output gpt2_opt.onnx --model_type gpt2
```

### Optimizer Options

See below for description of some options of optimizer.py:

- **input**: input model path
- **output**: output model path
- **model_type**: (*defaul: bert*)
    There are 4 model types: *bert* (exported by PyTorch), *gpt2* (exported by PyTorch), and *bert_tf* (BERT exported by tf2onnx), *bert_keras* (BERT exported by keras2onnx) respectively.
- **num_heads**: (*default: 12*)
    Number of attention heads. BERT-base and BERT-large has 12 and 16 respectively.
- **hidden_size**: (*default: 768*)
    BERT-base and BERT-large has 768 and 1024 hidden nodes respectively.
- **input_int32**: (*optional*)
    Exported model ususally uses int64 tensor as input. If this flag is specified, int32 tensors will be used as input, and it could avoid un-necessary Cast nodes and get better performance.
- **float16**: (*optional*)
    By default, model uses float32 in computation. If this flag is specified, half-precision float will be used. This option is recommended for NVidia GPU with Tensor Core like V100 and T4. For older GPUs, float32 is likely faster.
-  **use_gpu**: (*optional*)
    When opt_level > 1, please set this flag for GPU inference.
- **opt_level**: (*optional*)
    Set a proper graph optimization level of OnnxRuntime: 0 - disable all (default), 1 - basic, 2 - extended, 99 - all. If the value is positive, OnnxRuntime will be used to optimize graph first.
- **verbose**: (*optional*)
    Print verbose information when this flag is specified.

### Supported Models

Here is a list of PyTorch models from [Huggingface Transformers](https://github.com/huggingface/transformers/) that have been tested using the optimizer:
- BERT
- DistilBERT
- DistilGPT2
- RoBERTa
- ALBERT
- GPT-2 (**GPT2Model**, **GPT2LMHeadModel**)

For Tensorflow model, we only tested BERT model so far.

Most optimizations require exact match of a subgraph. Any layout change in subgraph might cause some optimization not working. Note that different versions of training or export tool might lead to different graph layouts. It is recommended to use latest released version of PyTorch and Transformers.

If your model is not in the list, it might only be partial optimized or not optimized at all.


## Benchmark
There is a bash script [run_benchmark.sh](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/run_benchmark.sh) for running benchmark. You can modify the bash script to choose your options (like models to test, batch sizes, sequence lengths, target device etc) before running.

The bash script will call benchmark.py script to measure inference performance of OnnxRuntime, PyTorch or PyTorch+TorchScript on pretrained models of Huggingface Transformers.

### Benchmark Results on V100

In the following benchmark results, ONNX Runtime uses optimizer for model optimization, and IO binding is enabled.

We tested on Tesla V100-PCIE-16GB GPU (CPU is Intel Xeon(R) E5-2690 v4) for different batch size (**b**) and sequence length (**s**). Below result is average latency of per inference in miliseconds.

#### bert-base-uncased (BertModel)

The model has 12 layers and 768 hidden, with input_ids as input.

| engine      | version | precision | b | s=8  | s=16 | s=32 | s=64 | s=128 | s=256 | s=512 |
|-------------|---------|-----------|---|------|------|------|------|-------|-------|-------|
| torchscript | 1.5.1   | fp32      | 1 | 7.92 | 8.78 | 8.91 | 9.18 | 9.56  | 9.39  | 12.83 |
| onnxruntime | 1.4.0   | fp32      | 1 | 1.38 | 1.42 | 1.67 | 2.15 | 3.11  | 5.37  | 10.74 |
| onnxruntime | 1.4.0   | fp16      | 1 | 1.30 | 1.29 | 1.31 | 1.33 | 1.45  | 1.95  | 3.36  |
| onnxruntime | 1.4.0   | fp32      | 4 | 1.51 | 1.93 | 2.98 | 5.01 | 9.13  | 17.95 | 38.15 |
| onnxruntime | 1.4.0   | fp16      | 4 | 1.27 | 1.35 | 1.43 | 1.83 | 2.66  | 4.40  | 9.76  |

[run_benchmark.sh](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/run_benchmark.sh) is used to get the results.

#### gpt2 (GPT2LMHeadModel)

The model has 12 layers and 768 hidden, with input_ids, position_ids, attention_mask and past state as inputs.

| engine      | version | precision | b | s=4  | s=8 | s=32 | s=128 |
|-------------|---------|-----------|---|------|------|------|------|
| torchscript | 1.5.1   | fp32      | 1 | 5.80 | 5.77 | 5.82 | 5.78 |
| onnxruntime | 1.4.0   | fp32      | 1 | 1.42 | 1.42 | 1.43 | 1.47 |
| onnxruntime | 1.4.0   | fp16      | 1 | 1.54 | 1.54 | 1.58 | 1.64 |
| onnxruntime | 1.4.0   | fp32      | 8 | 1.83 | 1.84 | 1.90 | 2.13 |
| onnxruntime | 1.4.0   | fp16      | 8 | 1.74 | 1.75 | 1.81 | 2.09 |
| onnxruntime | 1.4.0   | fp32      | 32 | 2.19 | 2.21 | 2.45 | 3.34 |
| onnxruntime | 1.4.0   | fp16      | 32 | 1.66 | 1.71 | 1.85 | 2.73 |
| onnxruntime | 1.4.0   | fp32      | 128 | 4.15 | 4.37 | 5.15 | 8.61 |
| onnxruntime | 1.4.0   | fp16      | 128 | 2.47 | 2.58 | 3.26 | 6.16 |

Since past state is used, sequence length in input_ids is 1. For example, s=4 means the past sequence length is 4 and the total sequence length is 5.

[benchmark_gpt2.py](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/benchmark_gpt2.py) is used to get the results like the following commands:

```console
python -m onnxruntime.transformers.benchmark_gpt2 --use_gpu -m gpt2 -o -v -b 1 8 32 128 -s 4 8 32 128 -p fp32
python -m onnxruntime.transformers.benchmark_gpt2 --use_gpu -m gpt2 -o -v -b 1 8 32 128 -s 4 8 32 128 -p fp16
```

### Benchmark.py

If you use run_benchmark.sh, you need not use benchmark.py directly. You can skip this section if you do not want to know the details.

Below is example to runing benchmark.py on pretrained model bert-base-cased on GPU.

```console
python -m onnxruntime.transformers.benchmark -g -m bert-base-cased -o -v -b 0
python -m onnxruntime.transformers.benchmark -g -m bert-base-cased -o
python -m onnxruntime.transformers.benchmark -g -m bert-base-cased -e torch
python -m onnxruntime.transformers.benchmark -g -m bert-base-cased -e torchscript
```
The first command will generate ONNX models (both before and after optimizations), but not run performance tests since batch size is 0. The other three commands will run performance test on each of three engines: OnnxRuntime, PyTorch and PyTorch+TorchScript.

If you remove -o parameter, optimizer script is not used in benchmark.

If your GPU (like V100 or T4) has TensorCore, you can append `-p fp16` to the above commands to enable mixed precision.

If you want to benchmark on CPU, you can remove -g option in the commands.

Note that our current benchmark on GPT2 and DistilGPT2 models has disabled past state from inputs and outputs.

By default, ONNX model has only one input (input_ids). You can use -i parameter to test models with multiple inputs. For example, we can add "-i 3" to command line to test a bert model with 3 inputs (input_ids, token_type_ids and attention_mask). This option only supports OnnxRuntime right now.

## BERT Model Verification

If your BERT model has three inputs (like input_ids, token_type_ids and attention_mask), a script compare_bert_results.py can be used to do a quick verification. The tool will generate some fake input data, and compare results from both the original and optimized models. If outputs are all close, it is safe to use the optimized model.

Example of verifying models optimized for CPU:

```console
python -m onnxruntime.transformers.compare_bert_results --baseline_model original_model.onnx --optimized_model optimized_model_cpu.onnx --batch_size 1 --sequence_length 128 --samples 100
```

For GPU, please append --use_gpu to the command.

## Performance Test

bert_perf_test.py can be used to check the BERT model inference performance. Below are examples:

```console
python -m onnxruntime.transformers.bert_perf_test --model optimized_model_cpu.onnx --batch_size 1 --sequence_length 128
```

For GPU, please append --use_gpu to the command.

After test is finished, a file like perf_results_CPU_B1_S128_<date_time>.txt or perf_results_GPU_B1_S128_<date_time>.txt will be output to the model directory.

## Profiling

profiler.py can be used to run profiling on a transformer model. It can help figure out the bottleneck of a model, and CPU time spent on a node or subgraph.

Examples commands:

```console
python -m onnxruntime.transformers.profiler --model bert.onnx --batch_size 8 --sequence_length 128 --samples 1000 --dummy_inputs bert --thread_num 8 --kernel_time_only
python -m onnxruntime.transformers.profiler --model gpt2.onnx --batch_size 1 --sequence_length 1 --past_sequence_length 128 --samples 1000 --dummy_inputs gpt2 --use_gpu
python -m onnxruntime.transformers.profiler --model longformer.onnx --batch_size 1 --sequence_length 4096 --global_length 8 --samples 1000 --dummy_inputs longformer --use_gpu
```

Result file like onnxruntime_profile__<date_time>.json will be output to current directory. Summary of nodes, top expensive nodes and results grouped by operator type will be printed to console.
