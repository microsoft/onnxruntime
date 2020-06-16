# Transformer Model Optimization Tool Overview

ONNX Runtime automatically applies most optimizations while loading a transformer model. Some of the latest optimizations that have not yet been integrated into ONNX Runtime are available in this tool that tunes models for the best performance.

This tool can help in the following senarios:
* Model is exported by tf2onnx or keras2onnx, and ONNX Runtime does not have graph optimization for them right now.
* Convert model to use float16 to boost performance using mixed precision on GPUs with Tensor Cores (like V100 or T4).
* Model has inputs with dynamic axis, which blocks some optimizations to be applied in ONNX Runtime due to shape inference.
* Disable or enable some fusions to see its impact on performance or accuracy.

## Installation
First you need install onnxruntime or onnxruntime-gpu package for CPU or GPU inference. To use onnxruntime-gpu, it is required to install CUDA and cuDNN and add their bin directories to PATH environment variable.

This tool can be installed using pip as follows:
```console
pip install onnxruntime-tools
```

## Export a transformer model to ONNX
PyTorch could export model to ONNX. The tf2onnx and keras2onnx tools can be used to convert model that trained by Tensorflow.
Huggingface transformers has a [notebook](https://github.com/huggingface/transformers/blob/master/notebooks/04-onnx-export.ipynb) shows an example of exporting a pretrained model to ONNX. 
For Keras2onnx, please refer to its [example script](https://github.com/onnx/keras-onnx/blob/master/applications/nightly_build/test_transformers.py).
For tf2onnx, please refer to its [BERT tutorial](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/BertTutorial.ipynb).


## Model Optimizer

In your python code, you can use it like the following:

```python
from onnxruntime_tools import optimizer
optimized_model = optimizer.optimize_model("gpt2.onnx", model_type='gpt2', num_heads=12, hidden_size=768)
optimized_model.convert_model_float32_to_float16()
optimized_model.save_model_to_file("gpt2_fp16.onnx")
```

You can also use command line. Example of optimizing a BERT-large model to use mixed precision (float16):
```console
python -m onnxruntime_tools.optimizer_cli --input bert_large.onnx --output bert_large_fp16.onnx --num_heads 16 --hidden_size 1024 --float16
```

You can also download the latest script files from [here](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers/). Then run it like the following:
```console
python optimizer.py --input gpt2.onnx --output gpt2_opt.onnx --model_type gpt2
```

### Options

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

Right now, this tool assumes input model has 3 inputs for input IDs, segment IDs, and attention mask. A model with less or addtional inputs might not be fully optimized.

Most optimizations require exact match of a subgraph. Any layout change in subgraph might cause some optimization not working. Note that different versions of training or export tool might lead to different graph layouts.

Here is list of models from [Huggingface Transformers](https://github.com/huggingface/transformers/) that have been tested using this tool:
- **BertForSequenceClassification** as in [transformers example](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py) exported by PyTorch 1.2-1.4 using opset version 10 or 11.
- **BertForQuestionAnswering** as in [transformers example](https://github.com/huggingface/transformers/blob/master/examples/run_squad.py) exported by PyTorch 1.2-1.4 using opset version 10 or 11.
- **TFBertForSequenceClassification** as in [transformers example](https://github.com/huggingface/transformers/blob/master/examples/run_tf_glue.py) exported by keras2onnx installed from its master source.
- **TFBertForQuestionAnswering** exported by keras2onnx installed from its master source.
- **GPT2Model** exported by PyTorch 1.4 using opset version 10 or 11.
- **GPT2LMHeadModel** exported by PyTorch 1.4 using opset version 10 or 11.
If your model is not in the list, the optimized model might not work. You are welcome to update the scripts to support new models.

For GPT2 models, current optimization does not support past state (both inputs and outputs). You need disable it in transformers by setting enable_cache=False during exporting.

## Benchmark

There is a benchmark script that measure inference performance of OnnxRuntime, PyTorch or PyTorch+TorchScript on pretrained models of Huggingface Transformers.

The benchmark script requires PyTorch be installed.

Here is an example to run benchmark on pretrained model bert-base-cased on GPU.

```console
python -m onnxruntime_tools.transformers.benchmark -g -m bert-base-cased -o -v -b 0
python -m onnxruntime_tools.transformers.benchmark -g -m bert-base-cased -o
python -m onnxruntime_tools.transformers.benchmark -g -m bert-base-cased -e torch
python -m onnxruntime_tools.transformers.benchmark -g -m bert-base-cased -e torchscript
```
The first command will generate ONNX models (both before and after optimizations), but not run performance tests since batch size is 0. The other three commands will run performance test on each of three engines: OnnxRuntime, PyTorch and PyTorch+TorchScript.

If you remove -o parameter, optimizer script is not used in benchmark.

If your GPU (like V100 or T4) has TensorCore, you can append --fp16 to the above commands to enable mixed precision using float16.

If you want to benchmark on CPU, you can remove -g option in the commands.

Note that our current benchmark on GPT2 and DistilGPT2 models has disabled past state from inputs and outputs.

By default, ONNX model has only one input (input_ids). You can use -i parameter to test models with multiple inputs. For example, we can add "-i 3" to command line to test a bert model with 3 inputs (input_ids, token_type_ids and attention_mask). This option only supports OnnxRuntime right now.

## Model Verification

If your model has three inputs (like input_ids, token_type_ids and attention_mask), a script compare_bert_results.py can be used to do a quick verification. The tool will generate some fake input data, and compare results from both the original and optimized models. If outputs are all close, it is safe to use the optimized model.

Example of verifying models optimized for CPU:

```console
python -m onnxruntime_tools.transformers.compare_bert_results --baseline_model original_model.onnx --optimized_model optimized_model_cpu.onnx --batch_size 1 --sequence_length 128 --samples 100
```

For GPU, please append --use_gpu to the command.

## Performance Test

bert_perf_test.py can be used to check the model inference performance. Below are examples:

```console
python -m onnxruntime_tools.transformers.bert_perf_test --model optimized_model_cpu.onnx --batch_size 1 --sequence_length 128 --samples 100 --test_times 10 --inclusive
```

For GPU, please append --use_gpu to the command.

After test is finished, a file like perf_results_CPU_B1_S128_<date_time>.txt or perf_results_GPU_B1_S128_<date_time>.txt will be output to the model directory.

