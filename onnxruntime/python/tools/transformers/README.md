## **Usage instructions:** [Transformers Tool Usage Documentation](https://onnxruntime.ai/docs/performance/transformers-optimization.html)

## Supported Models
The following models from [Huggingface Transformers](https://github.com/huggingface/transformers/) have been tested using the optimizer:

PyTorch:
- BERT
- DistilBERT
- DistilGPT2
- RoBERTa
- ALBERT
- GPT-2 (**GPT2Model**, **GPT2LMHeadModel**)

Tensorflow:
- BERT

Most optimizations require exact match of a subgraph. Any layout change in subgraph might cause some optimization not working. Note that different versions of training or export tool might lead to different graph layouts. It is recommended to use latest released version of PyTorch and Transformers.

Models not in the list may only be partially optimized or not optimized at all.

## Optimizer Options

- **input**: input model path
- **output**: output model path
- **model_type**: (*defaul: bert*)
    There are 4 model types: *bert* (exported by PyTorch), *gpt2* (exported by PyTorch), and *bert_tf* (BERT exported by tf2onnx), *bert_keras* (BERT exported by keras2onnx) respectively.
- **num_heads**: (*default: 12*)
    Number of attention heads. BERT-base and BERT-large has 12 and 16 respectively.
- **hidden_size**: (*default: 768*)
    BERT-base and BERT-large has 768 and 1024 hidden nodes respectively.
- **input_int32**: (*optional*)
    Exported model usually uses int64 tensor as input. If this flag is specified, int32 tensors will be used as input, and it could avoid un-necessary Cast nodes and get better performance.
- **float16**: (*optional*)
    By default, model uses float32 in computation. If this flag is specified, half-precision float will be used. This option is recommended for NVidia GPU with Tensor Core like V100 and T4. For older GPUs, float32 is likely faster.
-  **use_gpu**: (*optional*)
    When opt_level > 1, please set this flag for GPU inference.
- **opt_level**: (*optional*)
    Set a proper graph optimization level of OnnxRuntime: 0 - disable all (default), 1 - basic, 2 - extended, 99 - all. If the value is positive, OnnxRuntime will be used to optimize graph first.
- **verbose**: (*optional*)
    Print verbose information when this flag is specified.


## Benchmark Results

These benchmarks were executed on V100 machines using the optimizer with [IO binding](https://onnxruntime.ai/docs/performance/tune-performance/cuda-performance.html) enabled.

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

[run_benchmark.sh](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/run_benchmark.sh) is used to get the results.

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

[benchmark_gpt2.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/gpt2/benchmark_gpt2.py) is used to get the results like the following commands:

```console
python -m onnxruntime.transformers.models.gpt2.benchmark_gpt2 --use_gpu -m gpt2 -o -v -b 1 8 32 128 -s 4 8 32 128 -p fp32
python -m onnxruntime.transformers.models.gpt2.benchmark_gpt2 --use_gpu -m gpt2 -o -v -b 1 8 32 128 -s 4 8 32 128 -p fp16
```

