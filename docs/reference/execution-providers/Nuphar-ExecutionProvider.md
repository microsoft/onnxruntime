---
title: NUPHAR
parent: Execution Providers
grand_parent: Reference
---

# Nuphar Execution Provider
{: .no_toc }

NUPHAR stands for Neural-network Unified Preprocessing Heterogeneous Architecture. As an execution provider in the ONNX Runtime, it is built on top of [TVM](https://github.com/dmlc/tvm) and [LLVM](https://llvm.org) to accelerate ONNX models by compiling nodes in subgraphs into optimized functions via JIT. It also provides JIT caching to save compilation time at runtime. 

Developers can tap into the power of Nuphar through ONNX Runtime to accelerate inferencing of ONNX models. The Nuphar execution provider comes with a common ONNX to TVM lowering [library](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/core/codegen) that can potentially be reused by other execution providers to leverage TVM. With the Nuphar execution provider, the ONNX Runtime delivers better inferencing performance on the same hardware compared to generic X64 CPU acceleration, especially for quantized recurrent neural networks. Various products at Microsoft have seen up to a 5x improvement in performance with no loss of accuracy, by running quantized LSTMs via the Nuphar execution provider in the ONNX Runtime.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Build
For build instructions, please see the [BUILD page](../../how-to/build/eps.md#nuphar).

## Usage
### C/C++

```c++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions sf;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nuphar(sf, /*allow_unaligned_buffers*/ 1, ""));
Ort::Session session(env, model_path, sf);
```

### Python

```python
import onnxruntime as ort

model_path = '<path to model>'
providers = ['NupharExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)
```

## Configuration Options

When there are conflicts of environment variables running Nuphar in multiple processes, user can specify settings string when creating the Nuphar execution provider. The string comprises of comma separated key:value pairs. Keys should be lower cased environment variable names as shown above, and separated from corresponding values with colon. For example, the equivalent string of setting environment variables of NUPHAR_CACHE_PATH/NUPHAR_CACHE_MODEL_CHECKSUM would be "nuphar_cache_path:<path_to_cache>, nuphar_cache_model_checksum:<model_file_checksum>".

* Using in C/C++

Settings string could be specified when creating execution provider to specify JIT cache path, as well as model checksum:

```c++
OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_Nuphar(session_options, 1, "nuphar_cache_path:/path/to/cache, nuphar_cache_model_checksum:<model_checksum>"));
```

* Using in C#

Settings string could be specified when creating session options:

```csharp
SessionOptions.MakeSessionOptionWithNupharProvider("nuphar_cache_path:/path/to/cache, nuphar_cache_model_checksum:<model_checksum>")
```

* Using in Python

Settings string can be set as an execution provider-specific option. Here's an example in Python to set cache path and model checksum:

```python
nuphar_settings = 'nuphar_cache_path:{}, nuphar_cache_model_checksum:{}'.format(cache_dir, model_checksum)
providers = [('NupharExecutionProvider', {'nuphar_settings': nuphar_settings}), 'CPUExecutionProvider']
sess = onnxruntime.InferenceSession(model_path, providers=providers)
```


## Performance Tuning

You can test your ONNX model's performance with [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/perftest/README.md), or test accuracy with [onnx_test_runner](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/test/onnx). To run these tools with the Nuphar execution provider, please pass `-e nuphar` in command line options.

Please note that Nuphar uses TVM thread pool and parallel schedule for multi-thread inference performance. When building with OpenMP or MKLML, TVM thread pool would use gomp or iomp as its implementation; otherwise, TVM creates its own thread pool. Because of this, the current default parallel schedule policy is:
- Default to on for USE_OPENMP or USE_MKLML. User can use OMP_NUM_THREADS/MKL_NUM_THREADS to control TVM thread pool, as well as TVM_NUM_THREADS
- Default to off for none of above. User can use TVM_NUM_THREADS to control TVM thread pool.

This choice is to ensure to get ideal performance with the different build options. When build with USE_OPENMP or USE_MKLML, users would have to avoid thread confliction from OpenMP or MKL with their inference invocations anyway, so parallel schedule is enable to leverage existing thread pool. When not building with gomp or iomp, TVM thread pool is turned off to avoid confliction with user threads. If needed, user can set env or settings with [NUPHAR_PARALLEL_MIN_WORKLOADS](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/core/providers/nuphar/common/nuphar_settings.cc#L61) to 0 to disable parallel schedule, or to some non-zero value to enable parallel schedule. The non-zero value indicates the minimal number of elements being computed per thread when parallel schedule would be turned on.

### Model Conversion and Quantization
You may use Python script [model_editor.py](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/core/providers/nuphar/scripts/model_editor.py) to turn LSTM/GRU/RNN ops to Scan ops for a given model, and then use [model_quantizer.py](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/core/providers/nuphar/scripts/model_quantizer.py) to quantize MatMul ops into MatMulInteger ops.

We use dynamic per-row quantization for inputs of LSTM MatMul, so MatMul becomes three parts: quantization, MatMulInteger and dequantization. Weights for MatMulInteger are statically quantized per-column to int8. We have observed good speed-up and no loss of accuracy with this quantization scheme inside Scan for various LSTM models.

To convert models with LSTM/GRU/RNN ops to Scan ops:
```
python model_editor.py --input /path/to/input/model --output /path/to/output/model --mode to_scan
```

To quantize MatMul ops to MatMulInteger ops (use option --only_for_scan to only quantize MatMuls inside Scan):
```
python model_quantizer.py --input /path/to/input/model --output /path/to/output/model --only_for_scan
```

As an experiment, you may test conversion and quantization on [the BiDAF model](https://github.com/onnx/models/tree/master/text/machine_comprehension/bidirectional_attention_flow) from the ONNX model zoo. This model has 5 bidirectional LSTM ops, and long sequence lengths. Our test shows that the quantized model has comparable accuracy of F1 76.24, EM 68.08, vs. floating point model accuracy of F1 76.20, EM 68.11.

Speed-up in this model is ~20% on Intel Xeon E5-1620v4 (Note that AVX2 is required for Nuphar int8 GEMV performance), when comparing CPU execution provider with the floating point model with LSTM ops, vs. the Nuphar execution provider with quantized MatMulInteger inside Scan ops. Profile shows that most of the cost is in input projection outside of Scan ops, which uses MKL SGEMM. It's worth noting that MKL int8 GEMM is about the same speed as SGEMM in this model, so quantization of SGEMMs outside of Scan won't help performance. We are looking at ways to speedup int8 GEMM for better performance on quantized models.

### JIT caching

Windows
```
REM You need to have Visual Studio 2017 for compile and link. Optionally, you can save model checksum to the output dll with FCIV tool from https://support.microsoft.com/en-us/help/841290
set NUPHAR_CACHE_PATH=\path\to\jit\cache
REM Then run Nuphar inference from either onnx_test_runner or onnxruntime_perf_test, or whatever inference using C++ or Python
REM JIT object files would be saved to \path\to\jit\cache\<NUPHAR_CACHE_VERSION>
create_shared.cmd \path\to\jit\cache\NUPHAR_CACHE_VERSION [optional_model_file_for_checksum] [optional_output_dll_name]
REM If checksum is embedded in dll, set NUPHAR_CACHE_MODEL_CHECKSUM to FCIV output for the model to inference to pass checksum verification at runtime
REM Checksum verification failure will cause Nuphar to fallback to JIT instead of loading binary from cache
REM Run Nuphar inference again with cached JIT dll
```

Linux
```
# You need to have GCC of the same version Nuphar is built with, for compile and link. Optionally, you can save model checksum to jit.so with md5sum
export NUPHAR_CACHE_PATH=/path/to/jit/cache
# Then run Nuphar inference from either onnx_test_runner or onnxruntime_perf_test, or whatever inference using C++ or Python
# JIT object files would be saved to /path/to/jit/cache/<NUPHAR_CACHE_VERSION>
create_shared.sh -c /path/to/jit/cache/NUPHAR_CACHE_VERSION [-m optional_model_file_for_checksum] [-o optional_output_so_name]
# If checksum is embedded in dll, set NUPHAR_CACHE_MODEL_CHECKSUM to md5sum output for the model to inference to pass checksum verification at runtime
# Checksum verification failure will cause Nuphar to fallback to JIT instead of loading binary from cache
# run Nuphar inference again with cached JIT dll
```
## Samples
* [Sample notebook for NUPHAR execution provider](https://github.com/microsoft/onnxruntime/blob/master/docs/python/inference/notebooks/onnxruntime-nuphar-tutorial.ipynb)

## Debugging

### NGEMM

NGEMM (Nuphar GEMM) is an optimized low-precision GEMM implementation based on compiler techniques.
Please refer to our paper for more details of NGEMM: ["NGEMM: Optimizing GEMM for Deep Learning via Compiler-based Techniques"](https://arxiv.org/abs/1910.00178).

#### NGEMM Tiling / Permutation Configuration

NGEMM has default tiling parameters, but users can overwrite them through environment variables:

* NUPHAR_IGEMM_TILE_M / NUPHAR_IGEMM_TILE_N / NUPHAR_IGEMM_TILE_K

    These 3 parameters are the tiling sizes for the corresponding dimensions of GEMM ([M x K] x [K x N]).

    Setting them to different values will generate GEMM with different tiling sizes.

* NUPHAR_IGEMM_PERMUTE

    This environment variable is to control the loop permutation in GEMM.

    The default is to not apply any loop permutation. Other options are "inner/outer/all",referring to apply permutations to only inner tile loops / only outer loops / both inner and outer loops, respectively.

    There are several [environment variables](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/codegen/common/settings.h) to dump debug information during code generation, plus [some more environment variables](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/nuphar/common/nuphar_settings.h) to dump/control the Nuphar execution provider. You can set environment variables prior to inference to dump debug info to the console. To list some most useful ones:

* CODEGEN_DUMP_LOWER

    Dumps the lowered function from TVM.

    Set it to "verbose" to dump all nodes, or node op_type to dump specific nodes. You may use "concise" to dump just the op_type of nodes.

* CODEGEN_DUMP_MODULE

    Dumps compiled binary.

    Set it to "ll" to dumps LLVM bit code, "asm" to dumps assembly.

* CODEGEN_DUMP_SCHEDULE

    Dumps the schedule used in TVM nodes, like compute_root/compute_inline/compute_at.

    Set it to "verbose" to dump all nodes, or node op_type to dump specific nodes. You may use "concise" to dump just the op_type of nodes.

* NUPHAR_DUMP_PARTITION

    Dumps nodes in each partition.

    Set it to "1" to dump partitions.


## Known issues
* ONNX shape inference dependency

To save runtime JIT cost, Nuphar requires models to have shape inference information from ONNX after model is loaded. Some nodes in ONNX can generate dynamic output tensor shapes from input data value, i.e. ConstantOfShape, Tile, Slice in opset 10, Compress, etc. Those ops may block ONNX shape inference and make the part of graph after such nodes not runnable in Nuphar.

User may use Python script [symbolic_shape_infer.py](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/symbolic_shape_infer.py) to run symbolic shape inference in ONNX model. This script adds output tensor shapes in the model in graph.value_info field, by doing symbolic dimension computation using sympy when there are Shape ops in model. Besides, running symbolic shape inference on ONNX model would make the graph more readable. Note that when using [model_editor.py](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/core/providers/nuphar/scripts/model_editor.py) to convert models with LSTM/GRU/RNN to Scan, the resulting model may have incomplete shape inference. Running symbolic_shape_infer.py is needed to get the Scan ops in the model to run in Nuphar. Besides, please note that quantization should be the last step, after verified accuracy and performance of the edited floating point model.

In addition, user may also manually add shapes to graph.value_info using [onnx.helper.make_tensor_value_info](https://github.com/onnx/onnx/blob/v1.5.0/onnx/helper.py#L290) with model specific knowledge. For example, if you have Hardmax output casted to bool as Compress input condition, then the unknown dimension of the output of Compress is actually 1.

* Performance benchmark

Current Nuphar's speed-up in quantized RNNs is optimized for AVX2, when running in single thread and batch size is 1. To help understand RNN performance in different configurations, please use Python script [rnn_benchmark.py](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/providers/nuphar/scripts/rnn_benchmark.py). For older X64 CPUs that do not support AVX2, quantized model may have worse performance than non-quantized ones.

* Patches to TVM

There are some changes/bug fixes in TVM for Nuphar to work properly. We are in the process of contributing them back to TVM, but for now patches are used in [our forked TVM](https://github.com/microsoft/onnxruntime-tvm). To build cleanly from scratch, please run following commands before running build.bat or build.sh:

```bash
git submodule sync
git submodule foreach --recursive git stash
git submodule foreach --recursive git clean -fd
git submodule update --init --recursive
```
