## Nuphar Execution Provider (preview)

NUPHAR stands for Neural-network Unified Preprocessing Heterogeneous ARchitecture. As an execution provider in the ONNX Runtime, it is built on top of [TVM](https://github.com/dmlc/tvm) and [LLVM](https://llvm.org) to accelerate ONNX models by compiling nodes in subgraphs into optimized functions via JIT. It also provides JIT caching to save compilation time at runtime. 

This execution provider release is currently in preview. With the Nuphar execution provider, the ONNX Runtime delivers better inferencing performance on the same hardware compared to generic X64 CPU acceleration, especially for quantized recurrent neural networks. Various products at Microsoft have seen up to a 5x improvement in performance with no loss of accuracy, by running quantized LSTMs via the Nuphar execution provider in the ONNX Runtime.

### Build Nuphar execution provider
Developers can now tap into the power of Nuphar through ONNX Runtime to accelerate inferencing of ONNX models. Besides, the Nuphar execution provider also comes with a common ONNX to TVM lowering [library](../../onnxruntime/core/codegen), that could be reused by other execution providers to leverage TVM. Instructions to build the Nuphar execution provider from source is available [here](../../BUILD.md#nuphar).

### Using the Nuphar execution provider
#### C/C++
The Nuphar execution provider needs to be registered with ONNX Runtime to enable in the inference session. 
```
InferenceSession session_object{so};
session_object.RegisterExecutionProvider(std::make_unique<::onnxruntime::NupharExecutionProvider>());
status = session_object.Load(model_file_name);
```
The C API details are [here](../C_API.md#c-api).

### Python
You can use the Nuphar execution provider via the python wheel from the ONNX Runtime build. The Nuphar execution provider will be automatically prioritized over the default CPU execution providers, thus no need to separately register the execution provider. Python APIs details are [here](../python/api_summary.rst#api-summary).

### Using onnxruntime_perf_test/onnx_test_runner for performance and accuracy test
You can test your ONNX model's performance with [onnxruntime_perf_test](../../onnxruntime/test/perftest/README.md), or test accuracy with [onnx_test_runner](../../onnxruntime/test/onnx/README.txt). To run these tools with the Nuphar execution provider, please pass `-e nuphar` in command line options.

### Model conversion/quantization
You may use Python Script [model_editor.py](../../onnxruntime/core/providers/nuphar/scripts/model_editor.py) to turn LSTM/GRU/RNN ops to Scan ops for a given model, and then quantize MatMul ops into MatMulInteger ops.

We use dynamic per-row quantization for inputs of LSTM MatMul, so MatMul becomes three parts: quantization, MatMulInteger and dequantization. Weights for MatMulInteger are statically quantized per-column to int8. We have observed good speed-up and no loss of accuracy with this quantization scheme inside Scan for various LSTM models.

To convert models with LSTM/GRU/RNN ops to Scan ops:
```
python model_editor.py --input /path/to/input/model --output /path/to/output/model --mode to_scan
```

To quantize MatMul ops to MatMulInteger ops (use option --only_for_scan to only quantize MatMuls inside Scan):
```
python model_editor.py --input /path/to/input/model --output /path/to/output/model --mode to_imatmul --only_for_scan
```

As an experiment, you may test conversion and quantization on [the BiDAF model](https://github.com/onnx/models/tree/master/bidaf) from the ONNX model zoo. This model has 5 bidirectional LSTM ops, and long sequence lengths. Our test shows that the quantized model has comparable accuracy of F1 76.24, EM 68.08, vs. floating point model accuracy of F1 76.20, EM 68.11.

Speed-up in this model is ~20% on Intel Xeon E5-1620v4 (Note that AVX2 is required for Nuphar int8 GEMV performance), when comparing CPU execution provider with the floating point model with LSTM ops, vs. the Nuphar execution provider with quantized MatMulInteger inside Scan ops. Profile shows that most of the cost is in input projection outside of Scan ops, which uses MKL SGEMM. It's worth noting that MKL int8 GEMM is about the same speed as SGEMM in this model, so quantization of that SGEMM won't help performance. We are looking at ways to speedup int8 GEMM for better performance on quantized models.

### JIT caching
You may cache JIT binaries to reduce model loading time spent in JIT, using [create_shared.cmd](../../onnxruntime/core/providers/nuphar/scripts/create_shared.cmd) on Windows with Visual Studio 2017, or [create_shared.sh](../../onnxruntime/core/providers/nuphar/scripts/create_shared.sh) on Linux with gcc.

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

### Debugging
There are several [environment variables](../../onnxruntime/core/codegen/common/settings.h) to dump debug information during code generation, plus [some more environment variables](../../onnxruntime/core/providers/nuphar/common/nuphar_settings.h) to dump/control the Nuphar execution provider. You can set environment variables prior to inference to dump debug info to the console. To list some most useful ones:
* CODEGEN_DUMP_LOWER

    Dumps the lowered function from TVM.

    Set it to "verbose" to dump all nodes, or node op_type to dump specific nodes. You may use "concise" to dump just the op_type of nodes.

* CODEGEN_DUMP_MODULE

    Dumps compiled binary.

    Set it to "ll" to dumps LLVM bit code, "asm" to dumps assembly.

* CODEGEN_DUMP_SCHEDULE

    Dumps the schedule used in TVM nodes, like compute_root/compute_inline/compute_at.

    Set it to "verbose" to dump all nodes, or node op_type to dump specific nodes. You may use "concise" to dump just the op_type of nodes.

* NUPHAR_DUMP_FUSED_NODES

    Dumps fused nodes in each subgraph.

    Set it to "1" to dump partitions.

### Known issues
* ONNX shape inference dependency

    To save runtime JIT cost, Nuphar requires models to have shape inference information from ONNX after model is loaded. Some nodes in ONNX can generate dynamic output tensor shapes from input data value, i.e. ConstantOfShape, Tile, Slice in opset 10, Compress, etc. Those ops may block ONNX shape inference and make the model not runnable in Nuphar.

    We are working on solutions to this. A workaround would be manually add output tensor shapes in the model in graph.value_info field, if the shape in model can be determined by user even though the op output shape is dynamic by ONNX spec. For example, if you have Hardmax output casted to bool as Compress input condition, then the unknown dimension of the output of Compress is actually 1.

* Patches to TVM

    There are some changes/bug fixes in TVM for Nuphar to work properly. We are in the process of contributing them back to TVM, but for now patches are used. To build cleanly from scratch, please run following commands before running build.bat or build.sh:
```
git submodule sync
git submodule foreach --recursive git stash
git submodule foreach --recursive git clean -fd
git submodule update --init --recursive
```