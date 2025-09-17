# ONNX Runtime EP Context Samples

This repository provides **sample programs** (C++ and Python) that demonstrate how to use the **ONNX Runtime Execution Provider (EP) Context API** for NVIDIA TensorRT RTX Execution Provider (`NvTensorRTRTXExecutionProvider`).

## What is EP Context?

ONNX Runtime introduced the concept of **EP Context** to allow:

* **Pre-compilation of models** with a specific Execution Provider (EP) (e.g., NVIDIA TensorRT RTX).
* Faster **loading of compiled models** by reusing previously generated execution engines.
* Two storage modes:
  * **Embedded Mode** : Compiled binary is embedded inside the ONNX file.
  * **External Mode** : Compiled binary is stored as an external file alongside the ONNX.
* Two ways of loading the models
  * **Disk Load** : Load the model files from direct disk access.
  * **Buffer Load** : Loads the models form the memory buffer access for the models.

This makes it easier to deploy optimized models across multiple environments.

Compilation is currently only supported by execution providers that compile subgraphs.
More traditional execution providers like CUDA or CPU currently do not support this feature.

---

## Samples Included

### C++ sample.cpp (File-based Model)

* Loads an ONNX model from disk.
* Compiles it with the selected EP.
* Saves compiled ONNX file.
* Loads the compiled model and measures load times.

### C++ sample_buffer.cpp (Buffer-based Model with External Initializers)

* Loads model and weights directly into memory buffers.
* Registers **external initializers** (for `.onnx.data` files).
* Compiles the model to an in-memory buffer.
* Loads the compiled EP Context model from memory.

---

### Python sample.py (File based Model)

An equivalent Python sample is located at [../../python/EP_Context/](../../python/EP_Context/)

* Uses ONNX Runtime  **ModelCompiler API**.
* Demonstrates file-based compilation (with embedded/external modes).
* Measures load time for normal vs. compiled models.

---

## How to Run

### C++ sample.cpp

./sample.exe `<input model path my_model.onnx> <output model path model_ctx.onnx> <embed mode 0> <EP selection NvTensorRTRTXExecutionProvider> `

* `embed_mode`: `0` = external, `1` = embedded.
* `provider`: execution provider name (default: `NvTensorRTRTXExecutionProvider`).

This sample currently assumes fixed input and output shapes for the ONNX file used.

---

### C++ sample_buffer.cpp

./sample_buffer.exe `<input model path my_model.onnx> <input path to model data model.onnx.data> <output model path model_ctx.onnx> <external data file name model.onnx.data> <embed mode 0> <EP selection NvTensorRTRTXExecutionProvider>`

* `embed_mode`: `0` = external, `1` = embedded.
* `provider`: execution provider name (default: `NvTensorRTRTXExecutionProvider`).

---

### Python

See [../../python/40_ep-context/](../../python/40_ep-context/)

python sample_compile.py -i `<input model path my_model.onnx>` -o `<output context model path model_ctx.onnx>` -p `<EP selection NvTensorRTRTXExecutionProvider>` -e `<embed mode False>`

---

## Performance Improvement (RTX 5090)

|Model                                 | Normal Load Time (sec) | Compile Time (sec) | EP Context Load (sec) | EP Context + Cache (sec) |
|--------------------------------------|------------------------|--------------------|-----------------------|--------------------------|
| Deepseek qwen 14B - INT4             | 31.2312                | 34.9162            | 4.95345               | 3.7258                   |
| Llama-3.1-8B-Instruct - FP16         | 28.264                 | 30.8706            | 6.77561               | 6.0288                   |
| Stable Diffusion 3.5 - transformer   | 107.296                | 121.263            | 24.8112               | 9.07548                  |

---

## Build steps

### Prerequisites

- CMake 3.16 or higher
- Visual Studio 2019/2022 (Windows) or GCC/Clang (Linux)
- ONNX Runtime with NV TensorRT RTX support
- CUDA and NV TensorRT RTX (for NV TensorRT RTX execution provider)
- Build TRT RTX EP from this doc [Build TRT RTX EP](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html#build-from-source)
