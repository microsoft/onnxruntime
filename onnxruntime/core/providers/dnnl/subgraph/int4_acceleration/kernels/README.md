# BesTLA
BesTLA is a lightweight, header-only acceleration library for high-performance GEMM and related computations on Intel platform. Inspired by Cutlass, it provides high-level template class abstractions for various elements required for computation, and allows flexible kernel construction through template combinations to meet specific needs, maximizing the reuse of existing template classes. Users can also develop custom template classes to expand BesTLA’s computational capabilities. BesTLA includes several different types of template classes, specifically:

- `Launcher`: Schedules computation-related template classes, allowing users to specify their own computation-related template classes, including GemmCore, Prologue, and Epilogue.
- `Parallel`: Specifies data splitting strategy for task distribution among different cores. BesTLA’s default Parallel template class adopts an L2-cache-fusion concept, i.e., each core tries to temporarily store the data it processes in its L2-cache during each round of gemm-tile computation.
- `GemmCore`: A computation-related template class that provides a micro-kernel for performing a tile gemm computation with a specific ISA. It is the most important template class in BesTLA. Currently, GemmCore supports the following ISAs:
   - AVX2
   - AVX_VNNI
   - AVX512F
   - AVX512_VNNI
   - AMX_BF16
   - AMX_INT8
   - AVX512_FP16
- `Prologue`: A computation-related template class that preprocesses (such as data type conversion/padding) input data to meet GemmCore’s input data requirements.
- `Epilogue`: A computation-related template class that post-processes (such as eltwiseop-fusion) the results of gemm-core computations to expand BesTLA’s application scenarios.
BesTLA supports users to configure thread libraries for multi-core parallelism (e.g. openMP), greatly facilitating user integrate BesTLA into their own projects. BesTLA also supports specifying the number of computing-threads at runtime, making the allocation of computing resources more flexible.

# Highlights 
## Weight-only 
BesTLA provides weight-only linear computational capabilities for LLM inference. We provide a series of Prologues for quantize/compress/serialize/deserialize fp32 weights in different ways. Specifically, the weight-only-quantization configs we support are given in the table below: 

| Weight dtype           |   Compute dtype    |    Scale dtype    |    algo    |
| ---------------------- | :----------------: | :---------------: | :--------: |
| INT8                   | INT8 / BF16 / FP32 |    BF16 / FP32    | sym / asym |
| INT4 (CLIP, FULLRANGE) | INT8 / BF16 / FP32 |    BF16 / FP32    | sym / asym |
| FP8 (E4M3, E5M2)       |    BF16 / FP32     | FP32 / FP8 (E8M0) |    sym     |
| FP4 (E2M1)             |    BF16 / FP32     |    BF16 / FP32    |    sym     |
| NF4                    |    BF16 / FP32     |    BF16 / FP32    |    sym     |

Config description of the table:
| Config        | Description                                         |
| ------------- | --------------------------------------------------- |
| Weight dtype  | Data type of quantized weight                       |
| Compute dtype | Data type of BesTLA internal Gemm computation       |
| Scale dtype   | Data type of scales                                 |
| alg           | Quantization algorithm to use(symmetric/asymmetric) |


## Postop-fusion 
BesTLA provides assembly-level postop-fusion through epilogue to minimize the overhead caused by data movement. Specifically, we support the following postop-fusions:

- GELU
- SWISH
- RELU
- EXP
- TANH
## Compilation Requirements and Usage
Compile: 

- GCC version >=8.5.0 
- CMake version >=3.5

Usage:
```cmake
add_subdirectory(bestla)
target_link_libraries("${YOUR_PROJECT}" bestla::bestla)
```
