#pragma once

// #define DEBUG_GENERATION 1  // uncomment it for debugging generation (like beam search etc)

#ifdef DEBUG_GENERATION
#define DUMP_TENSOR_LEVEL 2
#else
#define DUMP_TENSOR_LEVEL 0  // change it to 1 or 2 if want to enable dumping for code not in generation.
#endif

#define DUMP_CPU_TENSOR_LEVEL DUMP_TENSOR_LEVEL

// For CPU tensor dumping.
#if DUMP_CPU_TENSOR_LEVEL > 0
#define DUMP_CPU_TENSOR_INIT() onnxruntime::contrib::CpuTensorConsoleDumper cpu_dumper
#define DUMP_CPU_TENSOR(...) cpu_dumper.Print(__VA_ARGS__)
#else
#define DUMP_CPU_TENSOR_INIT()
#define DUMP_CPU_TENSOR(...)
#endif

#if DUMP_CPU_TENSOR_LEVEL > 1
#define DUMP_CPU_TENSOR_D(...) cpu_dumper.Print(__VA_ARGS__)
#else
#define DUMP_CPU_TENSOR_D(...)
#endif

// For GPU tensor dumping.
#if DUMP_TENSOR_LEVEL > 0
#define DUMP_TENSOR_INIT() onnxruntime::contrib::cuda::CudaTensorConsoleDumper dumper
#define DUMP_TENSOR(...) dumper.Print(__VA_ARGS__)
#else
#define DUMP_TENSOR_INIT()
#define DUMP_TENSOR(...)
#endif

#if DUMP_TENSOR_LEVEL > 1
#define DUMP_TENSOR_D(...) dumper.Print(__VA_ARGS__)
#else
#define DUMP_TENSOR_D(...)
#endif
