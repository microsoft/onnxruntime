#pragma once
#include <cstdio>
#include "core/common/make_string.h"

#if !defined(DUMP_TENSOR_LEVEL)
#define DUMP_TENSOR_LEVEL 0
#endif

#define DUMP_CPU_TENSOR_LEVEL DUMP_TENSOR_LEVEL

// For CPU tensor dumping.
#if DUMP_CPU_TENSOR_LEVEL > 0
#define DUMP_CPU_TENSOR_INIT() onnxruntime::contrib::CpuTensorConsoleDumper cpu_dumper
#define DUMP_CPU_TENSOR(...) cpu_dumper.Print(__VA_ARGS__)
#define DUMP_CPU_STRING_INIT() DUMP_CPU_TENSOR_INIT()
#define DUMP_CPU_STRING(...) cpu_dumper.Print(::onnxruntime::MakeString(__VA_ARGS__))
#else
#define DUMP_CPU_TENSOR_INIT(...)
#define DUMP_CPU_TENSOR(...)
#define DUMP_CPU_STRING_INIT(...)
#define DUMP_CPU_STRING(...)
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
#define DUMP_STRING_INIT() DUMP_TENSOR_INIT()
#define DUMP_STRING(...) dumper.Print(::onnxruntime::MakeString(__VA_ARGS__))
#else
#define DUMP_TENSOR_INIT(...)
#define DUMP_TENSOR(...)
#define DUMP_STRING_INIT(...)
#define DUMP_STRING(...)
#endif

#if DUMP_TENSOR_LEVEL > 1
#define DUMP_TENSOR_D(...) dumper.Print(__VA_ARGS__)
#else
#define DUMP_TENSOR_D(...)
#endif

#if (defined(__GNUC__) || defined(__clang__)) && (DUMP_TENSOR_LEVEL > 0)
#define DUMP_PRINTF(fmt, ...) \
  std::printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define DUMP_PRINTF(fmt, ...) \
  do {                        \
  } while (0)
#endif
