// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "core/mlas/lib/mlasi.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "benchmark/benchmark.h"

#include "bench_util.h"
#include "core/common/narrow.h"
#include "core/util/thread_utils.h"
#include "core/platform/env_var_utils.h"

using onnxruntime::narrow;

float* alloc_aligned_buffer(int D, int byte_aligned, void*& buffer) {
  buffer = (void*)(malloc(D * sizeof(float) + 64 * 2));

  //std::cout << "buffer: " << buffer << std::endl;
  //std::cout << "((uintptr_t)(buffer) & ~63): " << (void*)((uintptr_t)(buffer) & ~63) << std::endl;

  float* ptr = (float*)(((uintptr_t)(buffer) & ~63) + 64 + byte_aligned);
  switch (byte_aligned) {
    case 16:
    case 32:
      ORT_ENFORCE(((uintptr_t)(ptr) % byte_aligned == 0) && ((uintptr_t)(ptr) % (byte_aligned << 1) != 0));
      break;
    case 64:
      ORT_ENFORCE((uintptr_t)(ptr) % 64 == 0);
      break;
    default:
      throw std::invalid_argument("byte_aligned must be 16, 32, or 64!");
      break;
  }
  return ptr;
}

void COMPUTESOFTMAXINPLACE(benchmark::State& state) {
  const auto byte_aligned = narrow<int>(state.range(0));
  const auto N = narrow<int>(state.range(1));
  const auto D = narrow<int>(state.range(2));
  const auto threads = narrow<int>(state.range(3));

  if (N <= 0 || D <= 0 || threads <= 0) {
    throw std::invalid_argument("N, D, and Threads must be greater than 0!");
  }

  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = threads;
  tpo.auto_set_affinity = true;

  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
                                                 tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto data = RandomVectorUniform<float>(static_cast<size_t>(N * D), -1.0f, 1.0f);
  void* buffer = nullptr;
  float* input = alloc_aligned_buffer(N * D, byte_aligned, buffer);
  float* output = input;
  std::copy(data.begin(), data.end(), input); // Copy the data to the aligned memory

  // warm up run
  MlasComputeSoftmax(input, output, N, D, false, tp.get());

  for (auto _ : state) {
    MlasComputeSoftmax(input, output, N, D, false, tp.get());
  }

  free(buffer);
}

void REDUCEMAXIMUMF32KERNELAVX(benchmark::State& state) {
  const auto byte_aligned = narrow<int>(state.range(0));
  const auto D = narrow<int>(state.range(1));

  if (D <= 0) {
    throw std::invalid_argument("D must be greater than 0!");
  }

  auto data = RandomVectorUniform<float>(static_cast<size_t>(D), -1.0f, 1.0f);
  void* buffer = nullptr;
  float* input = alloc_aligned_buffer(D, byte_aligned, buffer);
  std::copy(data.begin(), data.end(), input); // Copy the data to the aligned memory

  // warm up run
  float Maximum = MlasReduceMaximumF32KernelAvx(input, D);

  for (auto _ : state) {
    Maximum = MlasReduceMaximumF32KernelAvx(input, D);
  }

  free(buffer);
  (void)Maximum;
}

void REDUCEMAXIMUMF32KERNELAVX512F(benchmark::State& state) {
  const auto byte_aligned = narrow<int>(state.range(0));
  const auto D = narrow<int>(state.range(1));

  if (D <= 0) {
    throw std::invalid_argument("D must be greater than 0!");
  }

  auto data = RandomVectorUniform<float>(static_cast<size_t>(D), -1.0f, 1.0f);
  void* buffer = nullptr;
  float* input = alloc_aligned_buffer(D, byte_aligned, buffer);
  std::copy(data.begin(), data.end(), input); // Copy the data to the aligned memory

  // warm up run
  float Maximum = MlasReduceMaximumF32KernelAvx512F(input, D);

  for (auto _ : state) {
    Maximum = MlasReduceMaximumF32KernelAvx512F(input, D);
  }

  free(buffer);
  (void)Maximum;
}

void COMPUTESUMEXPF32KERNELAVX512F(benchmark::State& state) {
  const auto byte_aligned = narrow<int>(state.range(0));
  const auto D = narrow<int>(state.range(1));

  if (D <= 0) {
    throw std::invalid_argument("D must be greater than 0!");
  }

  auto data = RandomVectorUniform<float>(static_cast<size_t>(D), -1.0f, 1.0f);

#if 0
  void* buffer = (void*)(malloc(D * sizeof(float) + 64 * 2));

  //std::cout << "buffer: " << buffer << std::endl;
  //std::cout << "((uintptr_t)(buffer) & ~63): " << (void*)((uintptr_t)(buffer) & ~63) << std::endl;

  float* input = (float*)(((uintptr_t)(buffer) & ~63) + 64 + byte_aligned);
  switch (byte_aligned) {
    case 16:
    case 32:
      ORT_ENFORCE(((uintptr_t)(input) % byte_aligned == 0) && ((uintptr_t)(input) % (byte_aligned << 1) != 0));
      break;
    case 64:
      ORT_ENFORCE((uintptr_t)(input) % 64 == 0);
      break;
    default:
      throw std::invalid_argument("byte_aligned must be 16, 32, or 64!");
      break;
  }
#endif

  void* buffer = nullptr;
  float* input = alloc_aligned_buffer(D, byte_aligned, buffer);
  std::copy(data.begin(), data.end(), input); // Copy the data to the aligned memory

  float Maximum = MlasReduceMaximumF32KernelAvx(input, D);
  float NegativeMaximum = -Maximum;

  // warm up run
  float Accumulation = MlasComputeSumExpF32KernelAvx512F(input, nullptr, D, &NegativeMaximum);

  for (auto _ : state) {
    Accumulation = MlasComputeSumExpF32KernelAvx512F(input, nullptr, D, &NegativeMaximum);
  }

  free(buffer);
  (void)Accumulation;
}

void COMPUTESOFTMAXOUTPUTF32KERNELAVX(benchmark::State& state) {
  const auto byte_aligned = narrow<int>(state.range(0));
  const auto D = narrow<int>(state.range(1));

  if (D <= 0) {
    throw std::invalid_argument("D must be greater than 0!");
  }

  auto data = RandomVectorUniform<float>(static_cast<size_t>(D), -1.0f, 1.0f);
  void* buffer = nullptr;
  float* input = alloc_aligned_buffer(D, byte_aligned, buffer);
  float* output = input;
  std::copy(data.begin(), data.end(), input); // Copy the data to the aligned memory

  float Maximum = MlasReduceMaximumF32KernelAvx(input, D);
  float NegativeMaximum = -Maximum;

  float Accumulation = MlasComputeSumExpF32KernelAvx512F(input, nullptr, D, &NegativeMaximum);

  float Parameters[] = { 1.0f / Accumulation };

  // warm up run
  MlasComputeSoftmaxOutputF32KernelAvx(output, D, Parameters);

  for (auto _ : state) {
    MlasComputeSoftmaxOutputF32KernelAvx(output, D, Parameters);
  }

  free(buffer);
}

static void ComputeSoftmaxInplaceArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"Byte Aligned", "N", "D", "Threads"});

  b->ArgsProduct({
      {16, 32, 64},     // Byte Aligned
      {208000, 240000}, // N
      {13, 15, 2000},   // D
      {1, 8},           // Threads
  });
}

BENCHMARK(COMPUTESOFTMAXINPLACE)->Apply(ComputeSoftmaxInplaceArgs)->UseRealTime();

#if 0
BENCHMARK(COMPUTESOFTMAXINPLACE)->ArgNames({"N", "D", "Threads"})
    ->Args({208000, 13, 1})
    ->Args({208000, 13, 8})
    ->Args({208000, 2000, 1})
    ->Args({208000, 2000, 8})
    ->Args({240000, 15, 1})
    ->Args({240000, 13, 8})
    ->Args({240000, 2000, 1})
    ->Args({240000, 2000, 8})
    ->UseRealTime();
#endif

BENCHMARK(REDUCEMAXIMUMF32KERNELAVX)->ArgNames({"Byte Aligned", "D"})
    ->ArgsProduct({
      {16, 32, 64},   // Byte Aligned
      {13, 15, 2000}, // D
      })
    ->UseRealTime();

BENCHMARK(REDUCEMAXIMUMF32KERNELAVX512F)->ArgNames({"Byte Aligned", "D"})
    ->ArgsProduct({
      {16, 32, 64},   // Byte Aligned
      {13, 15, 2000}, // D
      })
    ->UseRealTime();

BENCHMARK(COMPUTESUMEXPF32KERNELAVX512F)->ArgNames({"Byte Aligned", "D"})
    ->ArgsProduct({
      {16, 32, 64},   // Byte Aligned
      {13, 15, 2000}, // D
      })
    ->UseRealTime();

BENCHMARK(COMPUTESOFTMAXOUTPUTF32KERNELAVX)->ArgNames({"Byte Aligned", "D"})
    ->ArgsProduct({
      {16, 32, 64},   // Byte Aligned
      {13, 15, 2000}, // D
      })
    ->UseRealTime();