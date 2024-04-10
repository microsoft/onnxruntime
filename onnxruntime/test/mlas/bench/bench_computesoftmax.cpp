// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bench_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/util/thread_utils.h"

using onnxruntime::narrow;

float* alloc_aligned_buffer(int D, int byte_aligned, void*& buffer) {
  constexpr int max_byte_aligned = 128;

  buffer = malloc(D * sizeof(float) + max_byte_aligned * 2);

  float* ptr = reinterpret_cast<float*>(
    ((reinterpret_cast<uintptr_t>(buffer) + (max_byte_aligned - 1)) & ~(max_byte_aligned - 1)) + byte_aligned
  );

  switch (byte_aligned) {
    case 4:
    case 8:
    case 16:
    case 32:
    case 64:
      ORT_ENFORCE(((uintptr_t)(ptr) % byte_aligned == 0) && ((uintptr_t)(ptr) % (byte_aligned << 1) != 0));
      break;
    case max_byte_aligned:
      ORT_ENFORCE((uintptr_t)(ptr) % max_byte_aligned == 0);
      break;
    default:
      throw std::invalid_argument("byte_aligned must be 4, 8, 16, 32, 64, or 128!");
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
    onnxruntime::concurrency::CreateThreadPool(
      &onnxruntime::Env::Default(), tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP
    )
  );

  auto data = RandomVectorUniform<float>(static_cast<size_t>(N * D), -1.0f, 1.0f);
  void* buffer = nullptr;
  float* input = alloc_aligned_buffer(N * D, byte_aligned, buffer);
  float* output = input;
  std::copy(data.begin(), data.end(), input); // Copy the data to the aligned memory

  // warming up run
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

  // warming up run
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

  // warming up run
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

  void* buffer = nullptr;
  float* input = alloc_aligned_buffer(D, byte_aligned, buffer);
  float* output = input;
  std::copy(data.begin(), data.end(), input); // Copy the data to the aligned memory

  float Maximum = MlasReduceMaximumF32KernelAvx(input, D);
  float NegativeMaximum = -Maximum;

  // warming up run
  float Accumulation = MlasComputeSumExpF32KernelAvx512F(input, output, D, &NegativeMaximum);

  for (auto _ : state) {
    Accumulation = MlasComputeSumExpF32KernelAvx512F(input, output, D, &NegativeMaximum);
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

  float Accumulation = MlasComputeSumExpF32KernelAvx512F(input, output, D, &NegativeMaximum);

  float Parameters[] = { 1.0f / Accumulation };

  // warming up run
  MlasComputeSoftmaxOutputF32KernelAvx(output, D, Parameters);

  for (auto _ : state) {
    MlasComputeSoftmaxOutputF32KernelAvx(output, D, Parameters);
  }

  free(buffer);
}

static void ComputeSoftmaxInplaceArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"ByteAligned", "N", "D", "Threads"});
  for (int threads : {1, 8}) {
    for (int byte_aligned : {4, 8, 16, 32, 64, 128}) {
      b->Args({byte_aligned, 16000, 4, threads});
      b->Args({byte_aligned, 16000, 500, threads});
      b->Args({byte_aligned, 48000, 3, threads});
      b->Args({byte_aligned, 48000, 2000, threads});
      b->Args({byte_aligned, 80000, 5, threads});
      b->Args({byte_aligned, 80000, 2000, threads});
      b->Args({byte_aligned, 112000, 7, threads});
      b->Args({byte_aligned, 112000, 2000, threads});
      b->Args({byte_aligned, 144000, 9, threads});
      b->Args({byte_aligned, 144000, 2000, threads});
      b->Args({byte_aligned, 176000, 11, threads});
      b->Args({byte_aligned, 176000, 2000, threads});
      b->Args({byte_aligned, 208000, 13, threads});
      b->Args({byte_aligned, 208000, 2000, threads});
      b->Args({byte_aligned, 240000, 15, threads});
      b->Args({byte_aligned, 240000, 2000, threads});
    }
  }
}

BENCHMARK(COMPUTESOFTMAXINPLACE)->Apply(ComputeSoftmaxInplaceArgs)->UseRealTime();

BENCHMARK(REDUCEMAXIMUMF32KERNELAVX)->ArgNames({"ByteAligned", "D"})
  ->ArgsProduct({
    {4, 8, 16, 32, 64, 128},                    // ByteAligned
    {3, 4, 5, 7, 9, 11, 13, 15, 16, 500, 2000}, // D
    })
  ->UseRealTime();

BENCHMARK(REDUCEMAXIMUMF32KERNELAVX512F)->ArgNames({"ByteAligned", "D"})
  ->ArgsProduct({
    {4, 8, 16, 32, 64, 128},                    // ByteAligned
    {3, 4, 5, 7, 9, 11, 13, 15, 16, 500, 2000}, // D
    })
  ->UseRealTime();

BENCHMARK(COMPUTESUMEXPF32KERNELAVX512F)->ArgNames({"ByteAligned", "D"})
  ->ArgsProduct({
    {4, 8, 16, 32, 64, 128},   // ByteAligned
    {3, 4, 5, 7, 9, 11, 13, 15, 500, 2000}, // D
    })
  ->UseRealTime();

BENCHMARK(COMPUTESOFTMAXOUTPUTF32KERNELAVX)->ArgNames({"ByteAligned", "D"})
  ->ArgsProduct({
    {4, 8, 16, 32, 64, 128},   // ByteAligned
    {3, 4, 5, 7, 9, 11, 13, 15, 16, 500, 2000}, // D
    })
  ->UseRealTime();