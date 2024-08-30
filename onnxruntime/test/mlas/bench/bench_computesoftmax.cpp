// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/mlas/lib/mlasi.h"
#include "core/util/thread_utils.h"
#include "test/mlas/bench/bench_util.h"

using onnxruntime::narrow;

struct RestrictAlignedPtr {
  float* ptr;               // Aligned pointer within the underlying buffer
  void* underlying_buffer;  // Underlying buffer (including extra space for alignment)
};

// Return a RestrictAlignedPtr where the ptr is aligned to byte_aligned, but not to byte_aligned * 2
RestrictAlignedPtr restrict_aligned_alloc(int D, int byte_aligned) {
  if (byte_aligned <= 0 || (byte_aligned & (byte_aligned - 1)) != 0) {
    throw std::invalid_argument("Alignment must be a power of 2");
  }

  const int byte_alignedx2 = byte_aligned << 1;

  void* buffer = malloc(D * sizeof(float) + byte_alignedx2 * 2);
  if (buffer == nullptr) {
    ORT_THROW_EX(std::bad_alloc);
  }

  uintptr_t address = reinterpret_cast<uintptr_t>(buffer);
  uintptr_t aligned_address = ((address + byte_alignedx2 - 1) & ~(byte_alignedx2 - 1)) + byte_aligned;
  ORT_ENFORCE((aligned_address % byte_aligned == 0) && (aligned_address % byte_alignedx2 != 0));
  float* aligned_ptr = reinterpret_cast<float*>(aligned_address);

  return {aligned_ptr, buffer};
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
          &onnxruntime::Env::Default(), tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto data = RandomVectorUniform<float>(static_cast<size_t>(N * D), -1.0f, 1.0f);
  RestrictAlignedPtr ptr = restrict_aligned_alloc(N * D, byte_aligned);
  float* input = ptr.ptr;
  float* output = input;
  std::copy(data.begin(), data.end(), input);  // Copy the data to the aligned memory

  // warming up run
  MlasComputeSoftmax(input, output, N, D, false, false, tp.get());

  for (auto _ : state) {
    MlasComputeSoftmax(input, output, N, D, false, false, tp.get());
  }

  free(ptr.underlying_buffer);
}

#if defined(MLAS_TARGET_AMD64)

void REDUCEMAXIMUMF32KERNELAVX(benchmark::State& state) {
  const auto byte_aligned = narrow<int>(state.range(0));
  const auto D = narrow<int>(state.range(1));

  if (D <= 0) {
    throw std::invalid_argument("D must be greater than 0!");
  }

  auto data = RandomVectorUniform<float>(static_cast<size_t>(D), -1.0f, 1.0f);
  RestrictAlignedPtr ptr = restrict_aligned_alloc(D, byte_aligned);
  float* input = ptr.ptr;
  std::copy(data.begin(), data.end(), input);  // Copy the data to the aligned memory

  // warming up run
  float Maximum = MlasReduceMaximumF32KernelAvx(input, D);

  for (auto _ : state) {
    Maximum = MlasReduceMaximumF32KernelAvx(input, D);
  }

  free(ptr.underlying_buffer);
  (void)Maximum;
}

void REDUCEMAXIMUMF32KERNELAVX512F(benchmark::State& state) {
  const auto byte_aligned = narrow<int>(state.range(0));
  const auto D = narrow<int>(state.range(1));

  if (D <= 0) {
    throw std::invalid_argument("D must be greater than 0!");
  }

  auto data = RandomVectorUniform<float>(static_cast<size_t>(D), -1.0f, 1.0f);
  RestrictAlignedPtr ptr = restrict_aligned_alloc(D, byte_aligned);
  float* input = ptr.ptr;
  std::copy(data.begin(), data.end(), input);  // Copy the data to the aligned memory

  // warming up run
  float Maximum = MlasReduceMaximumF32KernelAvx512F(input, D);

  for (auto _ : state) {
    Maximum = MlasReduceMaximumF32KernelAvx512F(input, D);
  }

  free(ptr.underlying_buffer);
  (void)Maximum;
}

void COMPUTESUMEXPF32KERNELAVX512F(benchmark::State& state) {
  const auto byte_aligned = narrow<int>(state.range(0));
  const auto D = narrow<int>(state.range(1));

  if (D <= 0) {
    throw std::invalid_argument("D must be greater than 0!");
  }

  auto data = RandomVectorUniform<float>(static_cast<size_t>(D), -1.0f, 1.0f);
  RestrictAlignedPtr ptr = restrict_aligned_alloc(D, byte_aligned);
  float* input = ptr.ptr;
  float* output = input;
  std::copy(data.begin(), data.end(), input);  // Copy the data to the aligned memory

  float Maximum = MlasReduceMaximumF32KernelAvx(input, D);
  float NegativeMaximum = -Maximum;

  // warming up run
  float Accumulation = MlasComputeSumExpF32KernelAvx512F(input, output, D, &NegativeMaximum);

  for (auto _ : state) {
    Accumulation = MlasComputeSumExpF32KernelAvx512F(input, output, D, &NegativeMaximum);
  }

  free(ptr.underlying_buffer);
  (void)Accumulation;
}

void COMPUTESOFTMAXOUTPUTF32KERNELAVX(benchmark::State& state) {
  const auto byte_aligned = narrow<int>(state.range(0));
  const auto D = narrow<int>(state.range(1));

  if (D <= 0) {
    throw std::invalid_argument("D must be greater than 0!");
  }

  auto data = RandomVectorUniform<float>(static_cast<size_t>(D), -1.0f, 1.0f);
  RestrictAlignedPtr ptr = restrict_aligned_alloc(D, byte_aligned);
  float* input = ptr.ptr;
  float* output = input;
  std::copy(data.begin(), data.end(), input);  // Copy the data to the aligned memory

  float Maximum = MlasReduceMaximumF32KernelAvx(input, D);
  float NegativeMaximum = -Maximum;

  float Accumulation = MlasComputeSumExpF32KernelAvx512F(input, output, D, &NegativeMaximum);

  float Parameters[] = {1.0f / Accumulation};

  // warming up run
  MlasComputeSoftmaxOutputF32KernelAvx(output, D, Parameters);

  for (auto _ : state) {
    MlasComputeSoftmaxOutputF32KernelAvx(output, D, Parameters);
  }

  free(ptr.underlying_buffer);
}

#endif  // defined(MLAS_TARGET_AMD64)

static void ComputeSoftmaxInplaceArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"ByteAligned", "N", "D", "Threads"});
  for (int threads : {1, 8}) {
    for (int byte_aligned : {64}) {  // MLAS_DEFAULT_PREFERRED_BUFFER_ALIGNMENT is 64
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

#if defined(MLAS_TARGET_AMD64)

BENCHMARK(REDUCEMAXIMUMF32KERNELAVX)
    ->ArgNames({"ByteAligned", "D"})
    ->ArgsProduct({
        {4, 8, 16, 32, 64, 128},                     // ByteAligned
        {3, 4, 5, 7, 9, 11, 13, 15, 16, 500, 2000},  // D
    })
    ->UseRealTime();

BENCHMARK(REDUCEMAXIMUMF32KERNELAVX512F)
    ->ArgNames({"ByteAligned", "D"})
    ->ArgsProduct({
        {4, 8, 16, 32, 64, 128},                     // ByteAligned
        {3, 4, 5, 7, 9, 11, 13, 15, 16, 500, 2000},  // D
    })
    ->UseRealTime();

BENCHMARK(COMPUTESUMEXPF32KERNELAVX512F)
    ->ArgNames({"ByteAligned", "D"})
    ->ArgsProduct({
        {4, 8, 16, 32, 64, 128},                 // ByteAligned
        {3, 4, 5, 7, 9, 11, 13, 15, 500, 2000},  // D
    })
    ->UseRealTime();

BENCHMARK(COMPUTESOFTMAXOUTPUTF32KERNELAVX)
    ->ArgNames({"ByteAligned", "D"})
    ->ArgsProduct({
        {4, 8, 16, 32, 64, 128},                     // ByteAligned
        {3, 4, 5, 7, 9, 11, 13, 15, 16, 500, 2000},  // D
    })
    ->UseRealTime();

#endif  // defined(MLAS_TARGET_AMD64)
