#include "common.h"

#include <benchmark/benchmark.h>
#include <core/util/math_cpuonly.h>
#include <core/mlas/lib/mlasi.h>
#include <iostream>

using namespace onnxruntime;

static void BM_QuantizeLinearBase(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  uint8_t* output = (uint8_t*)aligned_alloc(sizeof(uint8_t) * batch_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);

  for (auto _ : state) {
    MlasQuantizeLinearU8Kernel(data, output, batch_size, 2.f / 512.f, 1);
  }
  aligned_free(data);
  aligned_free(output);
}

BENCHMARK(BM_QuantizeLinearBase)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(64)
    ->Arg(80)
    ->Arg(100)
    ->Arg(128)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(98304)
    ->Arg(1572864);

static void BM_QuantizeLinearAVX512(benchmark::State& state) {
  unsigned Cpuid1[4];
  unsigned Cpuid7[4];
#if defined(_WIN32)
  __cpuid((int*)Cpuid1, 1);
  __cpuidex((int*)Cpuid7, 7, 0);
#else
  __cpuid(1, Cpuid1[0], Cpuid1[1], Cpuid1[2], Cpuid1[3]);
  __cpuid_count(7, 0, Cpuid7[0], Cpuid7[1], Cpuid7[2], Cpuid7[3]);
#endif

  uint64_t xcr0 = MlasReadExtendedControlRegister(_XCR_XFEATURE_ENABLED_MASK);

  if ((Cpuid1[2] & 0x18000000) == 0x18000000 &&
      ((Cpuid7[1] & 0x10000) != 0) &&
      ((xcr0 & 0xE0) == 0xE0)) {
    const size_t batch_size = static_cast<size_t>(state.range(0));
    uint8_t* output = (uint8_t*)aligned_alloc(sizeof(uint8_t) * batch_size, 64);
    float* data = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);

    for (auto _ : state) {
      MlasQuantizeLinearU8KernelAvx512F(data, output, batch_size, 2.f / 512.f, 1);
    }
    aligned_free(data);
    aligned_free(output);
  } else {
    std::cerr << "CPU doesn't support AVX512" << std::endl;
  }
}

BENCHMARK(BM_QuantizeLinearAVX512)
    ->UseRealTime()
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(64)
    ->Arg(80)
    ->Arg(100)
    ->Arg(128)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(98304)
    ->Arg(1572864);
