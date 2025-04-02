#include "onnxruntime_config.h"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wdeprecated"

// _deps/eigen-src/unsupported/Eigen/CXX11/../../../Eigen/src/Core/arch/NEON/PacketMath.h:1671:9:
// error: ‘void* memcpy(void*, const void*, size_t)’ copying an object of non-trivial type ‘Eigen::internal::Packet4c’
//   {aka ‘struct Eigen::internal::eigen_packet_wrapper<int, 2>’} from an array of ‘const int8_t’
//   {aka ‘const signed char’} [-Werror=class-memaccess]
#ifdef HAS_CLASS_MEMACCESS
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif

#elif defined(_MSC_VER)
// build\windows\debug\external\eigen3\unsupported\eigen\cxx11\src/Tensor/Tensor.h(76):
// warning C4554: '&': check operator precedence for possible error; use parentheses to clarify precedence

// unsupported\eigen\cxx11\src\Tensor\TensorUInt128.h(150,0): Warning C4245: 'initializing': conversion from '__int64'
// to 'uint64_t', signed/unsigned mismatch
#pragma warning(push)
#pragma warning(disable : 4554)
#pragma warning(disable : 4245)
#pragma warning(disable : 4127)
#endif

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

#include <unsupported/Eigen/CXX11/ThreadPool>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorDeviceThreadPool.h>
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <benchmark/benchmark.h>

static void BM_EigenBroadCast(benchmark::State& state) {
  Eigen::ThreadPool threadpool(4);
  Eigen::ThreadPoolDevice device(&threadpool, 4);
  const std::vector<size_t> dims_vec{1, 64, 75, 75};
  const size_t C = 64;
  // calculate sample_size (per individual channel)
  const int sample_size = 5625;
  using T = float;
  Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> input_tensor(C, 1);
  input_tensor.setRandom();
  Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> output_tensor(C, sample_size);
  for (auto _ : state) {
    std::array<int, 2> bcast{1, sample_size};
    output_tensor.device(device) = input_tensor.broadcast(bcast);
  }
}

BENCHMARK(BM_EigenBroadCast)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

static void BM_EigenBroadCast_SingleThread(benchmark::State& state) {
  const std::vector<size_t> dims_vec{1, 64, 75, 75};
  const size_t C = 64;
  // calculate sample_size (per individual channel)
  const int sample_size = 5625;
  using T = float;
  Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> input_tensor(C, 1);
  input_tensor.setRandom();
  Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> output_tensor(C, sample_size);
  for (auto _ : state) {
    std::array<int, 2> bcast{1, sample_size};
    output_tensor = input_tensor.broadcast(bcast);
  }
}

BENCHMARK(BM_EigenBroadCast_SingleThread)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);
