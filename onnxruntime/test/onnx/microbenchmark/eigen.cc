#if defined(__GNUC__)
#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

#include <unsupported/Eigen/CXX11/ThreadPool>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorDeviceThreadPool.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
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
