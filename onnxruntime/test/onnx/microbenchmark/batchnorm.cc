#include "core/platform/threadpool.h"
#include "core/common/eigen_common_wrapper.h"
#include "core/util/thread_utils.h"
#include <benchmark/benchmark.h>

using namespace onnxruntime;
#if 0
static void BM_BatchNormEigenTensor(benchmark::State& state) {
  OrtThreadPoolParams param;
  param.auto_set_affinity = true;
  param.thread_pool_size = 0;
  std::unique_ptr<concurrency::ThreadPool> tp =
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), param, concurrency::ThreadPoolType::INTRA_OP);
  const size_t batch_size = state.range(0);

  const std::vector<size_t> dims_vec{batch_size, 64, 75, 75};
  const size_t N = dims_vec[0];
  const size_t C = dims_vec[1];  // assume NCHW as per the spec

  // calculate sample_size (per individual channel)
  size_t sample_size = 1;
  for (size_t i = 2; i < dims_vec.size(); ++i) {
    sample_size *= dims_vec[i];
  }
  using T = float;
  Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> input_tensor(N * C, sample_size);
  input_tensor.setRandom();
  Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> scale_arr(C);
  scale_arr.setRandom();
  Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> mean_arr(C);
  mean_arr.setRandom();
  Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> bias_arr(C);
  bias_arr.setRandom();
  Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> var_arr(C);
  var_arr.setRandom();
  Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> output_tensor(N * C, sample_size);

  float epsilon_ = 1e-5f;
  for (auto _ : state) {
    Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> eps(C);
    eps.setConstant(epsilon_);
    Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> inv_std(C);
    Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
    batch_by_one.set(0, C);
    Eigen::IndexList<int, int> bcast;
    bcast.set(0, N);
    bcast.set(1, sample_size);
    inv_std.device(tp->Device()) = (scale_arr / (var_arr + eps).sqrt());
    output_tensor.device(tp->Device()) = input_tensor * inv_std.reshape(batch_by_one).broadcast(bcast) +
                                         (bias_arr - mean_arr * inv_std).eval().reshape(batch_by_one).broadcast(bcast);
  }
}

BENCHMARK(BM_BatchNormEigenTensor)->Arg(1)->Arg(16)->Arg(64)->UseRealTime()->Unit(benchmark::TimeUnit::kMicrosecond);
#endif
static void BM_BatchNormEigenTensorSingleThread(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  const std::vector<size_t> dims_vec{batch_size, 64, 75, 75};
  const size_t N = dims_vec[0];
  const size_t C = dims_vec[1];  // assume NCHW as per the spec

  // calculate sample_size (per individual channel)
  size_t sample_size = 1;
  for (size_t i = 2; i < dims_vec.size(); ++i) {
    sample_size *= dims_vec[i];
  }
  using T = float;
  Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> input_tensor(N * C, sample_size);
  input_tensor.setRandom();
  Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> scale_arr(C);
  scale_arr.setRandom();
  Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> mean_arr(C);
  mean_arr.setRandom();
  Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> bias_arr(C);
  bias_arr.setRandom();
  Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> var_arr(C);
  var_arr.setRandom();
  Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex> output_tensor(N * C, sample_size);

  float epsilon_ = 1e-5f;
  for (auto _ : state) {
    Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> eps(C);
    eps.setConstant(epsilon_);
    Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex> inv_std(C);
    Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
    batch_by_one.set(0, C);
    Eigen::IndexList<int, int> bcast;
    bcast.set(0, N);
    bcast.set(1, sample_size);
    inv_std = (scale_arr / (var_arr + eps).sqrt());
    output_tensor = input_tensor * inv_std.reshape(batch_by_one).broadcast(bcast) +
                    (bias_arr - mean_arr * inv_std).eval().reshape(batch_by_one).broadcast(bcast);
  }
}

BENCHMARK(BM_BatchNormEigenTensorSingleThread)
    ->Arg(1)
    ->Arg(16)
    ->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);