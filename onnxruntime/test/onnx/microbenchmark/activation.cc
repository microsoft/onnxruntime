// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ort_env.h"
#include "core/graph/model.h"
#include "core/graph/graph.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/data_transfer_manager.h"
#include "core/util/thread_utils.h"
#include "core/framework/node_index_info.h"
#include "core/framework/execution_frame.h"
#include "contrib_ops/cpu/activations.h"
#include "core/providers/cpu/activation/activations.h"
#include <onnx/defs/attr_proto_util.h>
#include <benchmark/benchmark.h>
#include <random>
#include "dummy_execution_frame.h"

using namespace onnxruntime;
using namespace onnx;
extern OrtEnv* env;

 float* GenerateFloatArray(size_t batch_size, float low, float high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(low, high);
  float* data = (float*)_aligned_malloc(sizeof(float) * batch_size, 64);
  for (size_t i = 0; i != batch_size; ++i) {
    data[i] = dist(gen);
  }
  return data;
}


static void BM_GeluCompute(OrtThreadPoolImplType impl_type, benchmark::State& state) {
  RunSingleNode<contrib::Gelu<float>>("Gelu", kMSDomain, {}, impl_type, state);
}

#define DEFINE_BENCHMARK_WITH_TP_TYPE(X)\
static void BM_Win32TP_ ## X(benchmark::State& state) { \
  BM_ ## X(ORT_THREAD_POOL_TYPE_WIN32, state); \
} \
static void BM_EigenTP_ ## X(benchmark::State& state) { \
  BM_ ## X(ORT_THREAD_POOL_TYPE_DEFAULT, state); \
}

DEFINE_BENCHMARK_WITH_TP_TYPE(GeluCompute);

BENCHMARK(BM_EigenTP_GeluCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(98304)
    ->Arg(1572864);

BENCHMARK(BM_Win32TP_GeluCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(98304)
    ->Arg(1572864);
static void BM_ScaledTanhCompute(OrtThreadPoolImplType impl_type, benchmark::State& state) {
  RunSingleNode<contrib::ScaledTanh<float>>("ScaledTanh", kMSDomain,
                                            {MakeAttribute("alpha", 0.8f), MakeAttribute("beta", 0.3f)}, impl_type, state);
}
DEFINE_BENCHMARK_WITH_TP_TYPE(ScaledTanhCompute);

BENCHMARK(BM_EigenTP_ScaledTanhCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000);

BENCHMARK(BM_Win32TP_ScaledTanhCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000);
static void BM_EluCompute(OrtThreadPoolImplType impl_type, benchmark::State& state) {
  RunSingleNode<Elu<float>>("Elu", "",
                            {
                                MakeAttribute("alpha", 0.8f),
                            },
                            impl_type, state);
}
DEFINE_BENCHMARK_WITH_TP_TYPE(EluCompute);

BENCHMARK(BM_EigenTP_EluCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(2000)
    ->Arg(4000)
    ->Arg(8000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000);

BENCHMARK(BM_Win32TP_EluCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(2000)
    ->Arg(4000)
    ->Arg(8000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000);

static void BM_HardSigmoidCompute(OrtThreadPoolImplType impl_type, benchmark::State& state) {
  RunSingleNode<HardSigmoid<float>>("HardSigmoid", "", {MakeAttribute("alpha", 0.2f), MakeAttribute("beta", 0.5f)},
                                    impl_type, state, 0.1f, 0.6f);
}
DEFINE_BENCHMARK_WITH_TP_TYPE(HardSigmoidCompute);

BENCHMARK(BM_EigenTP_HardSigmoidCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_LeakyReluCompute(OrtThreadPoolImplType impl_type, benchmark::State& state) {
  RunSingleNode<LeakyRelu<float>>("LeakyRelu", "", {MakeAttribute("alpha", 0.2f)}, impl_type, state);
}
DEFINE_BENCHMARK_WITH_TP_TYPE(LeakyReluCompute);

BENCHMARK(BM_EigenTP_LeakyReluCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(4000)
    ->Arg(8000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000);

static void BM_SoftplusCompute(OrtThreadPoolImplType impl_type, benchmark::State& state) {
  RunSingleNode<Softplus<float>>("Softplus", "", {}, impl_type, state, -2.0f, 2.0f);
}
DEFINE_BENCHMARK_WITH_TP_TYPE(SoftplusCompute);

BENCHMARK(BM_EigenTP_SoftplusCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_Selu(OrtThreadPoolImplType impl_type, benchmark::State& state) {
  RunSingleNode<Selu<float>>("Selu", "", {}, impl_type, state, -2.0f, 2.0f);
}
DEFINE_BENCHMARK_WITH_TP_TYPE(Selu);

BENCHMARK(BM_EigenTP_Selu)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_Sigmoid(OrtThreadPoolImplType impl_type, benchmark::State& state) {
  RunSingleNode<Sigmoid<float>>("Sigmoid", "", {}, impl_type, state, -2.0f, 2.0f);
}
DEFINE_BENCHMARK_WITH_TP_TYPE(Sigmoid);

BENCHMARK(BM_EigenTP_Sigmoid)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_Softsign(OrtThreadPoolImplType impl_type, benchmark::State& state) {
  RunSingleNode<Softsign<float>>("Softsign", "", {}, impl_type, state, -2.0f, 2.0f);
}
DEFINE_BENCHMARK_WITH_TP_TYPE(Softsign);

BENCHMARK(BM_EigenTP_Softsign)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_Tanh(OrtThreadPoolImplType impl_type, benchmark::State& state) {
  RunSingleNode<Tanh<float>>("Tanh", "", {}, impl_type, state, -2.0f, 2.0f);
}
DEFINE_BENCHMARK_WITH_TP_TYPE(Tanh);

BENCHMARK(BM_EigenTP_Tanh)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_Relu(OrtThreadPoolImplType impl_type, benchmark::State& state) {
  RunSingleNode<Relu<float>>("Relu", "", {}, impl_type, state, -2.0f, 2.0f);
}
DEFINE_BENCHMARK_WITH_TP_TYPE(Relu);

BENCHMARK(BM_EigenTP_Relu)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

template <typename T>
struct Powx {
  const T* input1 = nullptr;
  const T* input2 = nullptr;
  T* output = nullptr;
  float Cost() const {
    return 30.f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    const T* in1 = this->input1 + first;
    const T* in2 = this->input2 + first;
    for (ptrdiff_t i = 0; i != len; ++i) {
      output_ptr[i] = std::pow(in1[i], in2[i]);
    }
  }
};

static void BM_Powx(OrtThreadPoolImplType impl_type, benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const int cost = static_cast<int>(state.range(1));
  float* output = (float*)_aligned_malloc(sizeof(float) * batch_size, 64);
  float* input2 = GenerateFloatArray(batch_size, -1, 1);
  float* input1 = GenerateFloatArray(batch_size, -1, 1);
  OrtThreadPoolParams tpo;
  tpo.impl_type = impl_type;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));
  Powx<float> f;
  f.input1 = input1;
  f.input2 = input2;
  f.output = output;
  for (auto _ : state) {
    tp->ParallelFor(batch_size, TensorOpCost{2, 1, static_cast<double>(cost)}, f);
  }
  _aligned_free(input1);
  _aligned_free(input2);
  _aligned_free(output);
}
DEFINE_BENCHMARK_WITH_TP_TYPE(Powx);

BENCHMARK(BM_EigenTP_Powx)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Args({100, 1})
    ->Args({100, 5})
    ->Args({100, 10})
    ->Args({100, 40})
    ->Args({100, 80})
    ->Args({100, 160})
    ->Args({100, 320})
    ->Args({100, 640})
    ->Args({500, 1})
    ->Args({500, 5})
    ->Args({500, 10})
    ->Args({500, 40})
    ->Args({500, 80})
    ->Args({500, 160})
    ->Args({500, 320})
    ->Args({500, 640})
    ->Args({1000, 1})
    ->Args({1000, 5})
    ->Args({1000, 10})
    ->Args({1000, 40})
    ->Args({1000, 80})
    ->Args({1000, 160})
    ->Args({1000, 320})
    ->Args({2000, 1})
    ->Args({2000, 5})
    ->Args({2000, 10})
    ->Args({2000, 40})
    ->Args({2000, 80})
    ->Args({2000, 160})
    ->Args({2000, 320})
    ->Args({2500, 1})
    ->Args({2500, 5})
    ->Args({2500, 10})
    ->Args({2500, 40})
    ->Args({2500, 80})
    ->Args({2500, 160})
    ->Args({5000, 1})
    ->Args({5000, 5})
    ->Args({5000, 10})
    ->Args({5000, 40})
    ->Args({5000, 80})
    ->Args({5000, 160});
