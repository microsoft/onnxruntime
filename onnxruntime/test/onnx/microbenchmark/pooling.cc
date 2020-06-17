// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <random>
#include <mlas.h>
#include "onnx/defs/attr_proto_util.h"

#include <core/graph/onnx_protobuf.h>
#include <core/graph/model.h>
#include <benchmark/benchmark.h>

#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/util/thread_utils.h>
#include <core/providers/cpu/nn/pool_functors.h>
#include <core/providers/cpu/nn/pool.h>
#include "dummy_execution_frame.h"
#include "core/session/ort_env.h"

using namespace onnxruntime;
using namespace onnx;
using namespace onnxruntime::concurrency;

extern OrtEnv* env;
extern const OrtApi* g_ort;

static int64_t CalcSize(int64_t* shape, size_t len) {
  int64_t ret = 1;
  for (size_t i = 0; i != len; ++i) ret *= shape[i];
  return ret;
}

static void RunMlasPool2D(const OrtThreadPoolParams& param, int64_t batch_size, benchmark::State& state) {
  std::unique_ptr<ThreadPool> tp = CreateThreadPool(&onnxruntime::Env::Default(), param, onnxruntime::concurrency::ThreadPoolType::INTRA_OP);
  int64_t input_shape[] = {1, 64, 112, 112};
  int64_t kernel_shape[] = {3, 3};
  int64_t padding[] = {0, 0, 1, 1};
  int64_t stride_shape[] = {2, 2};
  int64_t output_shape[] = {1, 64, 56, 56};
  input_shape[0] = batch_size;
  output_shape[0] = batch_size;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-10, 10);
  std::vector<float> input(CalcSize(input_shape, 4));
  std::vector<float> output(CalcSize(output_shape, 4));

  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = dist(gen);
  }
  for (size_t i = 0; i != output.size(); ++i) {
    output[i] = dist(gen);
  }
  for (auto _ : state) {
    MlasPool(MlasMaximumPooling, 2, input_shape, kernel_shape, padding, stride_shape, output_shape, input.data(),
             output.data(), tp.get());
  }
}

static void BM_MlasPoolWithSpinAndAffinity(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  OrtThreadPoolParams param;
  param.auto_set_affinity = true;
  param.allow_spinning = true;
  RunMlasPool2D(param, batch_size, state);
}

static void BM_MlasPoolWithSpinNoAffinity(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  OrtThreadPoolParams param;
  param.auto_set_affinity = false;
  param.allow_spinning = true;
  RunMlasPool2D(param, batch_size, state);
}

static void BM_MlasPoolNoSpinNoAffinity(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  OrtThreadPoolParams param;
  param.auto_set_affinity = false;
  param.allow_spinning = false;
  RunMlasPool2D(param, batch_size, state);
}

static void BM_MlasPoolNoSpinWithAffinity(benchmark::State& state) {
  const size_t batch_size = state.range(0);
  OrtThreadPoolParams param;
  param.auto_set_affinity = true;
  param.allow_spinning = false;
  RunMlasPool2D(param, batch_size, state);
}

BENCHMARK(BM_MlasPoolWithSpinAndAffinity)->UseRealTime()->Arg(1)->Arg(4)->Unit(benchmark::TimeUnit::kMicrosecond);
BENCHMARK(BM_MlasPoolWithSpinNoAffinity)->UseRealTime()->Arg(1)->Arg(4)->Unit(benchmark::TimeUnit::kMicrosecond);
BENCHMARK(BM_MlasPoolNoSpinWithAffinity)->UseRealTime()->Arg(1)->Arg(4)->Unit(benchmark::TimeUnit::kMicrosecond);
BENCHMARK(BM_MlasPoolNoSpinNoAffinity)->UseRealTime()->Arg(1)->Arg(4)->Unit(benchmark::TimeUnit::kMicrosecond);

using namespace onnxruntime;
using namespace onnxruntime::common;
float* GenerateFloatArray(size_t batch_size, float low, float high);

static void BM_Maxpool(benchmark::State& state) {
  
  std::vector<onnx::AttributeProto> attrs = {
  onnx::MakeAttribute("auto_pad", "NOTSET"),
  onnx::MakeAttribute("kernel_shape", std::vector<int64_t>{3,3}),
  onnx::MakeAttribute("pads", std::vector<int64_t>{0,0,2,2}),
  onnx::MakeAttribute("strides", std::vector<int64_t>{2,2}),
  onnx::MakeAttribute("storage_order", static_cast<int64_t>(0))};
  float low = -10.0;
  float high = 10.0;
  using KernelType = onnxruntime::MaxPoolV8;
  std::string op_name = "MaxPool";
  std::string domain = "";

  const int64_t batch_size = state.range(0);
  TensorShape shapes{1,batch_size,112,112};

  float* data = GenerateFloatArray(shapes.Size(), low, high);
  KernelAndDef k = KernelAndDef::CreateKernel<KernelType>(op_name, domain, attrs, shapes.GetDims());

  std::vector<int> feed_mlvalue_idxs(1);
  std::vector<int> fetch_mlvalue_idxs(1);
  ORT_THROW_IF_ERROR(k.ort_value_idx_map->GetIdx("input", feed_mlvalue_idxs[0]));
  ORT_THROW_IF_ERROR(k.ort_value_idx_map->GetIdx("output", fetch_mlvalue_idxs[0]));

  std::vector<OrtValue> feeds(1);
  std::vector<OrtValue> fetches(1);
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  OrtMemoryInfo info("cpu", OrtDeviceAllocator);
  feeds[0].Init(new Tensor(DataTypeImpl::GetType<float>(), shapes, data, info), ml_tensor, ml_tensor->GetDeleteFunc());                   
  GraphViewer v(k.model->MainGraph());
  NodeIndexInfo node_index_info(v, *k.ort_value_idx_map);
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));
  MyIExecutionFrame f(*k.a, feed_mlvalue_idxs, feeds, {}, fetch_mlvalue_idxs, fetches, *k.ort_value_idx_map,
                      node_index_info);
  for (auto _ : state) {
    OpKernelContext c(&f, k.kernel.get(), tp.get(), *k.test_logger);
    Status st = k.kernel->Compute(&c);
    if (!st.IsOK())
      state.SkipWithError(st.ErrorMessage().c_str());
  }
  _aligned_free(data);
}

BENCHMARK(BM_Maxpool)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond)
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(64);    
   