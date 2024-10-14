#ifdef _WIN32

#include "core/platform/threadpool.h"
#include "core/util/thread_utils.h"
#include <benchmark/benchmark.h>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "core/framework/allocator.h"
#include "core/framework/config_options.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/platform/windows/env.h"
#include "core/providers/cpu/nn/layer_norm_impl.h"
#include "core/providers/cpu/cpu_provider_factory.h"
#include "core/providers/cpu/cpu_provider_factory_creator.h"
#include "core/util/thread_utils.h"

#include "test/onnx/microbenchmark/common.h"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

using namespace onnxruntime;

namespace {

std::vector<MLFloat16> createMLFloat16Vector(float* vals, int64_t num_elems) {
  std::vector<MLFloat16> fp16vec;
  fp16vec.reserve(num_elems);

  for (int64_t i = 0; i < num_elems; i++) {
    fp16vec.push_back(MLFloat16(vals[i]));
  }

  return fp16vec;
}

}  // namespace

template <typename T, typename U>
static void BM_LayerNormalization(benchmark::State& state) {
  bool simplified = false;
  const float epsilon = 1e-05f;
  int64_t axis = 1;

  onnxruntime::Node node;
  // Required by LayerNormImpl constructor
  node.AddAttribute("axis", axis);
  node.AddAttribute("epsilon", epsilon);

  KernelDef kernel_def;
  std::unique_ptr<IExecutionProvider> execution_provider = CPUProviderFactoryCreator::Create(true)->CreateProvider();
  std::unordered_map<int, OrtValue> constant_initialized_tensors;
  OrtValueNameIdxMap mlvalue_name_idx_map;
  DataTransferManager data_transfer_mgr;
  AllocatorMap allocators;
  ConfigOptions config_options;

  OpKernelInfo op_kernel_info(node, kernel_def, *execution_provider, constant_initialized_tensors, mlvalue_name_idx_map,
                              data_transfer_mgr, allocators, config_options);

  LayerNormImpl layer_norm_impl(op_kernel_info);

  const std::vector<int64_t> dims{1, 256, 1024};
  const size_t num_elems = dims[0] * dims[1] * dims[2];

  TensorShape x_shape(dims);
  TensorShape scale_shape(dims);
  TensorShape bias_shape(dims);

  const float low = -1.0f;
  const float high = 1.0f;

  float* x_float = GenerateArrayWithRandomValue<float>(num_elems, low, high);
  float* scale_float = GenerateArrayWithRandomValue<float>(num_elems, 0.1f, high);
  float* bias_float = GenerateArrayWithRandomValue<float>(num_elems, low, high);

  std::vector<MLFloat16> x_MLFloat16 = createMLFloat16Vector(x_float, num_elems);
  std::vector<MLFloat16> scale_MLFloat16 = createMLFloat16Vector(scale_float, num_elems);
  std::vector<MLFloat16> bias_MLFloat16 = createMLFloat16Vector(bias_float, num_elems);

  T* x_data = nullptr;
  T* scale_data = nullptr;
  T* bias_data = nullptr;
  if (std::is_same_v<T*, MLFloat16*>) {
    x_data = (T*)x_MLFloat16.data();
    scale_data = (T*)scale_MLFloat16.data();
    bias_data = (T*)bias_MLFloat16.data();
  } else if (std::is_same_v<T*, float*>) {
    x_data = (T*)x_float;
    scale_data = (T*)scale_float;
    bias_data = (T*)bias_float;
  }
  assert(x_data);

  T* Y_data = static_cast<T*>(aligned_alloc(num_elems * sizeof(T), 64));
  U* mean_data = static_cast<U*>(aligned_alloc(num_elems * sizeof(U), 64));
  U* inv_std_dev_data = static_cast<U*>(aligned_alloc(num_elems * sizeof(U), 64));

  OrtThreadPoolParams tp_params;
  tp_params.name = ORT_TSTR("intra-op");
  std::unique_ptr<concurrency::ThreadPool> thread_pool = concurrency::CreateThreadPool(
      &Env::Default(), tp_params, concurrency::ThreadPoolType::INTRA_OP);

  OrtMemoryInfo memory_info(onnxruntime::CPU, OrtAllocatorType::OrtArenaAllocator);
  AllocatorPtr alloc = std::make_shared<CPUAllocator>(memory_info);
  for (auto _ : state) {
    auto status = layer_norm_impl.ComputeWithoutContext(x_data, x_shape, scale_data, scale_shape, bias_data, bias_shape,
                                                        Y_data, mean_data, inv_std_dev_data, thread_pool.get(), axis,
                                                        epsilon, simplified, alloc);
    if (!status.IsOK()) {
      std::cout << "ComputeWithoutContext status not OK: " << status.ErrorMessage() << std::endl;
      break;
    }
  }

  aligned_free(x_float);
  aligned_free(scale_float);
  aligned_free(bias_float);
  aligned_free(Y_data);
  aligned_free(mean_data);
  aligned_free(inv_std_dev_data);
}

BENCHMARK(BM_LayerNormalization<float, float>)
    ->Arg(1)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

BENCHMARK(BM_LayerNormalization<MLFloat16, float>)
    ->Arg(1)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

#endif
