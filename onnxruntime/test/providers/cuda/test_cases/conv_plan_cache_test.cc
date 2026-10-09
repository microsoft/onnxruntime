// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#if !defined(USE_CUDA_MINIMAL) && !defined(BUILD_CUDA_EP_AS_PLUGIN)

#include <cudnn_version.h>

#if CUDNN_MAJOR >= 9

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/framework/allocator.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "test/providers/cuda/test_cases/cuda_test_bridge.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime::test {
namespace {

constexpr int64_t kChannels = 8;

struct CudaStreamDeleter {
  void operator()(CUstream_st* stream) const {
    static_cast<void>(cudaStreamDestroy(stream));
  }
};

struct CudaGraphDeleter {
  void operator()(CUgraph_st* graph) const {
    static_cast<void>(cudaGraphDestroy(graph));
  }
};

struct CudaGraphExecDeleter {
  void operator()(CUgraphExec_st* graph) const {
    static_cast<void>(cudaGraphExecDestroy(graph));
  }
};

std::string ConvModel(int32_t element_type, bool with_bias, int64_t groups) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model.add_opset_import()->set_version(22);
  auto* graph = model.mutable_graph();
  graph->set_name("conv_plan_cache");
  auto* node = graph->add_node();
  node->set_op_type("Conv");
  node->set_name("conv");
  node->add_input("X");
  node->add_input("W");
  if (with_bias) node->add_input("B");
  node->add_output("Y");
  auto* group = node->add_attribute();
  group->set_name("group");
  group->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  group->set_i(groups);

  const auto add_value = [element_type](ONNX_NAMESPACE::ValueInfoProto* value,
                                        const char* name, std::initializer_list<int64_t> dims) {
    value->set_name(name);
    auto* type = value->mutable_type()->mutable_tensor_type();
    type->set_elem_type(element_type);
    for (int64_t dimension : dims) {
      auto* dim = type->mutable_shape()->add_dim();
      if (dimension < 0) {
        dim->set_dim_param(std::string(name) + "_dim_" + std::to_string(type->shape().dim_size()));
      } else {
        dim->set_dim_value(dimension);
      }
    }
  };
  add_value(graph->add_input(), "X", {-1, kChannels, -1});
  add_value(graph->add_input(), "W", {-1, kChannels / groups, -1});
  if (with_bias) add_value(graph->add_input(), "B", {-1});
  add_value(graph->add_output(), "Y", {-1, kChannels, -1});
  return model.SerializeAsString();
}

template <typename T>
class ConvPlanCacheTest : public testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "A CUDA device is required.";
    }
    cudaDeviceProp device{};
    ASSERT_EQ(cudaGetDeviceProperties(&device, 0), cudaSuccess);
    if constexpr (std::is_same_v<T, BFloat16>) {
      if (device.major < 8) GTEST_SKIP() << "BF16 convolution requires SM80 or newer.";
    }
  }

  void Initialize(bool with_bias = true, bool fuse_bias = !std::is_same_v<T, BFloat16>,
                  bool pad_to_nc1d = false, bool use_nonblocking_stream = false, int64_t groups = kChannels) {
    with_bias_ = with_bias;
    groups_ = groups;
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    options.enable_mem_pattern = false;
    gpu_ = std::make_unique<InferenceSessionWrapper>(options, GetEnvironment());
    cpu_ = std::make_unique<InferenceSessionWrapper>(options, GetEnvironment());
    CUDAExecutionProviderInfo info;
    info.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
    info.use_tf32 = false;
    info.fuse_conv_bias = fuse_bias;
    info.cudnn_conv1d_pad_to_nc1d = pad_to_nc1d;
    if (use_nonblocking_stream) {
      cudaStream_t stream = nullptr;
      ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
      stream_.reset(stream);
      info.has_user_compute_stream = true;
      info.user_compute_stream = stream;
    }
    ASSERT_STATUS_OK(gpu_->RegisterExecutionProvider(std::make_shared<CUDAExecutionProvider>(info)));
    element_type_ = std::is_same_v<T, BFloat16>    ? ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16
                    : std::is_same_v<T, MLFloat16> ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
                                                   : ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    const auto gpu_model = ConvModel(element_type_, with_bias, groups_);
    const auto cpu_model = ConvModel(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, with_bias, groups_);
    ASSERT_STATUS_OK(gpu_->Load(gpu_model.data(), static_cast<int>(gpu_model.size())));
    ASSERT_STATUS_OK(cpu_->Load(cpu_model.data(), static_cast<int>(cpu_model.size())));
    ASSERT_STATUS_OK(gpu_->Initialize());
    ASSERT_STATUS_OK(cpu_->Initialize());
    const auto& graph = gpu_->GetGraph();
    ASSERT_EQ(graph.NumberOfNodes(), 1u);
    const auto& node = *graph.Nodes().begin();
    ASSERT_EQ(node.GetExecutionProviderType(), kCudaExecutionProvider);
    kernel_ = gpu_->GetSessionState().GetKernel(node.Index());
    ASSERT_NE(kernel_, nullptr);
    device_allocator_ = std::make_shared<CUDAAllocator>(0, CUDA);
    host_allocator_ = std::make_shared<CPUAllocator>();
  }

  void Run(int64_t rows, int64_t kernel_width = 4, bool expect_success = true, int64_t bias_channels = kChannels,
           int64_t weight_channels = kChannels, int64_t batch = 1) {
    SCOPED_TRACE(MakeString("rows=", rows, ", kernel_width=", kernel_width, ", iteration=", iteration_));
    const int64_t input_length = rows + kernel_width - 1;
    std::vector<TensorShape> shapes{{batch, kChannels, input_length}, {weight_channels, kChannels / groups_, kernel_width}};
    std::vector<std::string> names{"X", "W"};
    if (with_bias_) {
      shapes.emplace_back(TensorShape{bias_channels});
      names.emplace_back("B");
    }
    std::vector<OrtValue> feeds(shapes.size());
    std::vector<OrtValue> cpu_feeds(shapes.size());
    for (size_t input_index = 0; input_index < shapes.size(); ++input_index) {
      Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), shapes[input_index], device_allocator_, feeds[input_index]);
      Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), shapes[input_index], host_allocator_, cpu_feeds[input_index]);
      std::vector<T> values(static_cast<size_t>(shapes[input_index].Size()));
      auto* reference = cpu_feeds[input_index].GetMutable<Tensor>()->MutableData<float>();
      for (size_t value_index = 0; value_index < values.size(); ++value_index) {
        const float value = static_cast<float>(static_cast<int>((value_index + iteration_) % 5) - 2) *
                            (input_index == 0 ? 1.0f : 0.25f);
        values[value_index] = T(value);
        reference[value_index] = value;
      }
      ASSERT_EQ(cudaMemcpy(feeds[input_index].GetMutable<Tensor>()->MutableDataRaw(), values.data(),
                           values.size() * sizeof(T), cudaMemcpyHostToDevice),
                cudaSuccess);
    }
    std::vector<OrtValue> outputs(1);
    Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), TensorShape{batch, kChannels, rows}, device_allocator_, outputs[0]);
    if (stream_ && rows > 0 && batch > 0) {
      ASSERT_EQ(cudaMemsetAsync(outputs[0].GetMutable<Tensor>()->MutableDataRaw(), 0x7f,
                                outputs[0].Get<Tensor>().SizeInBytes(), stream_.get()),
                cudaSuccess);
    }
    const std::array<std::string, 1> output_names{"Y"};
    const auto status = gpu_->Run(RunOptions{}, names, feeds, output_names, &outputs);
    snapshot_ = GetConvPlanCacheForTest(kernel_, element_type_);
    ++iteration_;
    retained_feeds_.push_back(std::move(feeds));
    retained_outputs_.push_back(std::move(outputs));
    if (!expect_success) {
      ASSERT_FALSE(status.IsOK());
      EXPECT_FALSE(state_->conv_plan_matches_inputs);
      return;
    }
    ASSERT_STATUS_OK(status);
    if (rows == 0 || batch == 0) {
      EXPECT_EQ(retained_outputs_.back()[0].Get<Tensor>().Shape().Size(), 0);
      EXPECT_FALSE(state_->conv_plan_matches_inputs);
      return;
    }
    std::vector<OrtValue> reference_outputs;
    ASSERT_STATUS_OK(cpu_->Run(RunOptions{}, names, cpu_feeds, output_names, &reference_outputs));
    const auto& actual = retained_outputs_.back()[0].Get<Tensor>();
    const auto& expected = reference_outputs[0].Get<Tensor>();
    ASSERT_EQ(actual.Shape(), expected.Shape());
    std::vector<T> actual_values(static_cast<size_t>(actual.Shape().Size()));
    ASSERT_EQ(cudaMemcpy(actual_values.data(), actual.DataRaw(), actual.SizeInBytes(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (size_t value_index = 0; value_index < actual_values.size(); ++value_index) {
      EXPECT_FLOAT_EQ(static_cast<float>(actual_values[value_index]), expected.Data<float>()[value_index]);
    }
    ASSERT_NE(state_->conv_plan, nullptr);
    EXPECT_TRUE(state_->conv_plan_matches_inputs);
    EXPECT_EQ(state_->workspace_bytes, state_->plan_workspace_bytes);
    EXPECT_EQ(state_->x_binding, retained_feeds_.back()[0].Get<Tensor>().DataRaw());
    EXPECT_EQ(state_->w_binding, retained_feeds_.back()[1].Get<Tensor>().DataRaw());
    EXPECT_EQ(state_->y_binding, actual.DataRaw());
    if (state_->bias_fused && with_bias_) {
      EXPECT_EQ(state_->b_binding, retained_feeds_.back()[2].Get<Tensor>().DataRaw());
      EXPECT_EQ(state_->z_binding, actual.DataRaw());
    }
  }

  void CheckRecurringShapes() {
    std::array<std::shared_ptr<void>, 4> plans;
    for (int64_t rows = 1; rows <= 4; ++rows) {
      ASSERT_NO_FATAL_FAILURE(Run(rows));
      plans[static_cast<size_t>(rows - 1)] = state_->conv_plan;
    }
    for (int64_t rows : {1, 4, 2, 3, 1, 2, 4, 3, 3}) {
      ASSERT_NO_FATAL_FAILURE(Run(rows));
      EXPECT_EQ(state_->conv_plan, plans[static_cast<size_t>(rows - 1)]);
      EXPECT_EQ(state_->cached_plan_count, 4u);
    }
  }

  void CaptureAndReplay() {
    ASSERT_NE(stream_, nullptr);
    ASSERT_NO_FATAL_FAILURE(Run(1));
    if constexpr (std::is_same_v<T, float>) {
      ASSERT_TRUE(state_->bias_fused);
    }
    const auto first_plan = state_->conv_plan;
    auto& outputs = retained_outputs_.back();
    auto& output = *outputs[0].GetMutable<Tensor>();
    std::vector<T> expected(static_cast<size_t>(output.Shape().Size()));
    ASSERT_EQ(cudaMemcpy(expected.data(), output.DataRaw(), output.SizeInBytes(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    RunOptions run_options;
    ASSERT_STATUS_OK(run_options.config_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1"));
    const std::array<std::string, 3> input_names{"X", "W", "B"};
    const std::array<std::string, 1> output_names{"Y"};
    ASSERT_EQ(cudaStreamBeginCapture(stream_.get(), cudaStreamCaptureModeThreadLocal), cudaSuccess);
    const auto memset_status = cudaMemsetAsync(output.MutableDataRaw(), 0x7f, output.SizeInBytes(), stream_.get());
    const auto run_status = gpu_->Run(run_options, input_names, retained_feeds_.back(), output_names, &outputs);
    cudaGraph_t raw_graph = nullptr;
    const auto capture_status = cudaStreamEndCapture(stream_.get(), &raw_graph);
    std::unique_ptr<CUgraph_st, CudaGraphDeleter> graph(raw_graph);
    ASSERT_EQ(memset_status, cudaSuccess);
    ASSERT_STATUS_OK(run_status);
    ASSERT_EQ(capture_status, cudaSuccess);
    cudaGraphExec_t raw_instance = nullptr;
    const auto instantiate_status = cudaGraphInstantiateWithFlags(&raw_instance, graph.get(), 0);
    std::unique_ptr<CUgraphExec_st, CudaGraphExecDeleter> instance(raw_instance);
    ASSERT_EQ(instantiate_status, cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(instance.get(), stream_.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_.get()), cudaSuccess);
    std::vector<T> actual(expected.size());
    ASSERT_EQ(cudaMemcpy(actual.data(), output.DataRaw(), output.SizeInBytes(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_FLOAT_EQ(static_cast<float>(actual[i]), static_cast<float>(expected[i]));
    }
    snapshot_ = GetConvPlanCacheForTest(kernel_, element_type_);
    EXPECT_EQ(state_->conv_plan, first_plan);
    EXPECT_EQ(state_->cached_plan_count, 1u);
  }

  bool with_bias_ = false;
  int64_t groups_ = kChannels;
  size_t iteration_ = 0;
  int32_t element_type_ = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  std::unique_ptr<CUstream_st, CudaStreamDeleter> stream_;
  std::unique_ptr<InferenceSessionWrapper> gpu_;
  std::unique_ptr<InferenceSessionWrapper> cpu_;
  const void* kernel_ = nullptr;
  ConvPlanCacheSnapshot snapshot_;
  const ConvPlanCacheSnapshot* state_ = &snapshot_;
  AllocatorPtr device_allocator_;
  AllocatorPtr host_allocator_;
  std::vector<std::vector<OrtValue>> retained_feeds_;
  std::vector<std::vector<OrtValue>> retained_outputs_;
};

using ConvPlanCacheTypes = testing::Types<float, MLFloat16, BFloat16>;
TYPED_TEST_SUITE(ConvPlanCacheTest, ConvPlanCacheTypes);

TYPED_TEST(ConvPlanCacheTest, RecurringShapesRefreshPointers) {
  ASSERT_NO_FATAL_FAILURE(this->Initialize());
  ASSERT_NO_FATAL_FAILURE(this->CheckRecurringShapes());
}

TYPED_TEST(ConvPlanCacheTest, RecurringShapesWithoutBias) {
  ASSERT_NO_FATAL_FAILURE(this->Initialize(false));
  ASSERT_NO_FATAL_FAILURE(this->CheckRecurringShapes());
}

TYPED_TEST(ConvPlanCacheTest, RecurringShapesWithUnfusedBias) {
  ASSERT_NO_FATAL_FAILURE(this->Initialize(true, false));
  ASSERT_NO_FATAL_FAILURE(this->CheckRecurringShapes());
  EXPECT_FALSE(this->state_->bias_fused);
}

TYPED_TEST(ConvPlanCacheTest, RecurringShapesWithAlternate1dPromotion) {
  ASSERT_NO_FATAL_FAILURE(this->Initialize(true, !std::is_same_v<TypeParam, BFloat16>, true));
  ASSERT_NO_FATAL_FAILURE(this->CheckRecurringShapes());
}

TYPED_TEST(ConvPlanCacheTest, RecurringShapesOnNonblockingStream) {
  ASSERT_NO_FATAL_FAILURE(this->Initialize(true, !std::is_same_v<TypeParam, BFloat16>, false, true));
  ASSERT_NO_FATAL_FAILURE(this->CheckRecurringShapes());
}

TYPED_TEST(ConvPlanCacheTest, CapturingCachedPlanOnNonblockingStream) {
  ASSERT_NO_FATAL_FAILURE(this->Initialize(true, !std::is_same_v<TypeParam, BFloat16>, false, true, 1));
  ASSERT_NO_FATAL_FAILURE(this->CaptureAndReplay());
}

TYPED_TEST(ConvPlanCacheTest, EmptyOutputDoesNotReplaceCachedPlan) {
  ASSERT_NO_FATAL_FAILURE(this->Initialize());
  ASSERT_NO_FATAL_FAILURE(this->Run(1));
  const auto first_plan = this->state_->conv_plan;
  ASSERT_NO_FATAL_FAILURE(this->Run(1, 4, true, kChannels, kChannels, 0));
  EXPECT_EQ(this->state_->cached_plan_count, 1u);
  ASSERT_NO_FATAL_FAILURE(this->Run(1));
  EXPECT_EQ(this->state_->conv_plan, first_plan);
  EXPECT_EQ(this->state_->cached_plan_count, 1u);
}

TYPED_TEST(ConvPlanCacheTest, CapacityAndLeastRecentlyUsedEviction) {
  ASSERT_NO_FATAL_FAILURE(this->Initialize());
  std::array<std::shared_ptr<void>, 8> plans;
  for (int64_t rows = 1; rows <= 8; ++rows) {
    ASSERT_NO_FATAL_FAILURE(this->Run(rows));
    plans[static_cast<size_t>(rows - 1)] = this->state_->conv_plan;
  }
  ASSERT_NO_FATAL_FAILURE(this->Run(1));
  EXPECT_EQ(this->state_->conv_plan, plans[0]);
  ASSERT_NO_FATAL_FAILURE(this->Run(9));
  EXPECT_EQ(this->state_->cached_plan_count, 8u);
  ASSERT_NO_FATAL_FAILURE(this->Run(1));
  EXPECT_EQ(this->state_->conv_plan, plans[0]);
  ASSERT_NO_FATAL_FAILURE(this->Run(2));
  EXPECT_NE(this->state_->conv_plan, plans[1]);
  EXPECT_EQ(this->state_->cached_plan_count, 8u);
}

TYPED_TEST(ConvPlanCacheTest, DynamicWeightsSelectMatchingPlan) {
  ASSERT_NO_FATAL_FAILURE(this->Initialize());
  ASSERT_NO_FATAL_FAILURE(this->Run(4, 4));
  const auto first_plan = this->state_->conv_plan;
  const auto input_dims = this->state_->last_x_dims;
  ASSERT_NO_FATAL_FAILURE(this->Run(5, 3));
  EXPECT_EQ(this->state_->last_x_dims, input_dims);
  EXPECT_NE(this->state_->conv_plan, first_plan);
  ASSERT_NO_FATAL_FAILURE(this->Run(4, 4));
  EXPECT_EQ(this->state_->conv_plan, first_plan);
  EXPECT_EQ(this->state_->cached_plan_count, 2u);
}

TYPED_TEST(ConvPlanCacheTest, FailedInitializationDoesNotPoisonCache) {
  ASSERT_NO_FATAL_FAILURE(this->Initialize());
  ASSERT_NO_FATAL_FAILURE(this->Run(1));
  const auto first_plan = this->state_->conv_plan;
  if constexpr (std::is_same_v<TypeParam, BFloat16>) {
    ASSERT_NO_FATAL_FAILURE(this->Run(1, 4, false, kChannels, kChannels - 1));
  } else {
    ASSERT_NO_FATAL_FAILURE(this->Run(1, 4, false, kChannels - 1));
  }
  EXPECT_EQ(this->state_->cached_plan_count, 1u);
  ASSERT_NO_FATAL_FAILURE(this->Run(1));
  EXPECT_EQ(this->state_->conv_plan, first_plan);
  EXPECT_EQ(this->state_->cached_plan_count, 1u);
}

TYPED_TEST(ConvPlanCacheTest, InvalidUnfusedBiasDoesNotPublishPlan) {
  ASSERT_NO_FATAL_FAILURE(this->Initialize(true, false));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(this->Run(1, 4, false, kChannels - 1));
    EXPECT_EQ(this->state_->cached_plan_count, 0u);
  }
  ASSERT_NO_FATAL_FAILURE(this->Run(1));
  EXPECT_EQ(this->state_->cached_plan_count, 1u);
}

}  // namespace
}  // namespace onnxruntime::test

#endif
#endif