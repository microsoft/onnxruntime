// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && \
    !defined(BUILD_CUDA_EP_AS_PLUGIN)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <cuda_runtime_api.h>

#include "nlohmann/json.hpp"
#include "core/framework/op_kernel.h"
#include "core/framework/resource_accountant.h"
#include "core/framework/session_state.h"
#include "core/graph/onnx_protobuf.h"
#include "core/optimizer/cuda_runtime_fusion.h"
#include "core/optimizer/initializer.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/temp_dir.h"

#include "gtest/gtest.h"

namespace onnxruntime::test {
namespace {

constexpr int64_t kChannels = 10240;

std::unique_ptr<IExecutionProvider> ProductionMathCudaExecutionProvider() {
  OrtCUDAProviderOptionsV2 options{};
  options.do_copy_in_default_stream = true;
  // DefaultCudaExecutionProvider is a test helper that disables TF32, unlike production.
  options.use_tf32 = true;
  return CudaExecutionProviderWithOptions(&options);
}

void AddTensorInfo(ONNX_NAMESPACE::ValueInfoProto& value, const std::string& name,
                   std::initializer_list<int64_t> shape,
                   int element_type = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
  value.set_name(name);
  auto* type = value.mutable_type()->mutable_tensor_type();
  type->set_elem_type(element_type);
  int axis = 0;
  for (auto dim : shape) {
    auto* dimension = type->mutable_shape()->add_dim();
    if (dim < 0) {
      dimension->set_dim_param("d" + std::to_string(axis));
    } else {
      dimension->set_dim_value(dim);
    }
    ++axis;
  }
}

ONNX_NAMESPACE::NodeProto& AddOp(ONNX_NAMESPACE::GraphProto& graph, const std::string& op,
                                 std::initializer_list<const char*> inputs,
                                 std::initializer_list<const char*> outputs) {
  auto& node = *graph.add_node();
  node.set_op_type(op);
  node.set_name(op + std::to_string(graph.node_size()));
  for (auto input : inputs) node.add_input(input);
  for (auto output : outputs) node.add_output(output);
  return node;
}

void IntAttribute(ONNX_NAMESPACE::NodeProto& node, const char* name, int64_t value) {
  auto* attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr->set_i(value);
}

void AddSmallConstant(ONNX_NAMESPACE::GraphProto& graph, const char* name,
                      std::initializer_list<int64_t> shape, std::initializer_list<int64_t> values) {
  auto& node = AddOp(graph, "Constant", {}, {name});
  auto* attr = node.add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  auto* tensor = attr->mutable_t();
  tensor->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  for (auto dim : shape) tensor->add_dims(dim);
  for (auto value : values) tensor->add_int64_data(value);
}

// Reproduce the serialized topology, including the five Constants that ORT normalizes on load.
// No checkpoint data is used. Keep S anonymous to exercise the exported model's input domain.
template <typename T = BFloat16>
ONNX_NAMESPACE::ModelProto MakeMobiusModel(bool target, int64_t channels = kChannels,
                                           int64_t kernel = 4, int64_t dilation = 1) {
  const int type = std::is_same_v<T, float>       ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT
                   : std::is_same_v<T, MLFloat16> ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
                                                  : ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
  const int64_t pad = (kernel - 1) * dilation;
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(11);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(24);
  auto& graph = *model.mutable_graph();
  graph.set_name("mobius_causal_conv_admission");
  AddTensorInfo(*graph.add_input(), "x", {-1, channels, -1}, type);
  AddTensorInfo(*graph.add_input(), "h", {-1, channels, pad}, type);
  AddTensorInfo(*graph.add_output(), "y", {-1, channels, -1}, type);
  AddTensorInfo(*graph.add_output(), "state", {-1, channels, pad}, type);
  for (bool weight : {true, false}) {
    auto* tensor = graph.add_initializer();
    tensor->set_name(weight ? "w" : "bias");
    tensor->set_data_type(type);
    tensor->add_dims(channels);
    if (weight) {
      tensor->add_dims(1);
      tensor->add_dims(kernel);
    }
    const size_t count = static_cast<size_t>(channels) * (weight ? kernel : 1);
    std::vector<T> data;
    data.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      data.emplace_back(static_cast<float>(static_cast<int>(i % 23) - 11) / 37.0f);
    }
    tensor->set_raw_data(data.data(), data.size() * sizeof(T));
  }
  if (target) {
    opset = model.add_opset_import();
    opset->set_domain(kMSDomain);
    opset->set_version(1);
    auto& node = AddOp(graph, "CausalConvWithState", {"x", "w", "bias", "h"}, {"y", "state"});
    node.set_domain(kMSDomain);
    IntAttribute(node, "ndim", 1);
    IntAttribute(node, "channels_last", 0);
    IntAttribute(node, "dilation", dilation);
    IntAttribute(node, "state_window", 0);
    auto* activation = node.add_attribute();
    activation->set_name("activation");
    activation->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
    activation->set_s("silu");
    return model;
  }
  AddSmallConstant(graph, "index", {}, {2});
  AddSmallConstant(graph, "window", {}, {pad});
  AddSmallConstant(graph, "vector_shape", {1}, {1});
  AddSmallConstant(graph, "axes", {1}, {2});
  AddSmallConstant(graph, "bias_shape", {3}, {1, -1, 1});
  IntAttribute(AddOp(graph, "Concat", {"h", "x"}, {"c"}), "axis", 2);
  AddOp(graph, "Shape", {"c"}, {"shape"});
  AddOp(graph, "Gather", {"shape", "index"}, {"length"});
  AddOp(graph, "Sub", {"length", "window"}, {"start"});
  AddOp(graph, "Reshape", {"start", "vector_shape"}, {"start1"});
  AddOp(graph, "Reshape", {"length", "vector_shape"}, {"end1"});
  AddOp(graph, "Slice", {"c", "start1", "end1", "axes"}, {"state"});
  auto& conv = AddOp(graph, "Conv", {"c", "w"}, {"z"});
  IntAttribute(conv, "group", channels);
  auto* spacing = conv.add_attribute();
  spacing->set_name("dilations");
  spacing->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
  spacing->add_ints(dilation);
  AddOp(graph, "Reshape", {"bias", "bias_shape"}, {"bias3d"});
  AddOp(graph, "Add", {"z", "bias3d"}, {"a"});
  AddOp(graph, "Swish", {"a"}, {"y"});
  return model;
}

template <typename T = BFloat16>
NameMLValMap MakeFeeds(int64_t length, int64_t batch = 1, int64_t channels = kChannels, int64_t pad = 3) {
  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  NameMLValMap feeds;
  std::vector<T> x;
  x.reserve(static_cast<size_t>(batch * channels * length));
  for (int64_t i = 0; i < batch * channels * length; ++i) {
    constexpr int min_subnormal = std::is_same_v<T, float> ? -149 : std::is_same_v<T, MLFloat16> ? -24
                                                                                                 : -133;
    const float value = i % 41 == 0   ? std::ldexp(1.0f, min_subnormal)
                        : i % 41 == 1 ? std::nextafter(0.5f, 1.0f)
                                      : static_cast<float>(i % 41 - 20) / 29.0f;
    x.emplace_back(value);
  }
  std::vector<T> h;
  h.reserve(static_cast<size_t>(batch * channels * pad));
  for (int64_t i = 0; i < batch * channels * pad; ++i) {
    h.emplace_back(static_cast<float>(i % 17 + 1) / 16.0f);
  }
  CreateMLValue<T>(allocator, {batch, channels, length}, x, &feeds["x"]);
  CreateMLValue<T>(allocator, {batch, channels, pad}, h, &feeds["h"]);
  return feeds;
}

TEST(CudaRuntimeFusionTest, SwishCoefficientIsAlpha) {
  auto model = MakeMobiusModel(false);
  auto* swish = model.mutable_graph()->mutable_node(model.graph().node_size() - 1);
  ASSERT_EQ(swish->op_type(), "Swish");
  auto* coefficient = swish->add_attribute();
  coefficient->set_name("alpha");
  coefficient->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  coefficient->set_f(0.5f);
  const auto bytes = model.SerializeAsString();
  InferenceSessionWrapper session(SessionOptions{}, GetEnvironment());
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  bool swish_seen = false;
  for (const auto& node : session.GetGraph().Nodes()) {
    if (node.OpType() == "Swish") {
      swish_seen = true;
      EXPECT_EQ(node.SinceVersion(), 24);
      EXPECT_EQ(node.GetAttributes().at("alpha").f(), 0.5f);
      EXPECT_EQ(node.GetAttributes().count("beta"), 0u);
    }
  }
  ASSERT_TRUE(swish_seen);
}

TEST(CudaRuntimeFusionTest, AnonymousSequenceZeroAdmissionGate) {
  if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  auto source_model = MakeMobiusModel(false);
  ASSERT_EQ(source_model.graph().node_size(), 16);
  ASSERT_EQ(source_model.opset_import_size(), 1);
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  InferenceSessionWrapper source(options, GetEnvironment());
  ASSERT_STATUS_OK(source.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
  const auto source_bytes = source_model.SerializeAsString();
  ASSERT_STATUS_OK(source.Load(source_bytes.data(), static_cast<int>(source_bytes.size())));
  EXPECT_EQ(source.GetGraph().NumberOfNodes(), 11);
  ASSERT_NE(source.GetGraph().DomainToVersionMap().find(kMSDomain),
            source.GetGraph().DomainToVersionMap().end());
  EXPECT_EQ(source.GetGraph().DomainToVersionMap().at(kMSDomain), 1);
  bool swish_seen = false;
  for (const auto& node : source.GetGraph().Nodes()) {
    if (node.OpType() == "Swish") {
      swish_seen = true;
      const auto& attributes = node.GetAttributes();
      ASSERT_NE(attributes.find("alpha"), attributes.end());
      EXPECT_EQ(attributes.at("alpha").f(), 1.0f);
      EXPECT_EQ(attributes.count("beta"), 0u);
    }
  }
  ASSERT_TRUE(swish_seen);
  ASSERT_STATUS_OK(source.Initialize());
  bool cuda_conv = false;
  for (const auto& node : source.GetGraph().Nodes()) {
    if (node.OpType() == "Conv") {
      EXPECT_EQ(node.GetExecutionProviderType(), kCudaExecutionProvider);
      cuda_conv = true;
    }
  }
  ASSERT_TRUE(cuda_conv);
  const auto feeds = MakeFeeds(0);
  const std::vector<std::string> output_names{"y", "state"};
  std::vector<OrtValue> outputs;
  const auto source_status = source.Run(RunOptions{}, feeds, output_names, &outputs);
  EXPECT_FALSE(source_status.IsOK());
  EXPECT_NE(source_status.ErrorMessage().find("Invalid input shape: {3}"), std::string::npos)
      << source_status.ErrorMessage();

  // Use the identical target recipe with CPU fallback disabled: this failure must be CUDA's.
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  InferenceSessionWrapper target(options, GetEnvironment());
  ASSERT_STATUS_OK(target.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
  const auto target_bytes = MakeMobiusModel(true).SerializeAsString();
  ASSERT_STATUS_OK(target.Load(target_bytes.data(), static_cast<int>(target_bytes.size())));
  ASSERT_STATUS_OK(target.Initialize());
  ASSERT_EQ(target.GetGraph().NumberOfNodes(), 1);
  EXPECT_EQ(target.GetGraph().Nodes().begin()->GetExecutionProviderType(), kCudaExecutionProvider);
  outputs.clear();
  const auto status = target.Run(RunOptions{}, feeds, output_names, &outputs);
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("input length must be positive, got 0"), std::string::npos)
      << status.ErrorMessage();
}

// Per-dtype round-to-nearest error bound, including subnormals. Evaluate at the outer edge
// of an uncertainty interval so crossing a binade boundary cannot under-budget rounding.
template <typename T = BFloat16>
double HalfUlp(double value, double uncertainty = 0) {
  constexpr int precision = std::is_same_v<T, float> ? 24 : std::is_same_v<T, MLFloat16> ? 11
                                                                                         : 8;
  constexpr int min_half_subnormal = std::is_same_v<T, float> ? -150 : std::is_same_v<T, MLFloat16> ? -25
                                                                                                    : -134;
  if (value == 0 && uncertainty == 0) return std::ldexp(1.0, min_half_subnormal);
  int exponent;
  std::frexp(std::abs(value) + uncertainty, &exponent);
  return std::max(std::ldexp(1.0, exponent - precision - 1), std::ldexp(1.0, min_half_subnormal));
}

double Silu(double value) { return value / (1.0 + std::exp(-value)); }

struct RoundingBudget {
  double source;
  double target;
};

struct ProfileCleanup {
  InferenceSession& session;
  std::string path;
  ~ProfileCleanup() {
    if (path.empty()) path = session.EndProfiling();
    if (!path.empty()) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  }
};

template <typename T = float>
ONNX_NAMESPACE::ModelProto MakeUnitSwishModel(int rank, bool target = false) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(11);
  auto* opset = model.add_opset_import();
  opset->set_domain(kOnnxDomain);
  opset->set_version(24);
  auto& graph = *model.mutable_graph();
  graph.set_name("unit_swish");
  auto* input = graph.add_input();
  input->set_name("x");
  auto* tensor = input->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(std::is_same_v<T, float> ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT
                                                 : ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
  auto* shape = tensor->mutable_shape();
  for (int i = 0; i < rank; ++i) {
    auto* dim = shape->add_dim();
    if (i == 0) dim->set_dim_param("batch");  // Also retain anonymous extents.
  }
  *graph.add_output() = *input;
  graph.mutable_output(0)->set_name("y");
  auto& node = AddOp(graph, target ? "ScaledSiLU" : "Swish", {"x"}, {"y"});
  if (target) {
    opset = model.add_opset_import();
    opset->set_domain(kMSDomain);
    opset->set_version(1);
    node.set_domain(kMSDomain);
    auto* alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
    alpha->set_f(1.0f);
  }
  return model;
}

uint32_t UnitSwishBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float UnitSwishFloat(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint16_t UnitSwishBits(BFloat16 value) {
  return value.val;
}

template <typename T = float>
void RunUnitSwishNumerics(int rank, std::optional<TransformerLevel> level) {
  const auto bytes = MakeUnitSwishModel<T>(rank).SerializeAsString();
  SessionOptions options;
  if (level) options.graph_optimization_level = *level;
  options.intra_op_param.thread_pool_size = 2;
  options.inter_op_param.thread_pool_size = 1;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kUnitSwishConfig, "0"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kCausalConvConfig, "0"));
  options.enable_profiling = true;
  options.profile_file_prefix = ORT_TSTR("unit_swish_source");
  InferenceSessionWrapper source(options, GetEnvironment());
  ProfileCleanup source_cleanup{source, {}};
  options.profile_file_prefix = ORT_TSTR("unit_swish_fused");
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kUnitSwishConfig, "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kCausalConvConfig, "0"));
  InferenceSessionWrapper fused(options, GetEnvironment());
  ProfileCleanup fused_cleanup{fused, {}};
  options.enable_profiling = false;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kUnitSwishConfig, "0"));
  InferenceSessionWrapper direct(options, GetEnvironment());
  for (auto* session : {&source, &fused, &direct}) {
    OrtCUDAProviderOptionsV2 cuda_options{};
    cuda_options.use_tf32 = true;
    cuda_options.gpu_mem_limit = size_t{2} << 30;
    ASSERT_STATUS_OK(session->RegisterExecutionProvider(CudaExecutionProviderWithOptions(&cuda_options)));
    const auto model = session == &direct ? MakeUnitSwishModel<T>(rank, true).SerializeAsString() : bytes;
    ASSERT_STATUS_OK(session->Load(model.data(), static_cast<int>(model.size())));
    ASSERT_STATUS_OK(session->Initialize());
  }
  for (auto* session : {&fused, &direct}) {
    ASSERT_EQ(session->GetGraph().NumberOfNodes(), 1);
    const auto& node = *session->GetGraph().Nodes().begin();
    ASSERT_EQ(node.OpType(), "ScaledSiLU");
    ASSERT_EQ(node.GetExecutionProviderType(), kCudaExecutionProvider);
    ASSERT_EQ(node.InputDefs().size(), 1u);
    ASSERT_EQ(node.OutputDefs()[0]->Name(), "y");
  }
  bool sigmoid = false;
  for (const auto& node : source.GetGraph().Nodes()) {
    ASSERT_NE(node.OpType(), "ScaledSiLU");
    ASSERT_EQ(node.GetExecutionProviderType(), kCudaExecutionProvider);
    sigmoid |= node.OpType() == "Sigmoid";
  }
  ASSERT_TRUE(sigmoid);
  const std::array<uint32_t, 24> pinned{
      0, 0x80000000u, 1, 0x80000001u, 0x007fffffu, 0x807fffffu, 0x00800000u, 0x80800000u,
      0x3f800000u, 0xbf800000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
      0x7fc00000u, 0xffc00001u, 0x7f800001u, 0xff800001u, 0x41800000u, 0xc1800000u,
      0x42c80000u, 0xc2c80000u, 0x3f000000u, 0xbf000000u};
  const std::array<uint16_t, 24> bf16_pinned{
      0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080,
      0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0xffc1,
      0x7f81, 0xff81, 0x4180, 0xc180, 0x42c8, 0xc2c8, 0x3f00, 0xbf00};
  std::vector<std::vector<int64_t>> shapes;
  if (rank == 0) shapes = std::vector<std::vector<int64_t>>(pinned.size());
  if (rank == 1) shapes = {{0}, {257}, {1000003}};
  if (rank == 2) shapes = {{0, 128}, {1, 257}, {576, 128}};
  if constexpr (std::is_same_v<T, BFloat16>) {
    if (rank == 1) shapes = {{0}, {257}, {65536}, {1000003}};
    if (rank == 2) shapes = {{0, 17408}, {1, 257}, {12, 17408}, {1, 17408}};
  }
  if (rank == 3) shapes = {{1, 10240, 0}, {1, 10240, 1}, {1, 10240, 12}, {1, 12, 17408}, {1, 1, 17408}};
  std::mt19937 generator(20261030);
  size_t elements = 0, nans = 0, cases = 0;
  std::array<bool, 65536> observed_bf16_patterns{};
  const std::vector<std::string> output_names{"y"};
  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  for (const auto& dims : shapes) {
    SCOPED_TRACE(::testing::PrintToString(dims));
    size_t count = 1;
    for (auto dim : dims) count *= static_cast<size_t>(dim);
    std::vector<T> input;
    input.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if constexpr (std::is_same_v<T, float>) {
        input.push_back(UnitSwishFloat(rank == 0 ? pinned[cases] : i < pinned.size() ? pinned[i]
                                                                                     : generator()));
      } else {
        const auto bits = rank == 0                     ? bf16_pinned[cases]
                          : rank == 1 && count == 65536 ? static_cast<uint16_t>(i)
                          : i < bf16_pinned.size()      ? bf16_pinned[i]
                                                        : static_cast<uint16_t>(generator());
        input.push_back(BFloat16::FromBits(bits));
        observed_bf16_patterns[bits] = true;
      }
    }
    NameMLValMap feeds;
    CreateMLValue<T>(allocator, dims, input, &feeds["x"]);
    std::vector<OrtValue> expected, actual, target;
    ASSERT_STATUS_OK(source.Run(RunOptions{}, feeds, output_names, &expected));
    ASSERT_STATUS_OK(fused.Run(RunOptions{}, feeds, output_names, &actual));
    ASSERT_STATUS_OK(direct.Run(RunOptions{}, feeds, output_names, &target));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (const auto* outputs : {&expected, &actual, &target}) {
      ASSERT_EQ(outputs->size(), 1u);
      ASSERT_EQ((*outputs)[0].template Get<Tensor>().Shape(), TensorShape(dims));
      ASSERT_EQ((*outputs)[0].template Get<Tensor>().Shape().Size(), static_cast<int64_t>(count));
    }
    const auto* a = expected[0].Get<Tensor>().Data<T>();
    const auto* b = actual[0].Get<Tensor>().Data<T>();
    const auto* c = target[0].Get<Tensor>().Data<T>();
    for (size_t i = 0; i < count; ++i) {
      if (std::isnan(static_cast<float>(a[i]))) {
        ASSERT_TRUE(std::isnan(static_cast<float>(b[i]))) << i;
        ASSERT_TRUE(std::isnan(static_cast<float>(c[i]))) << i;
        ++nans;
      } else {
        ASSERT_EQ(UnitSwishBits(a[i]), UnitSwishBits(b[i])) << i;
        ASSERT_EQ(UnitSwishBits(a[i]), UnitSwishBits(c[i])) << i;
      }
    }
    elements += count;
    ++cases;
  }
  ASSERT_GT(elements, 0u);
  ASSERT_GT(nans, 0u);
  auto events = [](ProfileCleanup& cleanup) {
    cleanup.path = cleanup.session.EndProfiling();
    std::ifstream stream(cleanup.path);
    const auto profile = nlohmann::json::parse(stream);
    InlinedHashMap<std::string, size_t> counts;
    for (const auto& event : profile) {
      if (event.value("cat", "") != "Node" || !event.contains("args")) continue;
      const auto& args = event["args"];
      if (args.value("provider", "") == kCudaExecutionProvider) ++counts[args.value("op_name", "")];
    }
    return counts;
  };
  auto source_events = events(source_cleanup);
  auto fused_events = events(fused_cleanup);
  EXPECT_EQ(source_events["Sigmoid"], cases);  // Actual expanded source, not a fused proxy.
  EXPECT_EQ(source_events["Cast"], cases);
  EXPECT_EQ(source_events["Mul"], 2 * cases);
  EXPECT_EQ(source_events["ScaledSiLU"], 0u);
  EXPECT_EQ(fused_events["ScaledSiLU"], cases);
  EXPECT_EQ(fused_events["Sigmoid"], 0u);
  ::testing::Test::RecordProperty("unit_swish_cases", static_cast<int>(cases));
  ::testing::Test::RecordProperty(std::is_same_v<T, float> ? "unit_swish_float_elements" : "unit_swish_bf16_elements",
                                  static_cast<int>(elements));
  if constexpr (std::is_same_v<T, BFloat16>) {
    if (rank == 1) {
      const auto unique = std::count(observed_bf16_patterns.begin(), observed_bf16_patterns.end(), true);
      ASSERT_EQ(unique, 65536);
      ::testing::Test::RecordProperty("unit_swish_bf16_unique_vector_patterns", static_cast<int>(unique));
    }
  }
  ::testing::Test::RecordProperty("unit_swish_matching_nan_classifications", static_cast<int>(nans));
  ::testing::Test::RecordProperty("unit_swish_cuda_target_events", static_cast<int>(cases));
  ::testing::Test::RecordProperty("unit_swish_cuda_source_cast_events", static_cast<int>(source_events["Cast"]));
  ::testing::Test::RecordProperty("unit_swish_cuda_source_mul_events", static_cast<int>(source_events["Mul"]));
  ::testing::Test::RecordProperty("unit_swish_cuda_source_sigmoid_events", static_cast<int>(source_events["Sigmoid"]));
}

TEST(CudaRuntimeFusionTest, UnitSwishScalarFloatBitwise) {
  RunUnitSwishNumerics(0, TransformerLevel::Level2);
}

TEST(CudaRuntimeFusionTest, UnitSwishDynamicOddEmptyFloatBitwise) {
  RunUnitSwishNumerics(1, TransformerLevel::Level2);
}

TEST(CudaRuntimeFusionTest, UnitSwishModelRowsFloatBitwiseDefaultOptimization) {
  RunUnitSwishNumerics(2, std::nullopt);
}

TEST(CudaRuntimeFusionTest, UnitSwishScalarBFloat16Bitwise) {
  if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  RunUnitSwishNumerics<BFloat16>(0, TransformerLevel::Level2);
}

TEST(CudaRuntimeFusionTest, UnitSwishAllBFloat16PatternsDynamicOddEmptyBitwise) {
  if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  RunUnitSwishNumerics<BFloat16>(1, TransformerLevel::Level2);
}

TEST(CudaRuntimeFusionTest, UnitSwishModelRowsBFloat16BitwiseDefaultOptimization) {
  if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  RunUnitSwishNumerics<BFloat16>(2, std::nullopt);
}

TEST(CudaRuntimeFusionTest, UnitSwishRank3ModelRowsBFloat16BitwiseDefaultOptimization) {
  if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  RunUnitSwishNumerics<BFloat16>(3, std::nullopt);
}

TEST(CudaRuntimeFusionTest, UnitSwishDisabledAndBasicAndMalformedPolicy) {
  const auto bytes = MakeUnitSwishModel(1).SerializeAsString();
  for (const char* flag : {"0", "1", "true"}) {
    for (auto level : {TransformerLevel::Level1, TransformerLevel::Level2}) {
      SessionOptions options;
      options.graph_optimization_level = level;
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kUnitSwishConfig, flag));
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
      InferenceSessionWrapper session(options, GetEnvironment());
      ASSERT_STATUS_OK(session.RegisterExecutionProvider(ProductionMathCudaExecutionProvider()));
      ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
      const auto status = session.Initialize();
      if (std::string_view(flag) == "true") {
        EXPECT_FALSE(status.IsOK());
        EXPECT_NE(status.ErrorMessage().find("must be exactly 0 or 1"), std::string::npos);
      } else {
        ASSERT_STATUS_OK(status);
        bool target = false;
        for (const auto& node : session.GetGraph().Nodes()) target |= node.OpType() == "ScaledSiLU";
        EXPECT_EQ(target, std::string_view(flag) == "1" && level == TransformerLevel::Level2);
      }
    }
  }
}

TEST(CudaRuntimeFusionTest, UnitSwishLargerCausalOfferReservesUnaryNode) {
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kCausalConvConfig, "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kUnitSwishConfig, "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(ProductionMathCudaExecutionProvider()));
  const auto bytes = MakeMobiusModel<float>(false, 3, 2).SerializeAsString();
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());
  ASSERT_EQ(session.GetGraph().NumberOfNodes(), 1);
  EXPECT_EQ(session.GetGraph().Nodes().begin()->OpType(), "CausalConvWithState");
  std::vector<OrtValue> outputs;
  const std::vector<std::string> output_names{"y", "state"};
  ASSERT_STATUS_OK(session.Run(RunOptions{}, MakeFeeds<float>(17, 1, 3, 1), output_names, &outputs));
  EXPECT_EQ(outputs.size(), 2u);
}

TEST(CudaRuntimeFusionTest, UnitSwishBFloat16LargerCausalOfferReservesUnaryNode) {
  if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kCausalConvConfig, "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kUnitSwishConfig, "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(ProductionMathCudaExecutionProvider()));
  const auto bytes = MakeMobiusModel<BFloat16>(false, 3, 2).SerializeAsString();
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());
  ASSERT_EQ(session.GetGraph().NumberOfNodes(), 1);
  EXPECT_EQ(session.GetGraph().Nodes().begin()->OpType(), "CausalConvWithState");
  EXPECT_EQ(session.GetGraph().Nodes().begin()->GetExecutionProviderType(), kCudaExecutionProvider);
  std::vector<OrtValue> outputs;
  const std::vector<std::string> output_names{"y", "state"};
  ASSERT_STATUS_OK(session.Run(RunOptions{}, MakeFeeds<BFloat16>(17, 1, 3, 1), output_names, &outputs));
  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_EQ(outputs[0].Get<Tensor>().Shape(), TensorShape({1, 3, 17}));
  EXPECT_EQ(outputs[1].Get<Tensor>().Shape(), TensorShape({1, 3, 1}));
}

TEST(CudaRuntimeFusionTest, UnitSwishResourceAndUnsupportedConsumerDoNotProbeTarget) {
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kUnitSwishConfig, "1"));
  InferenceSessionWrapper session(options, GetEnvironment());
  auto cuda = ProductionMathCudaExecutionProvider();
  auto* provider = cuda.get();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda)));
  const auto bytes = MakeUnitSwishModel(1).SerializeAsString();
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  struct Lookup final : IExecutionProvider::IKernelLookup {
    mutable size_t target_calls = 0;
    const KernelCreateInfo* LookUpKernel(const Node& node) const override {
      if (node.OpType() == "ScaledSiLU") ++target_calls;
      return nullptr;
    }
  } lookup;
  struct Accountant final : IResourceAccountant {
    ResourceCount GetConsumedAmount() const override { return {}; }
    void AddConsumedAmount(const ResourceCount&) override {}
    void RemoveConsumedAmount(const ResourceCount&) override {}
    ResourceCount ComputeResourceCount(const Node&, std::optional<Level1MemoryEstimate>) override { return {}; }
  } accountant;
  GraphOptimizerRegistry registry(&options, TestCPUExecutionProvider(), &DefaultLoggingManager().DefaultLogger());
  const auto offers = provider->GetCapability(GraphViewer(session.GetGraph()), lookup, registry, &accountant);
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.target_calls, 0u);
  GraphOptimizerRegistry unsupported(&options, TestCPUExecutionProvider(),
                                     &DefaultLoggingManager().DefaultLogger(), false);
  const auto declined = provider->GetCapability(GraphViewer(session.GetGraph()), lookup, unsupported, nullptr);
  EXPECT_TRUE(declined.empty());
  EXPECT_EQ(lookup.target_calls, 0u);
}

TEST(CudaRuntimeFusionTest, UnitSwishOrtLoadFileAndAutodetectedBufferCompatibility) {
  const auto directory = std::filesystem::temp_directory_path() /
                         MakeString("unit_swish_ort_", Env::Default().NowMicros());
  TemporaryDirectory owned_directory(directory.native(), false);
  for (bool target_artifact : {false, true}) {
    SCOPED_TRACE(target_artifact);
    const auto path = directory / (target_artifact ? "target.ort" : "source.ort");
    SessionOptions options;
    options.optimized_model_filepath = path.native();
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.save_model_format", "ORT"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
    InferenceSessionWrapper saver(options, GetEnvironment());
    ASSERT_STATUS_OK(saver.RegisterExecutionProvider(ProductionMathCudaExecutionProvider()));
    const auto bytes = MakeUnitSwishModel(1, target_artifact).SerializeAsString();
    ASSERT_STATUS_OK(saver.Load(bytes.data(), static_cast<int>(bytes.size())));
    ASSERT_STATUS_OK(saver.Initialize());
    NameMLValMap feeds;
    auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
    CreateMLValue<float>(allocator, {5}, {-1.0f, -0.0f, 0.0f, 1.0f, 17.0f}, &feeds["x"]);
    std::vector<OrtValue> expected;
    const std::vector<std::string> output_names{"y"};
    ASSERT_STATUS_OK(saver.Run(RunOptions{}, feeds, output_names, &expected));
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> ort_bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(ort_bytes.empty());
    file.close();
    for (bool buffer_load : {false, true}) {
      SessionOptions load_options;
      ASSERT_STATUS_OK(load_options.config_options.AddConfigEntry(cuda_runtime_fusion::kUnitSwishConfig, "1"));
      ASSERT_STATUS_OK(load_options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
      InferenceSessionWrapper loaded(load_options, GetEnvironment());
      ASSERT_STATUS_OK(loaded.RegisterExecutionProvider(ProductionMathCudaExecutionProvider()));
      if (buffer_load) {
        ASSERT_STATUS_OK(loaded.Load(ort_bytes.data(), static_cast<int>(ort_bytes.size())));
      } else {
        ASSERT_STATUS_OK(loaded.Load(path.native()));
      }
      ASSERT_STATUS_OK(loaded.Initialize());
      bool target = false;
      for (const auto& node : loaded.GetGraph().Nodes()) {
        ASSERT_NE(node.OpType(), "Swish");
        ASSERT_EQ(node.GetExecutionProviderType(), kCudaExecutionProvider);
        target |= node.OpType() == "ScaledSiLU";
      }
      EXPECT_EQ(target, target_artifact);
      std::vector<OrtValue> actual;
      ASSERT_STATUS_OK(loaded.Run(RunOptions{}, feeds, output_names, &actual));
      ASSERT_EQ(actual.size(), 1u);
      ASSERT_EQ(actual[0].Get<Tensor>().Shape(), expected[0].Get<Tensor>().Shape());
      EXPECT_EQ(std::memcmp(actual[0].Get<Tensor>().DataRaw(), expected[0].Get<Tensor>().DataRaw(),
                            expected[0].Get<Tensor>().SizeInBytes()),
                0);
    }
  }
}

#ifdef ENABLE_CUDA_NHWC_OPS
TEST(CudaRuntimeFusionTest, UnitSwishNhwcPreferenceDoesNotProbeTarget) {
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kUnitSwishConfig, "1"));
  InferenceSessionWrapper session(options, GetEnvironment());
  auto cuda = DefaultCudaNHWCExecutionProvider();
  auto* provider = cuda.get();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda)));
  const auto bytes = MakeUnitSwishModel(1).SerializeAsString();
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  struct Lookup final : IExecutionProvider::IKernelLookup {
    mutable size_t target_calls = 0;
    const KernelCreateInfo* LookUpKernel(const Node& node) const override {
      if (node.OpType() == "ScaledSiLU") ++target_calls;
      return nullptr;
    }
  } lookup;
  GraphOptimizerRegistry registry(&options, TestCPUExecutionProvider(), &DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(provider->GetCapability(GraphViewer(session.GetGraph()), lookup, registry, nullptr).empty());
  EXPECT_EQ(lookup.target_calls, 0u);
}
#endif

template <typename T = BFloat16>
RoundingBudget Budget(double convolution, double bias, double sum_abs, int64_t kernel = 4,
                      double operand_error = 0) {
  constexpr double u = 0x1p-24;  // FP32 unit roundoff.
  const double n = static_cast<double>(2 * kernel + 1);
  const double gamma = n * u / (1 - n * u);
  const double a = convolution + bias;
  const double y = Silu(a);
  // Level1 can fold Add into Conv before discovery. Include bias in the accumulation
  // bound to allow either epilogue addition or accumulation with bias already present.
  // Separate Conv/Add element roundings below conservatively cover the unfolded path too.
  const double accumulation = operand_error + gamma * (sum_abs + operand_error + std::abs(bias));
  const double conv_error = accumulation + HalfUlp<T>(convolution, accumulation);
  const double add_error = conv_error + u * (std::abs(convolution) + std::abs(bias) + conv_error);
  const double a_error = add_error + HalfUlp<T>(a, add_error);
  const double sigmoid = 1 / (1 + std::exp(-a));
  // CUDA OP_Sigmoid computes in FLOAT then casts to T. Retain the historical
  // conservative causal-region bound, including extra intermediate rounding
  // allowances; this is not the unit-Swish bitwise qualification gate.
  const double exponential = std::exp(-std::abs(a));
  const double exp_float_error = a_error + 8 * u;  // expf <=2 ULP on bounded inputs.
  // The BF16 allowances below overestimate native intermediate rounding.
  using Intermediate = std::conditional_t<std::is_same_v<T, MLFloat16>, float, T>;
  const double exp_error = exp_float_error + HalfUlp<Intermediate>(exponential, exp_float_error);
  const double denominator = 1 + exponential;
  const double denominator_float_error = exp_error + u * (denominator + exp_error);
  const double denominator_error = denominator_float_error +
                                   HalfUlp<Intermediate>(denominator, denominator_float_error);
  const double reciprocal = 1 / denominator;
  const double division_error = denominator_error / (denominator * std::max(1.0, denominator - denominator_error));
  const double reciprocal_float_error = division_error + u * (reciprocal + division_error);
  const double reciprocal_error = reciprocal_float_error + HalfUlp<Intermediate>(reciprocal, reciprocal_float_error);
  double sigmoid_error = reciprocal_error;
  if (a - a_error <= 0) {
    const double subtraction_error = reciprocal_error + u * (1 + reciprocal + reciprocal_error);
    sigmoid_error = subtraction_error + HalfUlp<Intermediate>(1 - reciprocal, subtraction_error);
  }
  // If the a interval straddles zero, either branch is possible; their ideal
  // values differ by <= |a|/2 <= a_error/2.
  if (std::abs(a) <= a_error) sigmoid_error += 0.5 * a_error;
  if constexpr (std::is_same_v<T, MLFloat16>) sigmoid_error += HalfUlp<T>(sigmoid, sigmoid_error);
  const double product_error = sigmoid * a_error + std::abs(a) * sigmoid_error + a_error * sigmoid_error;
  const double source_error = product_error + HalfUlp<T>(y, product_error);
  // The target accumulates bias + K products in FP32, then performs one SiLU
  // and one element rounding. Global |SiLU'| < 1.1; no empirical tolerance fitting.
  const double target_sum_error = gamma * (sum_abs + std::abs(bias));
  const double target_float_error = 1.1 * target_sum_error + 8 * u * (std::abs(a) + 1);
  return {source_error, target_float_error + HalfUlp<T>(y, target_float_error)};
}

TEST(CudaRuntimeFusionTest, AnalyticalBudgetUsesDtypeKernelAndOuterBinade) {
  EXPECT_EQ(HalfUlp<float>(0), 0x1p-150);
  EXPECT_EQ(HalfUlp<MLFloat16>(0), 0x1p-25);
  EXPECT_EQ(HalfUlp<BFloat16>(0), 0x1p-134);
  EXPECT_EQ(HalfUlp<float>(1 - 0x1p-24, 0x1p-24), 0x1p-24);
  EXPECT_EQ(HalfUlp<MLFloat16>(1 - 0x1p-11, 0x1p-11), 0x1p-11);
  EXPECT_EQ(HalfUlp<BFloat16>(1 - 0x1p-8, 0x1p-8), 0x1p-8);
  EXPECT_GT(Budget<float>(0.25, -0.125, 1, 6).target, Budget<float>(0.25, -0.125, 1, 1).target);
  EXPECT_GT(Budget<MLFloat16>(0.25, -0.125, 1, 6).target, Budget<float>(0.25, -0.125, 1, 6).target);
  EXPECT_GT(Budget<BFloat16>(0.25, -0.125, 1, 6).target, Budget<MLFloat16>(0.25, -0.125, 1, 6).target);
}

TEST(CudaRuntimeFusionTest, NativeExtendedSharedPrefillCudaGraphReplay) {
  if (!HasCudaEnvironment(0)) GTEST_SKIP() << "CUDA is required.";
  int device = 0, default_bytes = 0, optin_bytes = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaDeviceGetAttribute(&default_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device), cudaSuccess);
  ASSERT_EQ(cudaDeviceGetAttribute(&optin_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device), cudaSuccess);
  if (optin_bytes <= default_bytes) GTEST_SKIP() << "Extended shared memory is required.";
  constexpr int64_t channels = 35, kernel = 3, pad = kernel - 1;
  const int64_t length = default_bytes / static_cast<int>(sizeof(float)) - pad - kernel + 1;
  const auto model = MakeMobiusModel<float>(true, channels, kernel).SerializeAsString();
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  InferenceSessionWrapper session(options, GetEnvironment());
  OrtCUDAProviderOptionsV2 cuda_options{};
  cuda_options.enable_cuda_graph = true;
  auto ep = CudaExecutionProviderWithOptions(&cuda_options);
  auto* provider = ep.get();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(ep)));
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());
  const auto allocators = provider->CreatePreferredAllocators();
  const auto gpu = std::find_if(allocators.begin(), allocators.end(), [](const auto& allocator) {
    return allocator->Info().device.Type() == OrtDevice::GPU &&
           allocator->Info().alloc_type != OrtAllocatorType::OrtReadOnlyAllocator &&
           allocator->Info().mem_type == OrtMemTypeDefault;
  });
  ASSERT_NE(gpu, allocators.end());
  const auto allocator = session.GetAllocator((*gpu)->Info());
  ASSERT_NE(allocator, nullptr);
  const auto cpu = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  auto allocate = [&](const TensorShape& shape) {
    OrtValue value;
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), shape, allocator, value);
    return value;
  };
  const TensorShape shape{1, channels, length}, history{1, channels, pad};
  auto x = allocate(shape), h = allocate(history), y = allocate(shape), state = allocate(history);
  std::vector<float> input(static_cast<size_t>(shape.Size()), 0);
  std::vector<float> past(static_cast<size_t>(history.Size()), 0);
  auto copy_to_gpu = [&](std::vector<float>& data, OrtValue& value) {
    Tensor source(DataTypeImpl::GetType<float>(), value.Get<Tensor>().Shape(), data.data(), cpu->Info());
    ASSERT_STATUS_OK(provider->GetDataTransfer()->CopyTensor(source, *value.GetMutable<Tensor>()));
  };
  copy_to_gpu(input, x);
  copy_to_gpu(past, h);
  std::unique_ptr<IOBinding> binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&binding));
  ASSERT_STATUS_OK(binding->BindInput("x", x));
  ASSERT_STATUS_OK(binding->BindInput("h", h));
  ASSERT_STATUS_OK(binding->BindOutput("y", y));
  ASSERT_STATUS_OK(binding->BindOutput("state", state));
  ASSERT_STATUS_OK(binding->SynchronizeInputs());
  RunOptions uncaptured, captured;
  ASSERT_STATUS_OK(uncaptured.config_options.AddConfigEntry("gpu_graph_id", "-1"));
  ASSERT_STATUS_OK(captured.config_options.AddConfigEntry("gpu_graph_id", "1"));
  Tensor expected(DataTypeImpl::GetType<float>(), shape, cpu);
  Tensor actual(DataTypeImpl::GetType<float>(), shape, cpu);
  Tensor actual_state(DataTypeImpl::GetType<float>(), history, cpu);
  for (float fill : {0.0f, 0.25f}) {
    std::fill(input.begin(), input.end(), fill);
    copy_to_gpu(input, x);
    // Warm the function's fixed opt-in capacity outside capture, then compare actual replay.
    ASSERT_STATUS_OK(session.Run(uncaptured, *binding));
    ASSERT_STATUS_OK(provider->GetDataTransfer()->CopyTensor(y.Get<Tensor>(), expected));
    for (int repeat = 0; repeat < 4; ++repeat) {
      ASSERT_STATUS_OK(session.Run(captured, *binding));
    }
    ASSERT_TRUE(provider->IsGraphCaptured(1));
    ASSERT_STATUS_OK(binding->SynchronizeOutputs());
    ASSERT_STATUS_OK(provider->GetDataTransfer()->CopyTensor(y.Get<Tensor>(), actual));
    ASSERT_STATUS_OK(provider->GetDataTransfer()->CopyTensor(state.Get<Tensor>(), actual_state));
    EXPECT_EQ(std::memcmp(expected.DataRaw(), actual.DataRaw(), expected.SizeInBytes()), 0);
    for (float value : actual_state.DataAsSpan<float>()) EXPECT_EQ(value, fill);
  }
}

template <typename T = BFloat16>
std::vector<T> InitializerData(const ONNX_NAMESPACE::TensorProto& tensor) {
  std::vector<T> result(tensor.raw_data().size() / sizeof(T));
  std::memcpy(result.data(), tensor.raw_data().data(), tensor.raw_data().size());
  return result;
}

template <typename T = BFloat16>
void CheckNumerics(const ONNX_NAMESPACE::ModelProto& model, const NameMLValMap& feeds,
                   const std::vector<OrtValue>& source, const std::vector<OrtValue>& fused,
                   const std::vector<OrtValue>& direct, int64_t batch, int64_t length, int step = 0) {
  const auto weights = InitializerData<T>(model.graph().initializer(0));
  const auto bias = InitializerData<T>(model.graph().initializer(1));
  const int64_t channels = model.graph().initializer(0).dims(0);
  const int64_t kernel = model.graph().initializer(0).dims(2);
  const int64_t pad = feeds.at("h").Get<Tensor>().Shape()[2];
  int64_t dilation = 1;
  for (const auto& node : model.graph().node()) {
    if (node.op_type() != "Conv") continue;
    for (const auto& attr : node.attribute()) {
      if (attr.name() == "dilations") {
        ASSERT_EQ(attr.ints_size(), 1);
        dilation = attr.ints(0);
      }
    }
  }
  const auto* x = feeds.at("x").Get<Tensor>().Data<T>();
  const auto* h = feeds.at("h").Get<Tensor>().Data<T>();
  auto concatenated = [&](int64_t b, int64_t c, int64_t t) -> T {
    return t < pad ? h[(b * channels + c) * pad + t] : x[(b * channels + c) * length + t - pad];
  };
  for (const std::vector<OrtValue>* outputs : {&source, &fused, &direct}) {
    ASSERT_EQ(outputs->size(), 2u);
    EXPECT_EQ((*outputs)[0].Get<Tensor>().Shape(), (TensorShape{batch, channels, length}));
    EXPECT_EQ((*outputs)[1].Get<Tensor>().Shape(), (TensorShape{batch, channels, pad}));
  }
  for (int output : {0, 1}) {
    const auto& a = fused[output].Get<Tensor>();
    const auto& b = direct[output].Get<Tensor>();
    ASSERT_EQ(a.SizeInBytes(), b.SizeInBytes());
    if (a.SizeInBytes()) {
      ASSERT_EQ(std::memcmp(a.DataRaw(), b.DataRaw(), a.SizeInBytes()), 0);
    }
  }
  const auto* source_y = source[0].Get<Tensor>().Data<T>();
  const auto* fused_y = fused[0].Get<Tensor>().Data<T>();
  double max_source_error = 0, max_target_error = 0;
  double max_source_fraction = 0, max_target_fraction = 0;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t t = 0; t < length; ++t) {
        double convolution = 0, sum_abs = 0, operand_error = 0;
        for (int64_t k = 0; k < kernel; ++k) {
          const double sample = static_cast<float>(concatenated(b, c, t + k * dilation));
          const double weight = static_cast<float>(weights[static_cast<size_t>(c * kernel + k)]);
          const double product = sample * weight;
          convolution += product;
          sum_abs += std::abs(product);
          if constexpr (std::is_same_v<T, float>) {
            // The provider permits TF32 by default. Allow either rounding or truncation
            // to its 10 fraction bits, and possible flushing of FP32 subnormal operands.
            const double dx = std::abs(sample) * 0x1p-10 + 0x1p-126;
            const double dw = std::abs(weight) * 0x1p-10 + 0x1p-126;
            operand_error += std::abs(sample) * dw + std::abs(weight) * dx + dx * dw;
          }
        }
        const double bias_value = static_cast<float>(bias[static_cast<size_t>(c)]);
        const double reference = Silu(convolution + bias_value);
        const auto budget = Budget<T>(convolution, bias_value, sum_abs, kernel, operand_error);
        const size_t index = static_cast<size_t>((b * channels + c) * length + t);
        const double source_error = std::abs(static_cast<float>(source_y[index]) - reference);
        const double target_error = std::abs(static_cast<float>(fused_y[index]) - reference);
        max_source_error = std::max(max_source_error, source_error);
        max_target_error = std::max(max_target_error, target_error);
        max_source_fraction = std::max(max_source_fraction, source_error / budget.source);
        max_target_fraction = std::max(max_target_fraction, target_error / budget.target);
        ASSERT_LE(source_error, budget.source) << index;
        ASSERT_LE(target_error, budget.target) << index;
        ASSERT_LE(std::abs(static_cast<float>(source_y[index]) - static_cast<float>(fused_y[index])),
                  budget.source + budget.target)
            << index;
      }
      for (int64_t t = 0; t < pad; ++t) {
        const auto expected = concatenated(b, c, length + t);
        const size_t index = static_cast<size_t>((b * channels + c) * pad + t);
        for (const std::vector<OrtValue>* outputs : {&source, &fused, &direct}) {
          ASSERT_EQ(std::memcmp(&(*outputs)[1].Get<Tensor>().Data<T>()[index], &expected, sizeof(T)), 0) << index;
        }
      }
    }
    const auto suffix = MakeString("_T", model.graph().initializer(0).data_type(), "_C", channels,
                                   "_K", kernel, "_D", dilation, "_B", batch, "_S", length, "_step", step);
    ::testing::Test::RecordProperty("source_max_abs_error" + suffix, MakeString(max_source_error));
    ::testing::Test::RecordProperty("target_max_abs_error" + suffix, MakeString(max_target_error));
    ::testing::Test::RecordProperty("source_max_budget_fraction" + suffix, MakeString(max_source_fraction));
    ::testing::Test::RecordProperty("target_max_budget_fraction" + suffix, MakeString(max_target_fraction));
  }
}

template <typename T = BFloat16>
void PositiveNumericsAndCudaDispatch(std::optional<TransformerLevel> level, int64_t channels = kChannels,
                                     int64_t kernel = 4, int64_t dilation = 1) {
  if constexpr (std::is_same_v<T, BFloat16>) {
    if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  } else {
    if (!HasCudaEnvironment(0)) GTEST_SKIP() << "CUDA is required.";
  }
  const int64_t pad = (kernel - 1) * dilation;
  const auto model = MakeMobiusModel<T>(false, channels, kernel, dilation);
  const auto bytes = model.SerializeAsString();
  SessionOptions baseline_options;
  if (level) baseline_options.graph_optimization_level = *level;
  InferenceSessionWrapper source(baseline_options, GetEnvironment());
  ASSERT_STATUS_OK(source.RegisterExecutionProvider(ProductionMathCudaExecutionProvider()));
  ASSERT_STATUS_OK(source.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(source.Initialize());
  bool has_conv = false, has_sigmoid = false;
  for (const auto& node : source.GetGraph().Nodes()) {
    EXPECT_NE(node.OpType(), "CausalConvWithState");
    has_conv |= node.OpType() == "Conv" && node.GetExecutionProviderType() == kCudaExecutionProvider;
    has_sigmoid |= node.OpType() == "Sigmoid" && node.GetExecutionProviderType() == kCudaExecutionProvider;
  }
  ASSERT_TRUE(has_conv);
  ASSERT_TRUE(has_sigmoid);  // Confirms which source rounding path the budget covers.

  SessionOptions enabled = baseline_options;
  ASSERT_STATUS_OK(enabled.config_options.AddConfigEntry(cuda_runtime_fusion::kCausalConvConfig, "1"));
  ASSERT_STATUS_OK(enabled.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  enabled.enable_profiling = true;
  // A test-owned profile is read and deleted below; no checkpoint/optimized-model files are written.
  enabled.profile_file_prefix = ORT_TSTR("cuda_runtime_fusion_test");
  InferenceSessionWrapper fused(enabled, GetEnvironment());
  ProfileCleanup cleanup{fused, {}};
  ASSERT_STATUS_OK(fused.RegisterExecutionProvider(ProductionMathCudaExecutionProvider()));
  ASSERT_STATUS_OK(fused.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_EQ(fused.GetGraph().NumberOfNodes(), 11);
  ASSERT_EQ(fused.GetGraph().GetProducerNode("y")->OpType(), "Swish");
  ASSERT_STATUS_OK(fused.Initialize());
  ASSERT_EQ(fused.GetGraph().NumberOfNodes(), 1);
  const auto& target_node = *fused.GetGraph().Nodes().begin();
  ASSERT_EQ(target_node.OpType(), "CausalConvWithState");
  ASSERT_EQ(target_node.GetExecutionProviderType(), kCudaExecutionProvider);
  EXPECT_EQ(target_node.OutputDefs()[0]->Name(), "y");
  EXPECT_EQ(target_node.OutputDefs()[1]->Name(), "state");
  SessionOptions target_options = baseline_options;
  ASSERT_STATUS_OK(target_options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  InferenceSessionWrapper direct(target_options, GetEnvironment());
  ASSERT_STATUS_OK(direct.RegisterExecutionProvider(ProductionMathCudaExecutionProvider()));
  const auto target_bytes = MakeMobiusModel<T>(true, channels, kernel, dilation).SerializeAsString();
  ASSERT_STATUS_OK(direct.Load(target_bytes.data(), static_cast<int>(target_bytes.size())));
  ASSERT_STATUS_OK(direct.Initialize());
  const std::vector<std::string> output_names{"y", "state"};
  for (int64_t batch : {1, 2}) {
    OrtValue previous_state;
    int step = 0;
    for (int64_t length : {batch == 1 ? int64_t{2} : int64_t{17}, int64_t{1}, int64_t{1}}) {
      SCOPED_TRACE(MakeString("B=", batch, ", S=", length));
      auto feeds = MakeFeeds<T>(length, batch, channels, pad);
      if (previous_state.IsAllocated()) feeds["h"] = previous_state;
      std::vector<OrtValue> source_outputs, fused_outputs, direct_outputs;
      ASSERT_STATUS_OK(source.Run(RunOptions{}, feeds, output_names, &source_outputs));
      ASSERT_STATUS_OK(fused.Run(RunOptions{}, feeds, output_names, &fused_outputs));
      ASSERT_STATUS_OK(direct.Run(RunOptions{}, feeds, output_names, &direct_outputs));
      CheckNumerics<T>(model, feeds, source_outputs, fused_outputs, direct_outputs, batch, length, step++);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      previous_state = fused_outputs[1];
    }
  }
  const auto profile_path = cleanup.path = fused.EndProfiling();
  ASSERT_FALSE(profile_path.empty());
  std::ifstream profile_stream(profile_path);
  const auto profile = nlohmann::json::parse(profile_stream);
  profile_stream.close();
  size_t cuda_target_events = 0;
  for (const auto& event : profile) {
    if (event.value("cat", "") != "Node" || !event.contains("args")) continue;
    const auto& args = event["args"];
    if (args.value("op_name", "") == "CausalConvWithState" &&
        args.value("provider", "") == kCudaExecutionProvider) ++cuda_target_events;
  }
  EXPECT_EQ(cuda_target_events, 6u);
  ::testing::Test::RecordProperty("cuda_target_profile_events", static_cast<int>(cuda_target_events));
  ::testing::Test::RecordProperty(
      MakeString("C", channels, "_K", kernel, "_D", dilation, "_",
                 level ? MakeString("level", static_cast<int>(*level)) : "default",
                 "_cuda_target_profile_events"),
      static_cast<int>(cuda_target_events));
  ::testing::Test::RecordProperty("production_use_tf32", "true");
  ::testing::Test::RecordProperty("fused_graph_nodes", fused.GetGraph().NumberOfNodes());
}

void OrdinaryBehaviorAtLowOptimizationLevel(TransformerLevel level) {
  if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  SessionOptions options;
  options.graph_optimization_level = level;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kCausalConvConfig, "1"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
  const auto bytes = MakeMobiusModel(false).SerializeAsString();
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());
  bool cuda_conv = false, cuda_sigmoid = false;
  for (const auto& node : session.GetGraph().Nodes()) {
    ASSERT_NE(node.OpType(), "CausalConvWithState");
    cuda_conv |= node.OpType() == "Conv" && node.GetExecutionProviderType() == kCudaExecutionProvider;
    cuda_sigmoid |= node.OpType() == "Sigmoid" && node.GetExecutionProviderType() == kCudaExecutionProvider;
  }
  ASSERT_TRUE(cuda_conv);
  ASSERT_TRUE(cuda_sigmoid);
  std::vector<OrtValue> outputs;
  const std::vector<std::string> names{"y", "state"};
  ASSERT_STATUS_OK(session.Run(RunOptions{}, MakeFeeds(2), names, &outputs));
}

TEST(CudaRuntimeFusionTest, DisabledOptimizersLeaveOrdinaryBehavior) {
  OrdinaryBehaviorAtLowOptimizationLevel(TransformerLevel::Default);
}

TEST(CudaRuntimeFusionTest, BasicOptimizersLeaveOrdinaryBehavior) {
  OrdinaryBehaviorAtLowOptimizationLevel(TransformerLevel::Level1);
}

template <typename T>
void ContractNumericsMatrix() {
  for (auto dims : {std::array<int64_t, 3>{1, 1, 3}, {3, 2, 1}, {7, 3, 2}, {35, 6, 3}}) {
    SCOPED_TRACE(MakeString(dims[0], "/", dims[1], "/", dims[2]));
    PositiveNumericsAndCudaDispatch<T>(std::nullopt, dims[0], dims[1], dims[2]);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    if (dims[1] == 1) {
      for (auto level : {TransformerLevel::Level2, TransformerLevel::MaxLevel}) {
        SCOPED_TRACE(MakeString("K1 optimization level=", static_cast<int>(level)));
        PositiveNumericsAndCudaDispatch<T>(level, dims[0], dims[1], dims[2]);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
      }
    }
  }
}

TEST(CudaRuntimeFusionTest, ContractFloatNumericsAndChainedState) {
  ContractNumericsMatrix<float>();
}

TEST(CudaRuntimeFusionTest, ContractFloat16NumericsAndChainedState) {
  ContractNumericsMatrix<MLFloat16>();
}

TEST(CudaRuntimeFusionTest, ContractBFloat16NumericsAndChainedState) {
  ContractNumericsMatrix<BFloat16>();
}

TEST(CudaRuntimeFusionTest, ExtendedOptimizersNumericsAndCudaDispatch) {
  PositiveNumericsAndCudaDispatch(TransformerLevel::Level2);
}

TEST(CudaRuntimeFusionTest, AllOptimizersNumericsAndCudaDispatch) {
  PositiveNumericsAndCudaDispatch(TransformerLevel::MaxLevel);
}

TEST(CudaRuntimeFusionTest, InvalidConfigurationPropagatesFromCudaCapability) {
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kCausalConvConfig, "true"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
  const auto bytes = MakeMobiusModel(false).SerializeAsString();
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  const auto status = session.Initialize();
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("must be exactly 0 or 1"), std::string::npos) << status.ErrorMessage();
}

TEST(CudaRuntimeFusionTest, DefaultOptimizersNumericsAndCudaDispatch) {
  PositiveNumericsAndCudaDispatch(std::nullopt);  // Leave the default optimization policy untouched.
}

// Local qualification only: CI uses generated weights above. The caller supplies existing
// external-data fixture pairs; this test never writes their learned data or personal paths.
TEST(CudaRuntimeFusionTest, LocalAuthenticExternalFixtureQualification) {
  const auto directory = Env::Default().GetEnvironmentVar("ORT_CUDA_RUNTIME_FUSION_FIXTURE_DIR");
  if (directory.empty()) GTEST_SKIP() << "Set ORT_CUDA_RUNTIME_FUSION_FIXTURE_DIR for local authentic fixture QA.";
  if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  const std::filesystem::path root(directory);
  SessionOptions baseline_options;  // Normal, untouched optimization policy.
  InferenceSessionWrapper source(baseline_options, GetEnvironment());
  ASSERT_STATUS_OK(source.RegisterExecutionProvider(ProductionMathCudaExecutionProvider()));
  ASSERT_STATUS_OK(source.Load((root / "mobius-causal-conv-bf16.onnx").native()));
  ASSERT_EQ(source.GetGraph().NumberOfNodes(), 11);
  auto reference = MakeMobiusModel(false);
  size_t learned_headers = 0;
  for (const auto& [name, proto] : source.GetGraph().GetAllInitializedTensors()) {
    ORT_UNUSED_PARAMETER(name);
    if (proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) continue;
    const bool weight = proto->dims_size() == 3;
    ASSERT_EQ(proto->dims(0), kChannels);
    ASSERT_EQ(proto->dims_size(), weight ? 3 : 1);
    if (weight) {
      ASSERT_EQ(proto->dims(1), 1);
      ASSERT_EQ(proto->dims(2), 4);
    }
    // Payload reads belong to this independent QA oracle, never to capability discovery.
    Initializer loaded(source.GetGraph(), *proto, source.GetGraph().ModelPath(), false);
    const auto bytes = loaded.DataAsByteSpan();
    reference.mutable_graph()->mutable_initializer(weight ? 0 : 1)->set_raw_data(bytes.data(), bytes.size());
    ++learned_headers;
  }
  ASSERT_EQ(learned_headers, 2u);

  SessionOptions enabled;
  ASSERT_STATUS_OK(enabled.config_options.AddConfigEntry(cuda_runtime_fusion::kCausalConvConfig, "1"));
  ASSERT_STATUS_OK(enabled.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  enabled.enable_profiling = true;
  enabled.profile_file_prefix = ORT_TSTR("cuda_runtime_fusion_authentic_test");
  InferenceSessionWrapper fused(enabled, GetEnvironment());
  ProfileCleanup cleanup{fused, {}};
  ASSERT_STATUS_OK(fused.RegisterExecutionProvider(ProductionMathCudaExecutionProvider()));
  ASSERT_STATUS_OK(fused.Load((root / "mobius-causal-conv-bf16.onnx").native()));
  ASSERT_EQ(fused.GetGraph().NumberOfNodes(), 11);
  ASSERT_STATUS_OK(fused.Initialize());
  ASSERT_EQ(fused.GetGraph().NumberOfNodes(), 1);
  const auto& target = *fused.GetGraph().Nodes().begin();
  ASSERT_EQ(target.OpType(), "CausalConvWithState");
  ASSERT_EQ(target.GetExecutionProviderType(), kCudaExecutionProvider);
  ASSERT_EQ(target.InputDefs()[2]->Shape()->dim_size(), 1);

  SessionOptions direct_options;
  ASSERT_STATUS_OK(direct_options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  InferenceSessionWrapper direct(direct_options, GetEnvironment());
  ASSERT_STATUS_OK(direct.RegisterExecutionProvider(ProductionMathCudaExecutionProvider()));
  ASSERT_STATUS_OK(direct.Load((root / "modelbuilder-causal-conv-bf16.onnx").native()));
  ASSERT_EQ(direct.GetGraph().NumberOfNodes(), 1);
  const auto& direct_target = *direct.GetGraph().Nodes().begin();
  ASSERT_EQ(direct_target.OpType(), "CausalConvWithState");
  for (int input : {1, 2}) {
    const auto* proto = direct.GetGraph().GetConstantInitializer(direct_target.InputDefs()[input]->Name(), false);
    ASSERT_NE(proto, nullptr);
    Initializer loaded(direct.GetGraph(), *proto, direct.GetGraph().ModelPath(), false);
    const auto bytes = loaded.DataAsByteSpan();
    const auto& expected = reference.graph().initializer(input - 1).raw_data();
    ASSERT_EQ(bytes.size(), expected.size());
    ASSERT_EQ(std::memcmp(bytes.data(), expected.data(), bytes.size()), 0);
  }
  ASSERT_STATUS_OK(direct.Initialize());
  ASSERT_EQ(direct.GetGraph().Nodes().begin()->GetExecutionProviderType(), kCudaExecutionProvider);
  ASSERT_STATUS_OK(source.Initialize());
  bool source_cuda_conv = false, source_cuda_sigmoid = false;
  for (const auto& node : source.GetGraph().Nodes()) {
    ASSERT_NE(node.OpType(), "CausalConvWithState");
    source_cuda_conv |= node.OpType() == "Conv" && node.GetExecutionProviderType() == kCudaExecutionProvider;
    source_cuda_sigmoid |= node.OpType() == "Sigmoid" && node.GetExecutionProviderType() == kCudaExecutionProvider;
  }
  ASSERT_TRUE(source_cuda_conv);
  ASSERT_TRUE(source_cuda_sigmoid);
  auto run = [](InferenceSessionWrapper& session, const NameMLValMap& canonical, std::vector<OrtValue>& outputs) {
    const auto& graph = session.GetGraph();
    NameMLValMap feeds{{graph.GetInputs()[0]->Name(), canonical.at("x")},
                       {graph.GetInputs()[1]->Name(), canonical.at("h")}};
    const std::vector<std::string> names{graph.GetOutputs()[0]->Name(), graph.GetOutputs()[1]->Name()};
    return session.Run(RunOptions{}, feeds, names, &outputs);
  };
  for (int64_t batch : {1, 2}) {
    for (int64_t length : {1, 2, 17}) {
      SCOPED_TRACE(MakeString("authentic B=", batch, ", S=", length));
      const auto feeds = MakeFeeds(length, batch);
      std::vector<OrtValue> source_outputs, fused_outputs, direct_outputs;
      ASSERT_STATUS_OK(run(source, feeds, source_outputs));
      ASSERT_STATUS_OK(run(fused, feeds, fused_outputs));
      ASSERT_STATUS_OK(run(direct, feeds, direct_outputs));
      CheckNumerics(reference, feeds, source_outputs, fused_outputs, direct_outputs, batch, length);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
    }
  }
  cleanup.path = fused.EndProfiling();
  std::ifstream stream(cleanup.path);
  const auto profile = nlohmann::json::parse(stream);
  stream.close();
  int cuda_events = 0;
  for (const auto& event : profile) {
    if (event.value("cat", "") != "Node" || !event.contains("args")) continue;
    const auto& args = event["args"];
    if (args.value("op_name", "") == "CausalConvWithState" &&
        args.value("provider", "") == kCudaExecutionProvider) ++cuda_events;
  }
  EXPECT_EQ(cuda_events, 6);
  RecordProperty("authentic_cuda_target_profile_events", cuda_events);
  RecordProperty("authentic_fused_graph_nodes", fused.GetGraph().NumberOfNodes());
  RecordProperty("source_and_modelbuilder_learned_bits_equal", "true");
}

TEST(CudaRuntimeFusionTest, NonUnitAlphaFallsBackThroughFunctionExpansion) {
  if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  auto model = MakeMobiusModel(false);
  auto* swish = model.mutable_graph()->mutable_node(model.graph().node_size() - 1);
  auto* alpha = swish->add_attribute();
  alpha->set_name("alpha");
  alpha->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  alpha->set_f(0.5f);
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kCausalConvConfig, "1"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
  const auto bytes = model.SerializeAsString();
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());
  bool sigmoid = false;
  for (const auto& node : session.GetGraph().Nodes()) {
    EXPECT_NE(node.OpType(), "CausalConvWithState");
    EXPECT_NE(node.OpType(), "Swish");
    sigmoid |= node.OpType() == "Sigmoid";
  }
  EXPECT_TRUE(sigmoid);
  std::vector<OrtValue> outputs;
  const std::vector<std::string> names{"y", "state"};
  ASSERT_STATUS_OK(session.Run(RunOptions{}, MakeFeeds(2), names, &outputs));
  EXPECT_EQ(outputs.size(), 2u);
}

TEST(CudaRuntimeFusionTest, OrtFormatOrdinaryFileAndAutodetectedBufferCompatibility) {
  if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  const auto directory = std::filesystem::temp_directory_path() /
                         MakeString("cuda_runtime_fusion_ort_", Env::Default().NowMicros());
  TemporaryDirectory owned_directory(directory.native(), false);
  for (bool target_artifact : {false, true}) {
    SCOPED_TRACE(target_artifact);
    const auto path = directory / (target_artifact ? "target.ort" : "source.ort");
    SessionOptions save_options;
    save_options.optimized_model_filepath = path.native();
    ASSERT_STATUS_OK(save_options.config_options.AddConfigEntry("session.save_model_format", "ORT"));
    InferenceSessionWrapper saver(save_options, GetEnvironment());
    ASSERT_STATUS_OK(saver.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    const auto bytes = MakeMobiusModel(target_artifact).SerializeAsString();
    ASSERT_STATUS_OK(saver.Load(bytes.data(), static_cast<int>(bytes.size())));
    ASSERT_STATUS_OK(saver.Initialize());
    for (const auto& node : saver.GetGraph().Nodes()) {
      // Normal source serialization expands Swish, so these compatibility cases alone
      // do not exercise an eligible match. The host lookup-spy test covers that boundary.
      ASSERT_NE(node.OpType(), "Swish");
    }
    const auto feeds = MakeFeeds(2);
    const std::vector<std::string> names{"y", "state"};
    std::vector<OrtValue> expected;
    ASSERT_STATUS_OK(saver.Run(RunOptions{}, feeds, names, &expected));
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> ort_bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(ort_bytes.empty());
    file.close();
    for (const char* setting : {"0", "1"}) {
      for (bool buffer_load : {false, true}) {
        SCOPED_TRACE(MakeString("setting=", setting, ", buffer=", buffer_load));
        SessionOptions load_options;
        ASSERT_STATUS_OK(load_options.config_options.AddConfigEntry(cuda_runtime_fusion::kCausalConvConfig, setting));
        if (target_artifact) {
          ASSERT_STATUS_OK(load_options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
        }
        InferenceSessionWrapper loaded(load_options, GetEnvironment());
        ASSERT_STATUS_OK(loaded.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
        // No load-format hint: exercise the actual ORT file and byte autodetection paths.
        if (buffer_load) {
          ASSERT_STATUS_OK(loaded.Load(ort_bytes.data(), static_cast<int>(ort_bytes.size())));
        } else {
          ASSERT_STATUS_OK(loaded.Load(path.native()));
        }
        ASSERT_STATUS_OK(loaded.Initialize());
        bool cuda_conv = false;
        for (const auto& node : loaded.GetGraph().Nodes()) {
          ASSERT_NE(node.OpType(), "Swish");
          cuda_conv |= node.OpType() == "Conv" && node.GetExecutionProviderType() == kCudaExecutionProvider;
          if (!target_artifact) {
            ASSERT_NE(node.OpType(), "CausalConvWithState");
          }
        }
        if (target_artifact) {
          ASSERT_EQ(loaded.GetGraph().NumberOfNodes(), 1);
          ASSERT_EQ(loaded.GetGraph().Nodes().begin()->OpType(), "CausalConvWithState");
          ASSERT_EQ(loaded.GetGraph().Nodes().begin()->GetExecutionProviderType(), kCudaExecutionProvider);
        } else {
          ASSERT_TRUE(cuda_conv);
        }
        std::vector<OrtValue> actual;
        ASSERT_STATUS_OK(loaded.Run(RunOptions{}, feeds, names, &actual));
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t output = 0; output < actual.size(); ++output) {
          const auto& a = actual[output].Get<Tensor>();
          const auto& e = expected[output].Get<Tensor>();
          ASSERT_EQ(a.Shape(), e.Shape());
          ASSERT_EQ(a.SizeInBytes(), e.SizeInBytes());
          EXPECT_EQ(std::memcmp(a.DataRaw(), e.DataRaw(), e.SizeInBytes()), 0);
        }
      }
    }
  }
}

// Positive qualification for the dynamic shapes that previously exposed target launch failures.
// Large cases remain opt-in, with sessions/feeds scoped to release CUDA arenas between runs.
enum class ShapeProbeMode { Source,
                            Fused,
                            Direct };

void CheckLongSequenceSamples(ShapeProbeMode mode, const ONNX_NAMESPACE::ModelProto& model,
                              const NameMLValMap& feeds, const std::vector<OrtValue>& outputs,
                              int64_t length, const std::string& prefix) {
  const auto weights = InitializerData(model.graph().initializer(0));
  const auto bias = InitializerData(model.graph().initializer(1));
  const auto* x = feeds.at("x").Get<Tensor>().Data<BFloat16>();
  const auto* h = feeds.at("h").Get<Tensor>().Data<BFloat16>();
  const auto* y = outputs[0].Get<Tensor>().Data<BFloat16>();
  auto sample = [&](int64_t c, int64_t t) {
    return t < 3 ? h[c * 3 + t] : x[c * length + t - 3];
  };
  double max_error = 0, max_fraction = 0;
  for (int64_t c : {0, 1, 2, 31, 255, 1023, 5119, 10239}) {
    for (int64_t t : {0, 1, 2, 3, 4, 17, 257, 32768, 65534, 65535}) {
      double convolution = 0, sum_abs = 0;
      for (int64_t k = 0; k < 4; ++k) {
        const double product = static_cast<double>(sample(c, t + k).ToFloat()) *
                               weights[static_cast<size_t>(c * 4 + k)].ToFloat();
        convolution += product;
        sum_abs += std::abs(product);
      }
      const auto rounding = Budget(convolution, bias[static_cast<size_t>(c)].ToFloat(), sum_abs);
      const double budget = mode == ShapeProbeMode::Source ? rounding.source : rounding.target;
      const double error = std::abs(y[c * length + t].ToFloat() -
                                    Silu(convolution + bias[static_cast<size_t>(c)].ToFloat()));
      ASSERT_LE(error, budget) << "c=" << c << " t=" << t;
      max_error = std::max(max_error, error);
      max_fraction = std::max(max_fraction, error / budget);
    }
  }
  // The entire small state is raw-bit exact, not sampled.
  const auto* state = outputs[1].Get<Tensor>().Data<BFloat16>();
  for (int64_t c = 0; c < kChannels; ++c) {
    for (int64_t t = 0; t < 3; ++t) {
      ASSERT_EQ(state[c * 3 + t].val, sample(c, length + t).val) << c << ":" << t;
    }
  }
  ::testing::Test::RecordProperty(prefix + "_sample_count", 80);
  ::testing::Test::RecordProperty(prefix + "_sample_max_abs_error", MakeString(max_error));
  ::testing::Test::RecordProperty(prefix + "_sample_max_budget_fraction", MakeString(max_fraction));
}

void RunShapeProbe(ShapeProbeMode mode, int64_t batch, int64_t length, bool bounded_memory = false,
                   std::vector<OrtValue>* saved_outputs = nullptr) {
  const char* label = mode == ShapeProbeMode::Source  ? "source"
                      : mode == ShapeProbeMode::Fused ? "fused"
                                                      : "direct";
  const auto prefix = MakeString(label, "_B", batch, "_S", length);
  SessionOptions options;  // Keep the normal default optimization policy.
  options.enable_profiling = true;
  options.profile_file_prefix = ORT_TSTR("cuda_runtime_shape");
  if (mode == ShapeProbeMode::Fused) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kCausalConvConfig, "1"));
  }
  if (mode != ShapeProbeMode::Source) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
  }
  InferenceSessionWrapper session(options, GetEnvironment());
  ProfileCleanup cleanup{session, {}};
  if (bounded_memory) {
    OrtCUDAProviderOptionsV2 provider_options{};
    provider_options.use_tf32 = true;
    // Leave room inside the 16 GiB working budget for context/library allocations.
    provider_options.gpu_mem_limit = size_t{12} << 30;
    provider_options.arena_extend_strategy = ArenaExtendStrategy::kSameAsRequested;
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(CudaExecutionProviderWithOptions(&provider_options)));
    ::testing::Test::RecordProperty("cuda_arena_limit_bytes", MakeString(provider_options.gpu_mem_limit));
  } else {
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(ProductionMathCudaExecutionProvider()));
  }
  const auto model = MakeMobiusModel(mode == ShapeProbeMode::Direct);
  const auto bytes = model.SerializeAsString();
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());
  bool cuda_conv = false, cuda_sigmoid = false;
  std::string placement;
  for (const auto& node : session.GetGraph().Nodes()) {
    placement += MakeString(node.OpType(), ":", node.GetExecutionProviderType(), ";");
    cuda_conv |= node.OpType() == "Conv" && node.GetExecutionProviderType() == kCudaExecutionProvider;
    cuda_sigmoid |= node.OpType() == "Sigmoid" && node.GetExecutionProviderType() == kCudaExecutionProvider;
    if (mode == ShapeProbeMode::Source) {
      ASSERT_NE(node.OpType(), "CausalConvWithState");
    }
  }
  if (mode == ShapeProbeMode::Source) {
    ASSERT_TRUE(cuda_conv);
    ASSERT_TRUE(cuda_sigmoid);
  } else {
    ASSERT_EQ(session.GetGraph().NumberOfNodes(), 1);
    const auto& target = *session.GetGraph().Nodes().begin();
    ASSERT_EQ(target.OpType(), "CausalConvWithState");
    ASSERT_EQ(target.GetExecutionProviderType(), kCudaExecutionProvider);
  }
  ::testing::Test::RecordProperty(prefix + "_placement", placement);
  const auto feeds = MakeFeeds(length, batch);
  const std::vector<std::string> names{"y", "state"};
  std::vector<OrtValue> outputs;
  std::cout << prefix << " placement=" << placement << std::endl;
  const auto status = session.Run(RunOptions{}, feeds, names, &outputs);
  const auto sync = cudaDeviceSynchronize();
  ::testing::Test::RecordProperty(prefix + "_run_status", status.ToString());
  ::testing::Test::RecordProperty(prefix + "_synchronize", cudaGetErrorString(sync));
  std::cout << prefix << " run=" << status.ToString() << " synchronize=" << cudaGetErrorString(sync) << std::endl;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (!outputs[i].IsTensor()) continue;
    const auto shape = outputs[i].Get<Tensor>().Shape().ToString();
    ::testing::Test::RecordProperty(prefix + "_output" + MakeString(i), shape);
    std::cout << prefix << " output" << i << "=" << shape << std::endl;
  }
  ASSERT_STATUS_OK(status);
  ASSERT_EQ(sync, cudaSuccess);
  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_EQ(outputs[0].Get<Tensor>().Shape(), (TensorShape{batch, kChannels, length}));
  EXPECT_EQ(outputs[1].Get<Tensor>().Shape(), (TensorShape{batch, kChannels, 3}));
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& tensor = outputs[i].Get<Tensor>();
    ASSERT_TRUE(tensor.IsDataType<BFloat16>());
    const auto count = batch * kChannels * (i == 0 ? length : 3);
    ASSERT_EQ(tensor.Shape().Size(), count);
    ASSERT_EQ(tensor.SizeInBytes(), static_cast<size_t>(count) * sizeof(BFloat16));
  }
  if (batch == 1 && length == 65536) {
    CheckLongSequenceSamples(mode, model, feeds, outputs, length, prefix);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
  cleanup.path = session.EndProfiling();
  ASSERT_FALSE(cleanup.path.empty());
  std::ifstream profile_stream(cleanup.path);
  const auto profile = nlohmann::json::parse(profile_stream);
  size_t target_events = 0;
  for (const auto& event : profile) {
    if (event.value("cat", "") != "Node" || !event.contains("args")) continue;
    const auto& args = event["args"];
    if (args.value("op_name", "") == "CausalConvWithState" &&
        args.value("provider", "") == kCudaExecutionProvider) ++target_events;
  }
  EXPECT_EQ(target_events, mode == ShapeProbeMode::Source ? 0u : 1u);
  ::testing::Test::RecordProperty(prefix + "_cuda_target_node_events", static_cast<int>(target_events));
  // For B=0 this event proves node placement, NOT a GPU compute launch. The validated host early
  // return after allocating empty outputs is the no-launch proof; synchronization checks no error.
  if (bounded_memory) {
    size_t free_bytes = 0, total_bytes = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
    ::testing::Test::RecordProperty(prefix + "_gpu_free_bytes_after_run", MakeString(free_bytes));
  }
  if (saved_outputs) *saved_outputs = std::move(outputs);
}

TEST(CudaRuntimeFusionProbe, ZeroBatch) {
  if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  for (int64_t length : {1, 2, 17}) {
    for (auto mode : {ShapeProbeMode::Source, ShapeProbeMode::Fused, ShapeProbeMode::Direct}) {
      RunShapeProbe(mode, 0, length);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
    }
  }
}

void LongSequenceProbe(ShapeProbeMode mode) {
  if (!CudaHasBF16Support()) GTEST_SKIP() << "CUDA BF16 requires SM80 or newer.";
  if (Env::Default().GetEnvironmentVar("ORT_CUDA_RUNTIME_FUSION_LARGE_PROBE") != "1") {
    GTEST_SKIP() << "Local resource probe requires explicit opt-in and a <=300 second process timeout.";
  }
  constexpr size_t headroom = size_t{20} << 30;
#if defined(__linux__)
  std::ifstream memory("/proc/meminfo");
  std::string key, unit;
  size_t available_kib = 0;
  while (memory >> key >> available_kib >> unit) {
    if (key == "MemAvailable:") break;
  }
  if (key != "MemAvailable:" || available_kib < headroom / 1024) {
    GTEST_SKIP() << "Insufficient measured host memory headroom.";
  }
  ::testing::Test::RecordProperty("host_available_bytes_before_run", MakeString(available_kib * 1024));
#else
  GTEST_SKIP() << "This local resource probe currently measures Linux host headroom only.";
#endif
  size_t free_bytes = 0, total_bytes = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
  if (free_bytes < headroom) GTEST_SKIP() << "Insufficient measured GPU memory headroom.";
  int device = 0, shared_default = 0, shared_optin = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaDeviceGetAttribute(&shared_default, cudaDevAttrMaxSharedMemoryPerBlock, device), cudaSuccess);
  ASSERT_EQ(cudaDeviceGetAttribute(&shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device), cudaSuccess);
  ::testing::Test::RecordProperty("gpu_free_bytes_before_run", MakeString(free_bytes));
  ::testing::Test::RecordProperty("shared_memory_default_bytes", shared_default);
  ::testing::Test::RecordProperty("shared_memory_optin_bytes", shared_optin);
  ::testing::Test::RecordProperty("shared_candidate_bytes", (65536 + 3 + 4) * sizeof(float));
  ASSERT_GT((65536 + 3 + 4) * sizeof(float), static_cast<size_t>(shared_default));
  std::cout << "shared_memory_default=" << shared_default << " shared_memory_optin=" << shared_optin
            << " shared_candidate=" << (65536 + 3 + 4) * sizeof(float)
            << " target_prefill=direct_zero_shared" << std::endl;
  std::vector<OrtValue> fused;
  RunShapeProbe(mode, 1, 65536, true, mode == ShapeProbeMode::Fused ? &fused : nullptr);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  if (mode == ShapeProbeMode::Fused) {
    // The first session, its feeds and its CUDA arena have been destroyed. Retain only its actual
    // CPU outputs for chunked comparison with a separately scoped direct run, not a reference array.
    std::vector<OrtValue> direct;
    RunShapeProbe(ShapeProbeMode::Direct, 1, 65536, true, &direct);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    ASSERT_EQ(fused.size(), direct.size());
    constexpr size_t chunk_bytes = size_t{1} << 20;
    for (size_t i = 0; i < fused.size(); ++i) {
      const auto& a = fused[i].Get<Tensor>();
      const auto& b = direct[i].Get<Tensor>();
      ASSERT_EQ(a.Shape(), b.Shape());
      ASSERT_EQ(a.SizeInBytes(), b.SizeInBytes());
      const auto* ap = static_cast<const unsigned char*>(a.DataRaw());
      const auto* bp = static_cast<const unsigned char*>(b.DataRaw());
      for (size_t offset = 0; offset < a.SizeInBytes(); offset += chunk_bytes) {
        ASSERT_EQ(std::memcmp(ap + offset, bp + offset, std::min(chunk_bytes, a.SizeInBytes() - offset)), 0)
            << "output=" << i << " byte_offset=" << offset;
      }
    }
    ::testing::Test::RecordProperty("fused_direct_full_outputs_bitwise_equal", "1");
  }
}

TEST(CudaRuntimeFusionProbe, LongSequenceSource) {
  LongSequenceProbe(ShapeProbeMode::Source);
}

TEST(CudaRuntimeFusionProbe, LongSequenceFused) {
  LongSequenceProbe(ShapeProbeMode::Fused);
}

TEST(CudaRuntimeFusionProbe, LongSequenceDirect) {
  LongSequenceProbe(ShapeProbeMode::Direct);
}

#ifdef ENABLE_CUDA_NHWC_OPS
TEST(CudaRuntimeFusionTest, NhwcPreferenceDoesNotOfferRuntimeFusion) {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Level2;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kCausalConvConfig, "1"));
  InferenceSessionWrapper session(options, GetEnvironment());
  auto cuda = DefaultCudaNHWCExecutionProvider();
  auto* provider = cuda.get();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda)));
  const auto bytes = MakeMobiusModel(false).SerializeAsString();
  ASSERT_STATUS_OK(session.Load(bytes.data(), static_cast<int>(bytes.size())));
  struct Lookup final : IExecutionProvider::IKernelLookup {
    mutable size_t target_calls = 0;
    KernelCreateInfo target;
    const KernelCreateInfo* LookUpKernel(const Node& node) const override {
      if (node.OpType() != "CausalConvWithState") return nullptr;
      ++target_calls;
      return &target;
    }
  } lookup;
  GraphOptimizerRegistry registry(&options, TestCPUExecutionProvider(), &DefaultLoggingManager().DefaultLogger());
  const auto capabilities = provider->GetCapability(GraphViewer(session.GetGraph()), lookup, registry, nullptr);
  EXPECT_TRUE(capabilities.empty());
  EXPECT_EQ(lookup.target_calls, 0u);
}
#endif

}  // namespace
}  // namespace onnxruntime::test

#endif
