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
  options.enable_profiling = true;
  options.profile_file_prefix = ORT_TSTR("unit_swish_source");
  InferenceSessionWrapper source(options, GetEnvironment());
  ProfileCleanup source_cleanup{source, {}};
  options.profile_file_prefix = ORT_TSTR("unit_swish_fused");
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(cuda_runtime_fusion::kUnitSwishConfig, "1"));
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

TEST(CudaRuntimeFusionTest, UnitSwishCausalMlpRank3BFloat16BitwiseDefaultOptimization) {
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

}  // namespace
}  // namespace onnxruntime::test
#endif
