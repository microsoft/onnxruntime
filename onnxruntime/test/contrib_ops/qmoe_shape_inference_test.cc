#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

using ONNX_NAMESPACE::TensorProto;

struct QMoEShapeCase {
  int version = 2;
  int activation_type = TensorProto::FLOAT16;
  int64_t root_bits = 4;
  std::optional<int64_t> fc1_bits = 3;
  std::optional<int64_t> fc2_bits = 4;
  std::optional<int64_t> fc3_bits = 3;
  std::optional<int64_t> prepacked = 0;
  std::optional<int64_t> block = 32;
  std::optional<std::string> quant_type = "int";
  int64_t fusion = 1;
  std::string activation = "swiglu";
  std::map<size_t, std::vector<int64_t>> shapes{
      {0, {2, 64}}, {1, {2, 3}}, {2, {3, 256, 24}}, {3, {3, 256, 2}}, {5, {3, 64, 64}}, {6, {3, 64, 4}}};
  std::map<size_t, int> types;
  std::optional<std::pair<size_t, size_t>> symbolic_dimension;
  std::optional<size_t> unshaped_input;
};

common::Status ResolveQMoE(const QMoEShapeCase& spec,
                           const std::function<void(const Node&, const NodeArg&)>& verify = nullptr) {
  const std::unordered_map<std::string, int> versions{{kOnnxDomain, 17}, {kMSDomain, spec.version}};
  Model model("qmoe_shape_inference", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), versions, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto& empty = graph.GetOrCreateNodeArg("", nullptr);
  std::vector<NodeArg*> inputs(21, &empty);
  for (const auto& [index, dimensions] : spec.shapes) {
    ONNX_NAMESPACE::TypeProto type;
    int element_type = index == 2 || index == 5 || index == 8 || (index >= 11 && index <= 13)
                           ? TensorProto::UINT8
                           : spec.activation_type;
    if (auto override_type = spec.types.find(index); override_type != spec.types.end()) {
      element_type = override_type->second;
    }
    auto* tensor = type.mutable_tensor_type();
    tensor->set_elem_type(element_type);
    if (spec.unshaped_input != index) {
      auto* shape = tensor->mutable_shape();
      for (size_t axis = 0; axis < dimensions.size(); ++axis) {
        auto* dimension = shape->add_dim();
        if (spec.symbolic_dimension == std::make_pair(index, axis)) {
          dimension->set_dim_param("unknown");
        } else {
          dimension->set_dim_value(dimensions[axis]);
        }
      }
    }
    inputs[index] = &graph.GetOrCreateNodeArg("input_" + std::to_string(index), &type);
  }
  auto& output = graph.GetOrCreateNodeArg("output", nullptr);
  auto& node = graph.AddNode("qmoe", "QMoE", "", inputs, {&output}, nullptr, kMSDomain);
  node.AddAttribute("expert_weight_bits", spec.root_bits);
  if (spec.fc1_bits) node.AddAttribute("fc1_expert_weight_bits", *spec.fc1_bits);
  if (spec.fc2_bits) node.AddAttribute("fc2_expert_weight_bits", *spec.fc2_bits);
  if (spec.fc3_bits) node.AddAttribute("fc3_expert_weight_bits", *spec.fc3_bits);
  if (spec.prepacked) node.AddAttribute("weights_prepacked", *spec.prepacked);
  if (spec.block) node.AddAttribute("block_size", *spec.block);
  if (spec.quant_type) node.AddAttribute("quant_type", *spec.quant_type);
  node.AddAttribute("swiglu_fusion", spec.fusion);
  node.AddAttribute("activation_type", spec.activation);
  graph.SetOutputs({&output});
  auto status = graph.Resolve();
  if (status.IsOK() && verify) verify(node, output);
  return status;
}

TEST(QMoEShapeInferenceTest, Int3BaselineCombinationsAndTypes) {
  for (int element_type : {TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    for (int64_t block : {32, 64, 128}) {
      for (auto bits : {std::make_pair(3, 4), std::make_pair(3, 3), std::make_pair(4, 3)}) {
        SCOPED_TRACE(std::to_string(element_type) + ":" + std::to_string(block) + ":" +
                     std::to_string(bits.first) + ":" + std::to_string(bits.second));
        QMoEShapeCase spec;
        spec.activation_type = element_type;
        spec.block = block;
        spec.fc1_bits = bits.first;
        spec.fc2_bits = bits.second;
        spec.fc3_bits = bits.first;
        spec.shapes = {{0, {2, 128}}, {1, {2, 3}}, {2, {3, 512, 128 * bits.first / 8}}, {3, {3, 512, 128 / block}}, {5, {3, 128, 256 * bits.second / 8}}, {6, {3, 128, 256 / block}}, {4, {3, 512}}, {7, {3, 128}}, {14, {2, 3}}, {11, {3, 512, ((128 / block) * bits.first + 7) / 8}}, {12, {3, 128, ((256 / block) * bits.second + 7) / 8}}};
        ASSERT_STATUS_OK(ResolveQMoE(spec, [&](const Node& node, const NodeArg& output) {
          EXPECT_EQ(node.SinceVersion(), 2);
          ASSERT_NE(output.Shape(), nullptr);
          ASSERT_EQ(output.Shape()->dim_size(), 2);
          EXPECT_EQ(output.Shape()->dim(0).dim_value(), 2);
          EXPECT_EQ(output.Shape()->dim(1).dim_value(), 128);
          EXPECT_EQ(output.TypeAsProto()->tensor_type().elem_type(), element_type);
        }));
      }
    }
  }
}

TEST(QMoEShapeInferenceTest, InheritanceAndLegacyActivationChoices) {
  QMoEShapeCase spec;
  spec.root_bits = 3;
  spec.fc1_bits.reset();
  spec.fc2_bits.reset();
  spec.fc3_bits.reset();
  spec.shapes[5][2] = 48;
  for (const std::string activation : {"relu", "gelu", "silu", "swiglu", "identity", "relu2"}) {
    spec.activation = activation;
    ASSERT_STATUS_OK(ResolveQMoE(spec));
  }
}

TEST(QMoEShapeInferenceTest, IndependentProjectionWidthsAndPackedZeroPointRows) {
  for (int element_type : {TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    for (auto bits : {std::make_pair(3, 2), std::make_pair(2, 3), std::make_pair(3, 4),
                      std::make_pair(4, 3), std::make_pair(3, 8), std::make_pair(8, 3)}) {
      QMoEShapeCase spec;
      spec.activation_type = element_type;
      spec.fc1_bits = bits.first;
      spec.fc2_bits = bits.second;
      spec.fc3_bits = bits.first;
      spec.shapes = {{0, {2, 64}}, {1, {2, 3}}, {2, {3, 576, 8 * bits.first}}, {3, {3, 576, 2}}, {5, {3, 64, 36 * bits.second}}, {6, {3, 64, 9}}, {11, {3, 576, (2 * bits.first + 7) / 8}}, {12, {3, 64, (9 * bits.second + 7) / 8}}};
      ASSERT_STATUS_OK(ResolveQMoE(spec));
    }
  }
}

TEST(QMoEShapeInferenceTest, PaddedAndSymbolicInputs) {
  QMoEShapeCase spec;
  spec.shapes[0] = {1, 2, 64};
  ASSERT_STATUS_OK(ResolveQMoE(spec));
  spec.symbolic_dimension = {{0, 0}};
  ASSERT_STATUS_OK(ResolveQMoE(spec, [](const Node&, const NodeArg& output) {
    EXPECT_EQ(output.Shape()->dim(0).dim_param(), "unknown");
  }));
  spec.symbolic_dimension = {{0, 2}};
  ASSERT_STATUS_OK(ResolveQMoE(spec));
  spec.unshaped_input = 2;
  ASSERT_STATUS_OK(ResolveQMoE(spec));
}

TEST(QMoEShapeInferenceTest, LegacyWidthsRemainGenericInBothVersions) {
  for (int version : {1, 2}) {
    for (int64_t bits : {2, 4, 8}) {
      QMoEShapeCase spec;
      spec.version = version;
      spec.root_bits = bits;
      spec.fc1_bits.reset();
      spec.fc2_bits.reset();
      spec.fc3_bits.reset();
      spec.block.reset();
      spec.prepacked.reset();
      spec.quant_type.reset();
      spec.activation_type = TensorProto::FLOAT;
      spec.shapes.erase(3);
      spec.shapes.erase(6);
      spec.fusion = 0;
      ASSERT_STATUS_OK(ResolveQMoE(spec));
    }
  }
}

TEST(QMoEShapeInferenceTest, LargeHiddenDimensionDoesNotOverflowPackedByteCalculation) {
  QMoEShapeCase spec;
  const int64_t hidden = (std::numeric_limits<int64_t>::max() / 32) * 32;
  spec.shapes[0] = {1, hidden};
  spec.shapes[1][0] = 1;
  spec.shapes[2][2] = (hidden / 8) * 3;
  spec.shapes[3][2] = hidden / 32;
  spec.shapes[5][1] = hidden;
  spec.shapes[6][1] = hidden;
  ASSERT_STATUS_OK(ResolveQMoE(spec));
}

TEST(QMoEShapeInferenceTest, Version2HasNoCpuKernel) {
  CPUExecutionProvider provider(CPUExecutionProviderInfo{});
  auto registry = provider.GetKernelRegistry();
  ASSERT_NE(registry, nullptr);
  const KernelRegistry::TypeConstraintMap types{
      {"T", DataTypeImpl::GetTensorType<MLFloat16>()},
      {"T1", DataTypeImpl::GetTensorType<uint8_t>()},
      {"T2", DataTypeImpl::GetTensorType<MLFloat16>()}};
  for (int version : {1, 2}) {
    const KernelCreateInfo* kernel = nullptr;
    auto status = registry->TryFindKernel(kCpuExecutionProvider, "QMoE", kMSDomain, version, types,
                                          DefaultLoggingManager().DefaultLogger(), &kernel);
    EXPECT_EQ(status.IsOK(), version == 1) << status.ErrorMessage();
    EXPECT_EQ(kernel != nullptr, version == 1);
  }
}

TEST(QMoEShapeInferenceTest, OtherMicrosoftOperatorsStillResolveToVersion1) {
  const std::unordered_map<std::string, int> versions{{kOnnxDomain, 17}, {kMSDomain, 2}};
  Model model("legacy_ms_schema", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), versions, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  ONNX_NAMESPACE::TypeProto type;
  type.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
  auto& input = graph.GetOrCreateNodeArg("input", &type);
  auto& output = graph.GetOrCreateNodeArg("output", nullptr);
  auto& node = graph.AddNode("legacy", "QuickGelu", "", {&input}, {&output}, nullptr, kMSDomain);
  graph.SetOutputs({&output});
  ASSERT_STATUS_OK(graph.Resolve());
  EXPECT_EQ(node.SinceVersion(), 1);
}

#ifndef ORT_NO_EXCEPTIONS

TEST(QMoEShapeInferenceTest, RejectsInvalidInt3Metadata) {
  using Mutation = std::pair<std::string, std::function<void(QMoEShapeCase&)>>;
  const std::vector<Mutation> mutations{
      {"prepacked omitted", [](auto& spec) { spec.prepacked.reset(); }},
      {"prepacked auto", [](auto& spec) { spec.prepacked = -1; }},
      {"prepacked enabled", [](auto& spec) { spec.prepacked = 1; }},
      {"quant type omitted", [](auto& spec) { spec.quant_type.reset(); }},
      {"fp4", [](auto& spec) { spec.quant_type = "fp4"; }},
      {"fp8", [](auto& spec) { spec.quant_type = "fp8"; }},
      {"block omitted", [](auto& spec) { spec.block.reset(); }},
      {"block16", [](auto& spec) { spec.block = 16; }},
      {"block0", [](auto& spec) { spec.block = 0; }},
      {"block negative", [](auto& spec) { spec.block = -32; }},
      {"unfused", [](auto& spec) { spec.fusion = 0; }},
      {"concatenated fusion", [](auto& spec) { spec.fusion = 2; }},
      {"FC3 independent inheritance", [](auto& spec) { spec.fc3_bits.reset(); }},
      {"FC3 mismatch", [](auto& spec) { spec.fc3_bits = 4; }},
      {"activation float32", [](auto& spec) { spec.activation_type = TensorProto::FLOAT; }},
      {"scale float32", [](auto& spec) { spec.types[3] = TensorProto::FLOAT; }},
      {"scale fp8", [](auto& spec) { spec.types[3] = TensorProto::FLOAT8E4M3FN; }},
      {"scale integer", [](auto& spec) { spec.types[6] = TensorProto::UINT8; }},
      {"scale mismatched BF16", [](auto& spec) { spec.types[6] = TensorProto::BFLOAT16; }},
      {"weight float", [](auto& spec) { spec.types[2] = TensorProto::FLOAT16; }},
      {"zero-point int8", [](auto& spec) { spec.shapes[12] = {3, 64, 2}; spec.types[12] = TensorProto::INT8; }},
      {"FC1 scales missing", [](auto& spec) { spec.shapes.erase(3); }},
      {"FC2 scales missing", [](auto& spec) { spec.shapes.erase(6); }},
      {"FC3 weights", [](auto& spec) { spec.shapes[8] = {3, 128, 24}; }},
      {"FC3 scales", [](auto& spec) { spec.shapes[9] = {3, 128, 2}; }},
      {"FC3 bias", [](auto& spec) { spec.shapes[10] = {3, 128}; }},
      {"FC3 zero points", [](auto& spec) { spec.shapes[13] = {3, 128, 1}; }},
      {"global scale", [](auto& spec) { spec.shapes[15] = {3}; }},
      {"activation scale", [](auto& spec) { spec.shapes[18] = {1}; }},
      {"activation rank", [](auto& spec) { spec.shapes[0] = {64}; }},
      {"FC1 rank", [](auto& spec) { spec.shapes[2] = {256, 24}; }},
      {"FC2 rank", [](auto& spec) { spec.shapes[5] = {64, 64}; }},
      {"scale rank", [](auto& spec) { spec.shapes[3] = {3, 256}; }},
      {"zero point rank", [](auto& spec) { spec.shapes[11] = {3, 256}; }},
      {"bias rank", [](auto& spec) { spec.shapes[7] = {64}; }},
      {"odd fused FC1 rows", [](auto& spec) { spec.shapes[2][1] = 255; }},
      {"hidden tail", [](auto& spec) { spec.shapes[0][1] = 63; }},
      {"intermediate tail", [](auto& spec) { spec.shapes[2][1] = 258; }},
      {"zero hidden", [](auto& spec) { spec.shapes[0][1] = 0; }},
      {"FC1 packed bytes", [](auto& spec) { spec.shapes[2][2] = 32; }},
      {"FC2 packed bytes", [](auto& spec) { spec.shapes[5][2] = 32; }},
      {"INT3 FC2 uses I not H", [](auto& spec) { spec.fc2_bits = 3; spec.shapes[5][2] = 24; }},
      {"FC2 hidden rows", [](auto& spec) { spec.shapes[5][1] = 128; }},
      {"FC2 experts", [](auto& spec) { spec.shapes[5][0] = 4; }},
      {"FC1 scale experts", [](auto& spec) { spec.shapes[3][0] = 4; }},
      {"FC1 scale rows", [](auto& spec) { spec.shapes[3][1] = 128; }},
      {"FC1 scale blocks", [](auto& spec) { spec.shapes[3][2] = 4; }},
      {"FC2 scale blocks", [](auto& spec) { spec.shapes[6][2] = 2; }},
      {"symbolic hidden still checks FC2 scale rows", [](auto& spec) {
         spec.symbolic_dimension = {{0, 1}};
         spec.shapes[6][1] = 128;
       }},
      {"symbolic FC1 experts still checks metadata experts", [](auto& spec) {
         spec.symbolic_dimension = {{2, 0}};
         spec.shapes[6][0] = 4;
       }},
      {"FC1 zero point bytes", [](auto& spec) { spec.shapes[11] = {3, 256, 2}; }},
      {"FC2 zero point bytes", [](auto& spec) { spec.shapes[12] = {3, 64, 1}; }},
      {"INT3 zero points use ceil bits not integer pack", [](auto& spec) {
         spec.fc2_bits = 3;
         spec.shapes[2][1] = 576;
         spec.shapes[3][1] = 576;
         spec.shapes[5][2] = 108;
         spec.shapes[6][2] = 9;
         spec.shapes[12] = {3, 64, 5};
       }},
      {"FC2 bias rows", [](auto& spec) { spec.shapes[7] = {3, 128}; }},
      {"router tokens", [](auto& spec) { spec.shapes[1][0] = 3; }},
      {"padded router tokens", [](auto& spec) { spec.shapes[0] = {2, 2, 64}; }},
      {"padded token overflow", [](auto& spec) {
         spec.shapes[0] = {std::numeric_limits<int64_t>::max(), 2, 64};
       }},
      {"router experts", [](auto& spec) { spec.shapes[1][1] = 4; }},
      {"router weights tokens", [](auto& spec) { spec.shapes[14] = {3, 3}; }}};
  for (const auto& [name, mutate] : mutations) {
    SCOPED_TRACE(name);
    QMoEShapeCase spec;
    mutate(spec);
    auto status = ResolveQMoE(spec);
    EXPECT_FALSE(status.IsOK()) << name;
    EXPECT_NE(status.ErrorMessage().find("QMoE"), std::string::npos) << status.ErrorMessage();
  }
}

TEST(QMoEShapeInferenceTest, RejectsUnapprovedWidthsAndVersion1Int3) {
  for (int version : {1, 2}) {
    for (const std::string attribute : {"root", "fc1", "fc2", "fc3"}) {
      for (int64_t invalid_bits : {0, 1, 3, 5, 6, 7, 9}) {
        if ((version == 1 && invalid_bits != 3) || (version == 2 && invalid_bits == 3)) continue;
        SCOPED_TRACE(std::to_string(version) + ":" + attribute + ":" + std::to_string(invalid_bits));
        QMoEShapeCase spec;
        spec.version = version;
        spec.fc1_bits = 4;
        spec.fc2_bits = 4;
        spec.fc3_bits = 4;
        if (attribute == "root") spec.root_bits = invalid_bits;
        if (attribute == "fc1") spec.fc1_bits = invalid_bits;
        if (attribute == "fc2") spec.fc2_bits = invalid_bits;
        if (attribute == "fc3") spec.fc3_bits = invalid_bits;
        EXPECT_FALSE(ResolveQMoE(spec).IsOK());
      }
    }
  }
}

#endif

}  // namespace
}  // namespace test
}  // namespace onnxruntime