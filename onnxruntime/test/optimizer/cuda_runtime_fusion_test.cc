// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>

#include "core/framework/op_kernel.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/cuda_runtime_fusion.h"
#include "core/optimizer/pattern_rewriter.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime::test {
namespace {
using cuda_runtime_fusion::kCausalConvConfig;
using cuda_runtime_fusion::kMobiusBf16Rule;
using cuda_runtime_fusion::kUnitSwishConfig;
using cuda_runtime_fusion::kUnitSwishRule;

struct UnitSwishGraph {
  Model model;
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder{graph};
  NodeArg* x = builder.MakeInput<float>(std::vector<int64_t>{2, 17}, "x");
  NodeArg* y = builder.MakeOutput<float>(std::vector<int64_t>{2, 17});
  Node& swish = builder.AddNode("Swish", {x}, {y});

  explicit UnitSwishGraph(int type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      : model("unit-swish-host", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 24}, {kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger()) {
    for (auto* value : {x, y}) {
      auto metadata = *value->TypeAsProto();
      metadata.mutable_tensor_type()->set_elem_type(type);
      ORT_THROW_IF_ERROR(value->UpdateTypeAndShape(metadata, false, true,
                                                   DefaultLoggingManager().DefaultLogger()));
    }
  }
};

class UnitSwishLookup final : public IExecutionProvider::IKernelLookup {
 public:
  bool supported = true;
  int type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  mutable size_t calls = 0;
  KernelCreateInfo kernel;

  const KernelCreateInfo* LookUpKernel(const Node& node) const override {
    ++calls;
    EXPECT_EQ(node.OpType(), "ScaledSiLU");
    EXPECT_EQ(node.Domain(), kMSDomain);
    EXPECT_EQ(node.SinceVersion(), 1);
    EXPECT_EQ(node.InputDefs().size(), 1u);  // No optional M scale input.
    EXPECT_EQ(node.OutputDefs().size(), 1u);
    EXPECT_EQ(node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type(),
              type);
    EXPECT_EQ(node.OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type(),
              type);
    EXPECT_FLOAT_EQ(node.GetAttributes().at("alpha").f(), 1.0f);
    return supported ? &kernel : nullptr;
  }
};

class UnitSwishHostTest : public ::testing::Test {
 protected:
  SessionOptions options;
  CPUExecutionProvider cpu{CPUExecutionProviderInfo{}};
  UnitSwishLookup lookup;
  const logging::Logger& logger = DefaultLoggingManager().DefaultLogger();

  Status Select(const GraphViewer& graph, std::vector<std::unique_ptr<ComputeCapability>>& offers,
                bool consumer = true) {
    GraphOptimizerRegistry registry(&options, &cpu, &logger, consumer);
    return cuda_runtime_fusion::Select(graph, lookup, registry, kUnitSwishRule, offers);
  }
  void Enable() { ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kUnitSwishConfig, "1")); }
};

TEST_F(UnitSwishHostTest, UnitSwishIndependentPolicyAndConsumerGuards) {
  UnitSwishGraph f;
  ASSERT_STATUS_OK(f.graph.Resolve());
  GraphViewer viewer(f.graph);
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kCausalConvConfig, "1"));
  ASSERT_STATUS_OK(Select(viewer, offers));
  EXPECT_TRUE(offers.empty());
  for (const char* value : {"", "true", "01", " 1", "2", "-1"}) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kUnitSwishConfig, value));
    EXPECT_FALSE(Select(viewer, offers, false).IsOK());
    EXPECT_TRUE(offers.empty());
  }
  Enable();
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kCausalConvConfig, "0"));
  ASSERT_STATUS_OK(Select(viewer, offers, false));  // Includes the ORT-load consumer.
  EXPECT_TRUE(offers.empty());
  options.graph_optimization_level = TransformerLevel::Level1;
  ASSERT_STATUS_OK(Select(viewer, offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 0u);
  options.graph_optimization_level = TransformerLevel::Level2;
  ASSERT_STATUS_OK(Select(viewer, offers));
  ASSERT_EQ(offers.size(), 1u);
  EXPECT_EQ(lookup.calls, 1u);
}

TEST_F(UnitSwishHostTest, UnitSwishReadOnlyDiscoveryAndFanoutIdentity) {
  Enable();
  UnitSwishGraph f;
  auto* z = f.builder.MakeOutput();
  f.builder.AddNode("Add", {f.y, f.x}, {z});  // Shared input, fanout and graph output.
  auto* other = f.builder.MakeOutput();
  f.builder.AddNode("Identity", {f.y}, {other});
  f.graph.SetOutputs({f.y, z, other});
  ASSERT_STATUS_OK(f.graph.Resolve());
  const auto before = f.graph.ToGraphProto().SerializeAsString();
  const auto* original_y = f.y;
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  for (int repeat = 0; repeat < 2; ++repeat) {
    offers.clear();
    ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
    ASSERT_EQ(offers.size(), 1u);
    EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
  }
  GraphOptimizerRegistry registry(&options, &cpu, &logger);
  auto& selected = *offers[0]->nodes_to_optimize[0];
  ASSERT_STATUS_OK(selected.optimization_func(f.graph, selected, *offers[0], registry));
  EXPECT_EQ(f.graph.NumberOfNodes(), 3);
  const auto* target = f.graph.GetProducerNode(original_y->Name());
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(target->OpType(), "ScaledSiLU");
  EXPECT_EQ(target->InputDefs()[0], f.x);
  EXPECT_EQ(target->OutputDefs()[0], original_y);
  EXPECT_EQ(f.graph.GetOutputs()[0], original_y);
  EXPECT_EQ(target->GetOutputEdgesCount(), 2u);
  const auto after = f.graph.ToGraphProto().SerializeAsString();
  EXPECT_FALSE(selected.optimization_func(f.graph, selected, *offers[0], registry).IsOK());
  EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), after);
}

TEST_F(UnitSwishHostTest, BFloat16TypedProbeInitializerAndRewrite) {
  Enable();
  lookup.type = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
  for (int storage = 0; storage < 3; ++storage) {
    SCOPED_TRACE(storage);
    UnitSwishGraph f(lookup.type);
    if (storage != 0) {
      ONNX_NAMESPACE::TensorProto tensor;
      tensor.set_name(f.x->Name());
      tensor.set_data_type(lookup.type);
      tensor.add_dims(2);
      tensor.add_dims(17);
      if (storage == 1) {
        tensor.set_raw_data(std::string(34 * sizeof(uint16_t), '\0'));
      } else {
        tensor.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
        auto* location = tensor.add_external_data();
        location->set_key("location");
        location->set_value("bf16_unit_swish_must_not_open_missing_payload.bin");
      }
      f.graph.AddInitializedTensor(tensor);
    }
    auto* fanout = f.builder.MakeOutput<BFloat16>(std::vector<int64_t>{2, 17});
    f.builder.AddNode("Identity", {f.y}, {fanout});
    f.graph.SetOutputs({f.y, fanout});
    ASSERT_STATUS_OK(f.graph.Resolve());
    const auto before = f.graph.ToGraphProto().SerializeAsString();
    const auto calls = lookup.calls;
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
    ASSERT_EQ(offers.size(), 1u);
    EXPECT_EQ(lookup.calls - calls, 1u);
    EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    auto& selected = *offers[0]->nodes_to_optimize[0];
    ASSERT_STATUS_OK(selected.optimization_func(f.graph, selected, *offers[0], registry));
    ASSERT_EQ(f.graph.NumberOfNodes(), 2);
    const auto* target = f.graph.GetProducerNode(f.y->Name());
    ASSERT_NE(target, nullptr);
    EXPECT_EQ(target->OpType(), "ScaledSiLU");
    EXPECT_EQ(target->InputDefs()[0], f.x);
    EXPECT_EQ(target->OutputDefs()[0], f.y);
    EXPECT_EQ(f.graph.GetOutputs()[0], f.y);
    EXPECT_EQ(target->GetOutputEdgesCount(), 1u);
  }
}

TEST_F(UnitSwishHostTest, BFloat16SnapshotAndInitializerMetadataRevalidation) {
  Enable();
  lookup.type = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
  for (int mutation = 0; mutation < 5; ++mutation) {
    SCOPED_TRACE(mutation);
    UnitSwishGraph f(lookup.type);
    if (mutation >= 3) {
      ONNX_NAMESPACE::TensorProto tensor;
      tensor.set_name(f.x->Name());
      tensor.set_data_type(lookup.type);
      tensor.add_dims(2);
      tensor.add_dims(17);
      tensor.set_raw_data(std::string(34 * sizeof(uint16_t), '\0'));
      f.graph.AddInitializedTensor(tensor);
    }
    ASSERT_STATUS_OK(f.graph.Resolve());
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
    ASSERT_EQ(offers.size(), 1u);
    if (mutation < 3) {
      for (auto* value : {f.x, f.y}) {
        if (mutation == 0 && value == f.y) continue;
        if (mutation == 1 && value == f.x) continue;
        auto type = *value->TypeAsProto();
        type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        ASSERT_STATUS_OK(value->UpdateTypeAndShape(type, false, true, logger));
      }
    } else {
      auto& tensor = *const_cast<ONNX_NAMESPACE::TensorProto*>(
          f.graph.GetConstantInitializer(f.x->Name(), false));
      if (mutation == 3)
        tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      else
        tensor.set_dims(1, 18);
    }
    const auto before = f.graph.ToGraphProto().SerializeAsString();
    const auto calls = lookup.calls;
    if (mutation != 2) {  // Both changed to FLOAT is eligible anew, but not the saved snapshot.
      std::vector<std::unique_ptr<ComputeCapability>> fresh;
      ASSERT_STATUS_OK(Select(GraphViewer(f.graph), fresh));
      EXPECT_TRUE(fresh.empty());
      EXPECT_EQ(lookup.calls, calls);
    }
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    auto& selected = *offers[0]->nodes_to_optimize[0];
    EXPECT_FALSE(selected.optimization_func(f.graph, selected, *offers[0], registry).IsOK());
    EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
  }
}

TEST_F(UnitSwishHostTest, UnitSwishInvalidContractsDeclineBeforeLookup) {
  Enable();
  for (int mutation = 0; mutation < 20; ++mutation) {
    SCOPED_TRACE(mutation);
    UnitSwishGraph f;
    ASSERT_STATUS_OK(f.graph.Resolve());
    if (mutation < 5) {
      using Limits = std::numeric_limits<float>;
      const std::array<float, 5> alphas{0.0f, -1.0f, 0.5f, Limits::infinity(), Limits::quiet_NaN()};
      f.swish.GetMutableAttributes().at("alpha").set_f(alphas[mutation]);
    } else if (mutation == 5) {
      auto& alpha = f.swish.GetMutableAttributes().at("alpha");
      alpha.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
      alpha.clear_f();
      alpha.set_i(1);
    } else if (mutation == 6) {
      auto& beta = f.swish.GetMutableAttributes()["beta"];
      beta.set_name("beta");
      beta.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
      beta.set_f(1.0f);
    } else if (mutation < 10) {
      auto type = *f.x->TypeAsProto();
      type.mutable_tensor_type()->set_elem_type(std::array<int, 3>{
          ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
          ONNX_NAMESPACE::TensorProto_DataType_DOUBLE}[mutation - 7]);
      ASSERT_STATUS_OK(f.x->UpdateTypeAndShape(type, false, true, logger));
    } else if (mutation == 10) {
      f.swish.SetExecutionProviderType(kCpuExecutionProvider);
    } else if (mutation == 11) {
      auto shape = *f.y->Shape();
      shape.mutable_dim(1)->set_dim_value(18);
      f.y->SetShape(shape);
    } else if (mutation == 12) {
      f.swish.SetSinceVersion(23);
    } else if (mutation == 13) {
      auto type = *f.y->TypeAsProto();
      type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
      ASSERT_STATUS_OK(f.y->UpdateTypeAndShape(type, false, true, logger));
    } else if (mutation == 14) {
      const_cast<std::string&>(f.swish.Domain()) = kMSDomain;
    } else if (mutation == 15) {
      f.swish.MutableInputDefs().push_back(f.x);
    } else if (mutation == 16) {
      f.swish.MutableOutputDefs().push_back(f.y);
    } else if (mutation == 17) {
      auto& imports = const_cast<std::unordered_map<std::string, int>&>(f.graph.DomainToVersionMap());
      imports[kOnnxDomain] = 25;
    } else if (mutation == 18) {
      auto& imports = const_cast<std::unordered_map<std::string, int>&>(f.graph.DomainToVersionMap());
      imports[kMSDomain] = 2;
    } else {
      f.swish.SetOverload("model_local_function");
    }
    const auto before = f.graph.ToGraphProto().SerializeAsString();
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
    EXPECT_TRUE(offers.empty());
    EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
  }
  EXPECT_EQ(lookup.calls, 0u);
}

TEST_F(UnitSwishHostTest, MatchingFloat16AndDoubleStillDeclineBeforeLookup) {
  Enable();
  for (int type : {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE}) {
    UnitSwishGraph f(type);
    ASSERT_STATUS_OK(f.graph.Resolve());
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
    EXPECT_TRUE(offers.empty());
  }
  EXPECT_EQ(lookup.calls, 0u);
}

TEST_F(UnitSwishHostTest, UnitSwishMissingKernelFilteredViewerAndNativeLimits) {
  Enable();
  UnitSwishGraph f;
  ASSERT_STATUS_OK(f.graph.Resolve());
  lookup.supported = false;
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 1u);
  lookup.supported = true;
  IndexedSubGraph region;
  region.nodes.push_back(f.swish.Index());
  auto meta = std::make_unique<IndexedSubGraph::MetaDef>();
  meta->name = "filtered_unit_swish";
  meta->domain = "test";
  meta->since_version = 1;
  meta->inputs = {f.x->Name()};
  meta->outputs = {f.y->Name()};
  region.SetMetaDef(std::move(meta));
  GraphViewer filtered(f.graph, region);
  ASSERT_STATUS_OK(Select(filtered, offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 1u);
  constexpr int64_t limit = int64_t{256} * std::numeric_limits<int>::max();
  std::vector<std::vector<int64_t>> shapes{{}, {0}, {17}, {1, 10240}, {-1, 17}, {limit}, {limit + 1}};
  shapes.push_back({std::numeric_limits<int64_t>::max(), 2});
  shapes.push_back({std::numeric_limits<int64_t>::max(), 2, 0});
  for (const auto& dims : shapes) {
    SCOPED_TRACE(::testing::PrintToString(dims));
    ONNX_NAMESPACE::TensorShapeProto shape;
    for (auto dim : dims) {
      auto* d = shape.add_dim();
      if (dim >= 0) d->set_dim_value(dim);
    }
    f.x->SetShape(shape);
    f.y->SetShape(shape);
    offers.clear();
    const auto calls = lookup.calls;
    ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
    const bool eligible = dims != std::vector<int64_t>{limit + 1} &&
                          dims != std::vector<int64_t>{std::numeric_limits<int64_t>::max(), 2};
    EXPECT_EQ(offers.size(), eligible ? 1u : 0u);
    EXPECT_EQ(lookup.calls - calls, eligible ? 1u : 0u);
  }
}

TEST_F(UnitSwishHostTest, UnitSwishStaleSnapshotRejectsWithoutMutation) {
  Enable();
  for (int mutation = 0; mutation < 17; ++mutation) {
    SCOPED_TRACE(mutation);
    UnitSwishGraph f;
    ASSERT_STATUS_OK(f.graph.Resolve());
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
    ASSERT_EQ(offers.size(), 1u);
    auto& selected = *offers[0]->nodes_to_optimize[0];
    if (mutation == 0) f.swish.SetExecutionProviderType(kCudaExecutionProvider);
    if (mutation == 1) f.swish.AddAttribute("alpha", 0.5f);
    if (mutation == 2) {  // A semantically valid but changed canonical attribute contract.
      f.swish.AddAttribute("alpha", 1.0f);
      f.swish.GetMutableAttributes().at("alpha").set_doc_string("changed");
    }
    if (mutation == 3 || mutation == 4) {
      auto* value = mutation == 3 ? f.x : f.y;
      auto shape = *value->Shape();
      shape.mutable_dim(0)->clear_dim_value();
      shape.mutable_dim(0)->set_dim_param("changed");
      value->SetShape(shape);
    }
    if (mutation == 5) {
      auto& x = f.graph.GetOrCreateNodeArg("other_x", f.x->TypeAsProto());
      graph_utils::ReplaceNodeInput(f.swish, 0, x);
    }
    if (mutation == 6) {
      auto& y = f.graph.GetOrCreateNodeArg("other_y", f.y->TypeAsProto());
      f.swish.MutableOutputDefs()[0] = &y;
    }
    if (mutation == 7) selected.sub_graph->nodes.clear();
    if (mutation == 8) offers[0]->sub_graph->nodes.push_back(f.swish.Index());
    if (mutation == 9) f.swish.SetSinceVersion(25);
    if (mutation == 10) f.graph.RemoveNode(f.swish.Index());
    if (mutation == 11) {
      auto& imports = const_cast<std::unordered_map<std::string, int>&>(f.graph.DomainToVersionMap());
      imports[kOnnxDomain] = 25;
    }
    if (mutation == 12) {
      auto& imports = const_cast<std::unordered_map<std::string, int>&>(f.graph.DomainToVersionMap());
      imports[kMSDomain] = 2;
    }
    if (mutation == 13) const_cast<std::string&>(f.swish.Domain()) = kMSDomain;
    if (mutation == 14) f.swish.SetOverload("changed");
    if (mutation == 15 || mutation == 16) {
      auto* value = mutation == 15 ? f.x : f.y;
      auto type = *value->TypeAsProto();
      type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
      ASSERT_STATUS_OK(value->UpdateTypeAndShape(type, false, true, logger));
    }
    const auto before = f.graph.ToGraphProto().SerializeAsString();
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    EXPECT_FALSE(selected.optimization_func(f.graph, selected, *offers[0], registry).IsOK());
    EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
  }
}

TEST_F(UnitSwishHostTest, UnitSwishExternalInitializerDiscoveryNeverReadsPayload) {
  Enable();
  UnitSwishGraph f;
  ONNX_NAMESPACE::TensorProto tensor;
  tensor.set_name(f.x->Name());
  tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tensor.add_dims(2);
  tensor.add_dims(17);
  tensor.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto* location = tensor.add_external_data();
  location->set_key("location");
  location->set_value("unit_swish_must_not_open_missing_payload.bin");
  f.graph.AddInitializedTensor(tensor);
  ASSERT_STATUS_OK(f.graph.Resolve());
  const auto before = f.graph.ToGraphProto().SerializeAsString();
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
  ASSERT_EQ(offers.size(), 1u);
  EXPECT_EQ(lookup.calls, 1u);
  EXPECT_EQ(f.graph.ToGraphProto().SerializeAsString(), before);
}

TEST_F(UnitSwishHostTest, UnitSwishAbsentAlphaAndUnknownRankRemainEligible) {
  Enable();
  UnitSwishGraph f;
  ASSERT_STATUS_OK(f.graph.Resolve());
  f.swish.GetMutableAttributes().erase("alpha");
  f.x->ClearShape();
  f.y->ClearShape();
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
  ASSERT_EQ(offers.size(), 1u);
  GraphOptimizerRegistry registry(&options, &cpu, &logger);
  auto& selected = *offers[0]->nodes_to_optimize[0];
  ASSERT_STATUS_OK(selected.optimization_func(f.graph, selected, *offers[0], registry));
  EXPECT_EQ(f.graph.GetProducerNode(f.y->Name())->OpType(), "ScaledSiLU");
}

#if !defined(DISABLE_SPARSE_TENSORS)
TEST_F(UnitSwishHostTest, UnitSwishSparseInitializerDeclinesBeforeLookup) {
  Enable();
  for (int type : {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16}) {
    UnitSwishGraph f(type);
    ASSERT_STATUS_OK(f.graph.Resolve());
    auto proto = f.model.ToProto();
    auto* sparse = proto.mutable_graph()->add_sparse_initializer();
    sparse->add_dims(2);
    sparse->add_dims(17);
    auto* values = sparse->mutable_values();
    values->set_name(f.x->Name());
    values->set_data_type(type);
    values->add_dims(1);
    if (type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      values->add_float_data(0.25f);
    else
      values->add_int32_data(BFloat16(0.25f).val);
    auto* indices = sparse->mutable_indices();
    indices->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    indices->add_dims(1);
    indices->add_int64_data(7);
    std::shared_ptr<Model> loaded;
    ASSERT_STATUS_OK(Model::Load(proto, loaded, nullptr, logger));
    ASSERT_TRUE(loaded->MainGraph().IsSparseInitializer(f.x->Name()));
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(loaded->MainGraph()), offers));
    EXPECT_TRUE(offers.empty());
    EXPECT_EQ(lookup.calls, 0u);
  }
}
#endif

struct MobiusGraph {
  Model model{"mobius-host", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 24}, {kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger()};
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder{graph};
  int64_t channels;
  int64_t kernel;
  int64_t dilation;
  int type;
  NodeArg* x;
  NodeArg* h;
  NodeArg* w;
  NodeArg* bias;
  NodeArg* index = builder.MakeScalarInitializer<int64_t>(2);
  NodeArg* window;
  NodeArg* vector_shape = builder.MakeInitializer<int64_t>({1}, {1});
  NodeArg* axes = builder.MakeInitializer<int64_t>({1}, {2});
  NodeArg* bias_shape = builder.MakeInitializer<int64_t>({3}, {1, -1, 1});
  Node* conv = nullptr;
  Node* swish = nullptr;
  NodeArg* c = nullptr;
  NodeArg* y = nullptr;
  NodeArg* state = nullptr;

  explicit MobiusGraph(int64_t c_count = 10240, int64_t k = 4, int64_t d = 1,
                       int element = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16)
      : channels(c_count), kernel(k), dilation(d), type(element) {
    auto initialize = [&]<typename T>() {
      x = builder.MakeInput<T>(std::vector<int64_t>{-1, channels, -1}, "x");
      h = builder.MakeInput<T>(std::vector<int64_t>{-1, channels, (kernel - 1) * dilation}, "h");
      w = builder.MakeInitializer<T>({channels, 1, kernel},
                                     std::vector<T>(static_cast<size_t>(channels * kernel), T(0.125f)));
      bias = builder.MakeInitializer<T>({channels}, std::vector<T>(static_cast<size_t>(channels), T(0.25f)));
    };
    if (type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      initialize.template operator()<float>();
    else if (type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)
      initialize.template operator()<MLFloat16>();
    else
      initialize.template operator()<BFloat16>();
    window = builder.MakeScalarInitializer<int64_t>((kernel - 1) * dilation);
  }

  void Branch(bool folded = false, bool zero_tail = false, bool convolved = false) {
    ORT_ENFORCE(!folded || !convolved);
    if (folded && graph.GetConstantInitializer(bias->Name(), false)->dims_size() == 1) {
      graph.RemoveInitializedTensor(bias->Name());
      auto initialize = [&]<typename T>() {
        std::vector<T> values;
        for (int64_t i = 0; i < channels; ++i) values.emplace_back(static_cast<float>(i % 19 - 9) / 17.0f);
        bias = builder.MakeInitializer<T>({1, channels, 1}, values);
      };
      if (type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
        initialize.template operator()<float>();
      else if (type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)
        initialize.template operator()<MLFloat16>();
      else
        initialize.template operator()<BFloat16>();
    }
    c = builder.MakeIntermediate();
    builder.AddNode("Concat", {h, x}, {c}).AddAttribute("axis", int64_t{2});
    auto* shape = builder.MakeIntermediate();
    builder.AddNode("Shape", {c}, {shape});
    auto* length = builder.MakeIntermediate();
    builder.AddNode("Gather", {shape, index}, {length});
    NodeArg* start1 = nullptr;
    if (!zero_tail) {
      auto* start = builder.MakeIntermediate();
      builder.AddNode("Sub", {length, window}, {start});
      start1 = builder.MakeIntermediate();
      builder.AddNode("Reshape", {start, vector_shape}, {start1});
    }
    auto* end1 = builder.MakeIntermediate();
    builder.AddNode("Reshape", {length, vector_shape}, {end1});
    state = builder.MakeOutput();
    builder.AddNode("Slice", {c, zero_tail ? end1 : start1, end1, axes}, {state});
    if (zero_tail) {
      auto state_type = *h->TypeAsProto();
      state_type.mutable_tensor_type()->mutable_shape()->mutable_dim(2)->set_dim_value(0);
      ORT_THROW_IF_ERROR(state->UpdateTypeAndShape(state_type, false, true,
                                                   DefaultLoggingManager().DefaultLogger()));
    }
    auto* z = builder.MakeIntermediate();
    conv = &builder.AddNode("Conv", convolved ? std::vector<NodeArg*>{c, w, bias} : std::vector<NodeArg*>{c, w}, {z});
    conv->AddAttribute("group", channels);
    if (dilation != 1) conv->AddAttribute("dilations", std::vector<int64_t>{dilation});
    auto* a = z;
    if (!convolved) {
      auto* bias3d = bias;
      if (!folded) {
        bias3d = builder.MakeIntermediate();
        builder.AddNode("Reshape", {bias, bias_shape}, {bias3d});
      }
      a = builder.MakeIntermediate();
      builder.AddNode("Add", {z, bias3d}, {a});
    }
    y = builder.MakeOutput();
    swish = &builder.AddNode("Swish", {a}, {y});
  }

  void OwnBias() {
    const auto proto = *graph.GetConstantInitializer(bias->Name(), false);
    graph.RemoveInitializedTensor(bias->Name());
    graph_utils::AddInitializerWithOrtValue(graph, proto);
  }
};

void SetRuntimeDimensions(MobiusGraph& fixture, std::optional<int64_t> batch,
                          std::optional<int64_t> sequence) {
  for (auto& node : fixture.graph.Nodes()) {
    for (auto* output : node.MutableOutputDefs()) output->ClearShape();
  }
  auto shape = [&](std::optional<int64_t> length) {
    ONNX_NAMESPACE::TensorShapeProto result;
    auto* b = result.add_dim();
    if (batch) b->set_dim_value(*batch);
    result.add_dim()->set_dim_value(fixture.channels);
    auto* s = result.add_dim();
    if (length) s->set_dim_value(*length);
    return result;
  };
  fixture.x->SetShape(shape(sequence));
  fixture.h->SetShape(shape((fixture.kernel - 1) * fixture.dilation));
}

class TargetLookup final : public IExecutionProvider::IKernelLookup {
 public:
  bool supported = true;
  int type = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
  int64_t channels = 10240;
  int64_t dilation = 1;
  mutable size_t calls = 0;
  KernelCreateInfo kernel;

  const KernelCreateInfo* LookUpKernel(const Node& node) const override {
    ++calls;
    EXPECT_EQ(node.OpType(), "CausalConvWithState");
    EXPECT_EQ(node.Domain(), kMSDomain);
    EXPECT_EQ(node.SinceVersion(), 1);
    EXPECT_EQ(node.InputDefs().size(), 4u);
    EXPECT_EQ(node.OutputDefs().size(), 2u);
    for (const auto* input : node.InputDefs()) {
      EXPECT_EQ(input->TypeAsProto()->tensor_type().elem_type(), type);
    }
    EXPECT_EQ(node.InputDefs()[2]->Shape()->dim_size(), 1);
    EXPECT_EQ(node.InputDefs()[2]->Shape()->dim(0).dim_value(), channels);
    EXPECT_EQ(node.GetAttributes().at("dilation").i(), dilation);
    return supported ? &kernel : nullptr;
  }
};

class CUDARuntimeFusionHostTest : public ::testing::Test {
 protected:
  SessionOptions options;
  CPUExecutionProvider cpu{CPUExecutionProviderInfo{}};
  TargetLookup lookup;
  const logging::Logger& logger = DefaultLoggingManager().DefaultLogger();

  Status Select(const GraphViewer& graph, std::vector<std::unique_ptr<ComputeCapability>>& offers,
                const std::string& rule = kMobiusBf16Rule) {
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    return cuda_runtime_fusion::Select(graph, lookup, registry, rule, offers);
  }

  void Enable() { ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kCausalConvConfig, "1")); }
};

TEST_F(CUDARuntimeFusionHostTest, ContractFamilyIsIndependentOfDimensionsAndDtype) {
  Enable();
  for (int type : {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                   ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16}) {
    for (auto dims : {std::array<int64_t, 3>{1, 1, 3}, {3, 2, 1}, {7, 3, 2}, {35, 6, 3}}) {
      for (bool folded : {false, true}) {
        SCOPED_TRACE(MakeString(type, "/", dims[0], "/", dims[1], "/", dims[2], "/", folded));
        MobiusGraph fixture(dims[0], dims[1], dims[2], type);
        fixture.Branch(folded);
        const auto bias_bits = fixture.graph.GetConstantInitializer(fixture.bias->Name(), false)->raw_data();
        if (folded) fixture.OwnBias();
        ASSERT_STATUS_OK(fixture.graph.Resolve());
        OrtValue original_bias;
        if (folded) {
          // Small tensors stay embedded even when requesting graph-owned storage.
          (void)fixture.graph.GetOrtValueInitializer(fixture.bias->Name(), original_bias, false);
        }
        lookup.type = type;
        lookup.channels = dims[0];
        lookup.dilation = dims[2];
        const auto before = fixture.graph.ToGraphProto().SerializeAsString();
        std::vector<std::unique_ptr<ComputeCapability>> offers;
        ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
        ASSERT_EQ(offers.size(), 1u);
        EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
        GraphOptimizerRegistry registry(&options, &cpu, &logger);
        auto& selected = *offers[0]->nodes_to_optimize[0];
        ASSERT_STATUS_OK(selected.optimization_func(fixture.graph, selected, *offers[0], registry));
        ASSERT_EQ(fixture.graph.NumberOfNodes(), 1);
        const auto& target = *fixture.graph.Nodes().begin();
        EXPECT_EQ(target.GetAttributes().at("dilation").i(), dims[2]);
        EXPECT_EQ(target.OutputDefs()[0], fixture.y);
        EXPECT_EQ(target.OutputDefs()[1], fixture.state);
        if (folded) {
          OrtValue flat;
          if (fixture.graph.GetOrtValueInitializer(target.InputDefs()[2]->Name(), flat, false)) {
            EXPECT_EQ(flat.Get<Tensor>().Shape(), (TensorShape{dims[0]}));
            EXPECT_EQ(std::memcmp(flat.Get<Tensor>().DataRaw(), bias_bits.data(), bias_bits.size()), 0);
          } else {
            const auto* proto = fixture.graph.GetConstantInitializer(target.InputDefs()[2]->Name(), false);
            ASSERT_NE(proto, nullptr);
            EXPECT_EQ(proto->raw_data(), bias_bits);
            ASSERT_EQ(proto->dims_size(), 1);
            EXPECT_EQ(proto->dims(0), dims[0]);
          }
          if (original_bias.IsAllocated()) {
            EXPECT_EQ(original_bias.Get<Tensor>().Shape(), (TensorShape{1, dims[0], 1}));
            ASSERT_TRUE(flat.IsAllocated());
            EXPECT_NE(flat.Get<Tensor>().DataRaw(), original_bias.Get<Tensor>().DataRaw());
          }
        }
      }
    }
  }
}

TEST_F(CUDARuntimeFusionHostTest, KnownNativeDimensionAndActivationByteLimitsDeclineBeforeLookup) {
  Enable();
  const int64_t limit = std::numeric_limits<int>::max();
  for (auto dims : {std::array<int64_t, 3>{limit + 1, 1, 1},
                    {0, 1, limit + 1},
                    {limit, 3, limit}}) {
    SCOPED_TRACE(MakeString(dims[0], "/", dims[1], "/", dims[2]));
    MobiusGraph fixture(dims[1], 1, 1, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    fixture.Branch();
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    SetRuntimeDimensions(fixture, dims[0], dims[2]);
    lookup.type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    lookup.channels = dims[1];
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    const auto calls = lookup.calls;
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_TRUE(offers.empty());
    EXPECT_EQ(lookup.calls, calls);
    EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
  }
}

TEST_F(CUDARuntimeFusionHostTest, PartiallyKnownNativeByteLimitsDeclineBeforeLookup) {
  Enable();
  const int64_t limit = std::numeric_limits<int>::max();
  for (bool state_overflow : {true, false}) {
    SCOPED_TRACE(state_overflow);
    if (!state_overflow &&
        static_cast<uint64_t>(limit) > std::numeric_limits<size_t>::max() / sizeof(float)) continue;
    MobiusGraph fixture(state_overflow ? 3 : 1, state_overflow ? 2 : 1,
                        state_overflow ? limit : 1, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    fixture.Branch();
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    if (!state_overflow) {
      // Only headers describe these large learned tensors; no files or payloads are created.
      fixture.channels = limit;
      fixture.conv->GetMutableAttributes().at("group").set_i(limit);
      for (auto* value : {fixture.w, fixture.bias}) {
        auto* proto = const_cast<ONNX_NAMESPACE::TensorProto*>(
            fixture.graph.GetConstantInitializer(value->Name(), false));
        proto->set_dims(0, limit);
        proto->clear_raw_data();
        ExternalDataInfo::SetExternalLocationToProto("not-opened-native-limits.data", 0,
                                                     static_cast<size_t>(limit) * sizeof(float), *proto);
        auto shape = *value->Shape();
        shape.mutable_dim(0)->set_dim_value(limit);
        value->SetShape(shape);
      }
    }
    SetRuntimeDimensions(fixture, state_overflow ? std::optional<int64_t>{limit} : std::nullopt,
                         state_overflow ? std::nullopt : std::optional<int64_t>{limit});
    lookup.type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    lookup.channels = fixture.channels;
    lookup.dilation = fixture.dilation;
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    const auto calls = lookup.calls;
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_TRUE(offers.empty());
    EXPECT_EQ(lookup.calls, calls);
    EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
  }
}

TEST_F(CUDARuntimeFusionHostTest, UnknownRuntimeDimensionsDoNotInventNativeOverflow) {
  Enable();
  const int64_t limit = std::numeric_limits<int>::max();
  const std::array<std::array<std::optional<int64_t>, 2>, 5> dimensions{{{std::nullopt, std::nullopt}, {0, std::nullopt}, {limit, std::nullopt}, {std::nullopt, 1}, {std::nullopt, limit}}};
  for (const auto& dims : dimensions) {
    MobiusGraph fixture(3, 1, 1, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    fixture.Branch();
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    SetRuntimeDimensions(fixture, dims[0], dims[1]);
    lookup.type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    lookup.channels = 3;
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    const auto calls = lookup.calls;
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_EQ(offers.size(), 1u);
    EXPECT_EQ(lookup.calls, calls + 1);
    EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
  }
}

TEST_F(CUDARuntimeFusionHostTest, OmittedConvGroupDefaultsToOneForSingleChannel) {
  Enable();
  MobiusGraph fixture(1, 3, 2);
  fixture.Branch();
  fixture.conv->GetMutableAttributes().erase("group");
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  ASSERT_EQ(fixture.conv->GetAttributes().at("group").i(), 1);
  lookup.channels = 1;
  lookup.dilation = 2;
  const auto before = fixture.graph.ToGraphProto().SerializeAsString();
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
  EXPECT_EQ(offers.size(), 1u);
  EXPECT_EQ(lookup.calls, 1u);
  EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
}

TEST_F(CUDARuntimeFusionHostTest, DisabledAndBasicDeclineButMalformedPolicyStillErrors) {
  Enable();
  MobiusGraph fixture(3, 6, 2);
  fixture.Branch();
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  for (auto level : {TransformerLevel::Default, TransformerLevel::Level1}) {
    options.graph_optimization_level = level;
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_TRUE(offers.empty());
    EXPECT_EQ(lookup.calls, 0u);
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kCausalConvConfig, "true"));
    EXPECT_FALSE(Select(GraphViewer(fixture.graph), offers).IsOK());
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kCausalConvConfig, "1"));
  }
}

TEST_F(CUDARuntimeFusionHostTest, NormalizedZeroTailSupportsBothBiasFormsAndAllDtypes) {
  Enable();
  for (int type : {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                   ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16}) {
    for (bool folded : {false, true}) {
      MobiusGraph fixture(7, 1, 3, type);
      fixture.Branch(folded, true);
      const auto window_name = fixture.window->Name();
      ASSERT_STATUS_OK(fixture.graph.Resolve());
      EXPECT_FALSE(fixture.graph.IsInitializedTensor(window_name));
      lookup.type = type;
      lookup.channels = 7;
      lookup.dilation = 3;
      const auto before = fixture.graph.ToGraphProto().SerializeAsString();
      std::vector<std::unique_ptr<ComputeCapability>> offers;
      ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
      ASSERT_EQ(offers.size(), 1u);
      EXPECT_EQ(offers[0]->sub_graph->nodes.size(), folded ? 8u : 9u);
      EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
      GraphOptimizerRegistry registry(&options, &cpu, &logger);
      auto& selected = *offers[0]->nodes_to_optimize[0];
      ASSERT_STATUS_OK(selected.optimization_func(fixture.graph, selected, *offers[0], registry));
      ASSERT_EQ(fixture.graph.NumberOfNodes(), 1);
      const auto& target = *fixture.graph.Nodes().begin();
      EXPECT_EQ(target.OutputDefs()[0], fixture.y);
      EXPECT_EQ(target.OutputDefs()[1], fixture.state);
      EXPECT_EQ(target.OutputDefs()[1]->Shape()->dim(2).dim_value(), 0);
    }
  }
}

TEST_F(CUDARuntimeFusionHostTest, NormalizedZeroTailRejectsPositivePadAndUnequalSliceBindings) {
  Enable();
  for (int scenario = 0; scenario < 6; ++scenario) {
    SCOPED_TRACE(scenario);
    MobiusGraph fixture(7, scenario == 0 ? 2 : 1, 2);
    fixture.Branch(false, true);
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    if (scenario == 1) {
      auto* slice = fixture.graph.GetNode(fixture.graph.GetProducerNode(fixture.state->Name())->Index());
      const auto* reshape = fixture.graph.GetProducerNode(slice->InputDefs()[2]->Name());
      auto* other = fixture.builder.MakeIntermediate();
      fixture.builder.AddNode("Reshape", {fixture.graph.GetNodeArg(reshape->InputDefs()[0]->Name()), fixture.vector_shape}, {other});
      graph_utils::ReplaceNodeInput(*slice, 1, *other);
      ASSERT_STATUS_OK(fixture.graph.Resolve());
    }
    if (scenario == 2 || scenario == 3) {
      auto* value = scenario == 2 ? fixture.h : fixture.state;
      auto shape = *value->Shape();
      shape.mutable_dim(2)->set_dim_value(1);
      value->SetShape(shape);
    }
    if (scenario == 4) {
      auto* proto = const_cast<ONNX_NAMESPACE::TensorProto*>(
          fixture.graph.GetConstantInitializer(fixture.axes->Name(), false));
      proto->set_raw_data(std::string(sizeof(int64_t), '\0'));
    }
    if (scenario == 5) {
      fixture.state->ClearShape();
    }
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    const auto calls = lookup.calls;
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_TRUE(offers.empty());
    EXPECT_EQ(lookup.calls, calls);
    EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
  }
}

TEST_F(CUDARuntimeFusionHostTest, AcceptedTailAlternativeCannotChangeInEitherDirection) {
  Enable();
  for (bool originally_zero : {false, true}) {
    MobiusGraph fixture(7, 1, 3);
    fixture.Branch(true, originally_zero);
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    fixture.state->SetShape(*fixture.h->Shape());
    lookup.channels = 7;
    lookup.dilation = 3;
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    ASSERT_EQ(offers.size(), 1u);
    auto* slice = fixture.graph.GetNode(fixture.graph.GetProducerNode(fixture.state->Name())->Index());
    const auto* end_reshape = fixture.graph.GetProducerNode(slice->InputDefs()[2]->Name());
    if (originally_zero) {
      auto* window = fixture.builder.MakeScalarInitializer<int64_t>(0);
      auto* start = fixture.builder.MakeIntermediate();
      fixture.builder.AddNode("Sub", {fixture.graph.GetNodeArg(end_reshape->InputDefs()[0]->Name()), window}, {start});
      auto* start1 = fixture.builder.MakeIntermediate();
      fixture.builder.AddNode("Reshape", {start, fixture.vector_shape}, {start1});
      graph_utils::ReplaceNodeInput(*slice, 1, *start1);
    } else {
      const auto* start_reshape = fixture.graph.GetProducerNode(slice->InputDefs()[1]->Name());
      const auto* sub = fixture.graph.GetProducerNode(start_reshape->InputDefs()[0]->Name());
      const auto reshape_id = start_reshape->Index();
      const auto sub_id = sub->Index();
      graph_utils::RemoveNodeOutputEdges(fixture.graph, *fixture.graph.GetNode(reshape_id));
      graph_utils::ReplaceNodeInput(*slice, 1, *slice->MutableInputDefs()[2]);
      fixture.graph.RemoveNode(reshape_id);
      graph_utils::RemoveNodeOutputEdges(fixture.graph, *fixture.graph.GetNode(sub_id));
      fixture.graph.RemoveNode(sub_id);
    }
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    std::vector<std::unique_ptr<ComputeCapability>> fresh;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), fresh));
    ASSERT_EQ(fresh.size(), 1u);  // Both forms are legal; only the selected form may be accepted.
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    auto& selected = *offers[0]->nodes_to_optimize[0];
    EXPECT_FALSE(selected.optimization_func(fixture.graph, selected, *offers[0], registry).IsOK());
    EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
  }
}

TEST_F(CUDARuntimeFusionHostTest, NormalizedConvBiasSupportsAllDtypesAndBothTails) {
  Enable();
  for (int type : {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                   ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16}) {
    for (bool zero_tail : {false, true}) {
      MobiusGraph fixture(7, zero_tail ? 1 : 6, 3, type);
      fixture.Branch(false, zero_tail, true);
      ASSERT_STATUS_OK(fixture.graph.Resolve());
      lookup.type = type;
      lookup.channels = 7;
      lookup.dilation = 3;
      std::vector<std::unique_ptr<ComputeCapability>> offers;
      ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
      ASSERT_EQ(offers.size(), 1u);
      EXPECT_EQ(offers[0]->sub_graph->nodes.size(), zero_tail ? 7u : 9u);
      const auto* original_y = fixture.y;
      const auto* original_state = fixture.state;
      GraphOptimizerRegistry registry(&options, &cpu, &logger);
      auto& selected = *offers[0]->nodes_to_optimize[0];
      ASSERT_STATUS_OK(selected.optimization_func(fixture.graph, selected, *offers[0], registry));
      const auto* target = fixture.graph.GetNode(offers[0]->sub_graph->nodes.front());
      EXPECT_EQ(target->InputDefs()[2], fixture.bias);
      EXPECT_EQ(target->OutputDefs()[0], original_y);
      EXPECT_EQ(target->OutputDefs()[1], original_state);
    }
  }
}

TEST_F(CUDARuntimeFusionHostTest, NormalizedConvBiasRejectsOverridableOrStaleBias) {
  Enable();
  for (bool stale : {false, true}) {
    MobiusGraph fixture(3, 6, 2);
    fixture.Branch(false, false, true);
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    lookup.channels = 3;
    lookup.dilation = 2;
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    ASSERT_EQ(offers.size(), 1u);
    if (stale) {
      auto* other = fixture.builder.MakeInitializer<BFloat16>({3}, std::vector<BFloat16>(3, BFloat16(0.25f)));
      graph_utils::ReplaceNodeInput(*fixture.conv, 2, *other);
    } else {
      auto inputs = fixture.graph.GetInputs();
      inputs.push_back(fixture.bias);
      fixture.graph.SetInputs(inputs);
    }
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    if (!stale) {
      std::vector<std::unique_ptr<ComputeCapability>> declined;
      ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), declined));
      EXPECT_TRUE(declined.empty());
    }
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    auto& selected = *offers[0]->nodes_to_optimize[0];
    EXPECT_FALSE(selected.optimization_func(fixture.graph, selected, *offers[0], registry).IsOK());
    EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
  }
}

TEST_F(CUDARuntimeFusionHostTest, AcceptedContractCannotReselectDilationBindingsOrBiasAlternative) {
  Enable();
  for (int scenario = 0; scenario < 3; ++scenario) {
    MobiusGraph fixture(3, 1, 3);
    fixture.Branch(true);
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    lookup.channels = 3;
    lookup.dilation = 3;
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    ASSERT_EQ(offers.size(), 1u);
    if (scenario == 0) fixture.conv->AddAttribute("dilations", std::vector<int64_t>{2});  // Still P=0.
    if (scenario == 1) {
      auto* replacement = fixture.builder.MakeInitializer<BFloat16>({3, 1, 1},
                                                                    std::vector<BFloat16>(3, BFloat16(0.125f)));
      graph_utils::ReplaceNodeInput(*fixture.conv, 1, *replacement);
    }
    if (scenario == 2) {
      auto* flat = fixture.builder.MakeInitializer<BFloat16>({3}, std::vector<BFloat16>(3, BFloat16(0.25f)));
      auto* shaped = fixture.builder.MakeIntermediate();
      fixture.builder.AddNode("Reshape", {flat, fixture.bias_shape}, {shaped});
      for (auto& node : fixture.graph.Nodes()) {
        if (node.OpType() == "Add") graph_utils::ReplaceNodeInput(node, 1, *shaped);
      }
    }
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    auto& selected = *offers[0]->nodes_to_optimize[0];
    EXPECT_FALSE(selected.optimization_func(fixture.graph, selected, *offers[0], registry).IsOK());
    EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
  }
}

TEST_F(CUDARuntimeFusionHostTest, EmbeddedTypedLearnedStorageSupportsAllThreeDtypes) {
  Enable();
  for (int type : {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                   ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16}) {
    MobiusGraph fixture(7, 6, 2, type);
    fixture.Branch(true);
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    for (auto* value : {fixture.w, fixture.bias}) {
      auto* proto = const_cast<ONNX_NAMESPACE::TensorProto*>(fixture.graph.GetConstantInitializer(value->Name(), false));
      const auto bytes = proto->raw_data();
      proto->clear_raw_data();
      if (type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        for (size_t i = 0; i < bytes.size(); i += sizeof(float)) {
          float data;
          std::memcpy(&data, bytes.data() + i, sizeof(data));
          proto->add_float_data(data);
        }
      } else {
        for (size_t i = 0; i < bytes.size(); i += sizeof(uint16_t)) {
          uint16_t data;
          std::memcpy(&data, bytes.data() + i, sizeof(data));
          proto->add_int32_data(data);
        }
      }
    }
    lookup.type = type;
    lookup.channels = 7;
    lookup.dilation = 2;
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    ASSERT_EQ(offers.size(), 1u);
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    auto& selected = *offers[0]->nodes_to_optimize[0];
    ASSERT_STATUS_OK(selected.optimization_func(fixture.graph, selected, *offers[0], registry));
  }
}

TEST_F(CUDARuntimeFusionHostTest, GenericContractNearMissesDeclineWithoutLookupOrMutation) {
  Enable();
  for (int scenario = 0; scenario < 12; ++scenario) {
    SCOPED_TRACE(scenario);
    MobiusGraph fixture(7, 6, 2);
    fixture.Branch();
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    auto set_ints = [&](const char* name, std::initializer_list<int64_t> values) {
      ONNX_NAMESPACE::AttributeProto attr;
      attr.set_name(name);
      attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
      for (auto value : values) attr.add_ints(value);
      fixture.conv->GetMutableAttributes()[name] = attr;
    };
    if (scenario == 0) fixture.conv->GetMutableAttributes().at("group").set_i(6);
    if (scenario == 1) set_ints("strides", {2});
    if (scenario == 2) set_ints("pads", {0, 1});
    if (scenario == 3) set_ints("kernel_shape", {5});
    if (scenario == 4) set_ints("dilations", {0});
    if (scenario == 5) set_ints("dilations", {1, 2});
    if (scenario == 6) set_ints("dilations", {std::numeric_limits<int>::max()});
    if (scenario == 7) {
      auto* window = const_cast<ONNX_NAMESPACE::TensorProto*>(
          fixture.graph.GetConstantInitializer(fixture.window->Name(), false));
      window->clear_raw_data();
      window->add_int64_data(9);
    }
    if (scenario >= 8) {
      auto* weight = const_cast<ONNX_NAMESPACE::TensorProto*>(
          fixture.graph.GetConstantInitializer(fixture.w->Name(), false));
      if (scenario == 8) weight->set_dims(2, 0);
      if (scenario == 9) weight->set_dims(0, int64_t{std::numeric_limits<int>::max()} + 1);
      if (scenario == 10) weight->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
      if (scenario == 11) weight->mutable_raw_data()->pop_back();
    }
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    const auto calls = lookup.calls;
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_TRUE(offers.empty());
    EXPECT_EQ(lookup.calls, calls);
    EXPECT_EQ(fixture.graph.ToGraphProto().SerializeAsString(), before);
  }
}

TEST_F(CUDARuntimeFusionHostTest, UnsupportedCapabilityConsumerDeclinesBothFormsBeforeLookup) {
  Enable();
  struct ForbiddenLookup final : IExecutionProvider::IKernelLookup {
    const KernelCreateInfo* LookUpKernel(const Node&) const override {
      ADD_FAILURE() << "Unsupported capability consumer must not perform scratch kernel lookup.";
      return nullptr;
    }
  } forbidden_lookup;
  for (bool folded : {false, true}) {
    SCOPED_TRACE(folded);
    MobiusGraph fixture;
    fixture.Branch(folded);
    if (folded) fixture.OwnBias();
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    ASSERT_EQ(fixture.graph.NumberOfNodes(), folded ? 10 : 11);
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    const auto initializer_count = fixture.graph.GetAllInitializedTensors().size();
    OrtValue owned_bias;
    if (folded) {
      ASSERT_TRUE(fixture.graph.GetOrtValueInitializer(fixture.bias->Name(), owned_bias, false));
    }
    GraphOptimizerRegistry registry(&options, &cpu, &logger, false);
    GraphViewer viewer(fixture.graph);
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    for (int repeat = 0; repeat < 3; ++repeat) {
      ASSERT_STATUS_OK(cuda_runtime_fusion::Select(viewer, forbidden_lookup, registry, kMobiusBf16Rule, offers));
      EXPECT_TRUE(offers.empty());
      EXPECT_EQ(fixture.graph.GetAllInitializedTensors().size(), initializer_count);
      EXPECT_EQ(fixture.graph.ToGraphProto().SerializeAsString(), before);
      if (folded) {
        EXPECT_EQ(owned_bias.Get<Tensor>().Shape(), (TensorShape{1, 10240, 1}));
      }
    }
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kCausalConvConfig, "true"));
    EXPECT_FALSE(cuda_runtime_fusion::Select(viewer, forbidden_lookup, registry, kMobiusBf16Rule, offers).IsOK());
    EXPECT_TRUE(offers.empty());  // Consumer decline must not hide a malformed opt-in setting.
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kCausalConvConfig, "1"));
  }
}

TEST_F(CUDARuntimeFusionHostTest, FoldedRunOnceStagesRankThreeBiasBeforeCallbackFinalization) {
  MobiusGraph fixture;
  fixture.Branch(true);
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  ASSERT_EQ(fixture.graph.NumberOfNodes(), 10);
  rewrite::Rule rule{
      "staging-gate",
      R"((x,h,w,bias,index,window,vector_shape,axes)=>{
        c=Concat(h,x); shape=Shape(c); length=Gather(shape,index);
        start=Sub(length,window); start1=Reshape(start,vector_shape);
        end1=Reshape(length,vector_shape); state=Slice(c,start1,end1,axes);
        z=Conv(c,w); a=Add(z,bias); y=Swish(a); return y,state
      })",
      R"((x,w,bias,h)=>{
        y,state=com.microsoft.CausalConvWithState(x,w,bias,h,
          ndim=1,activation="silu",channels_last=0,dilation=1,state_window=0);
        return y,state
      })",
      {},
      {{kOnnxDomain, 24, 24}, {kMSDomain, 1, 1}}};
  rewrite::PatternRewriter rewriter({std::move(rule)});
  ComputeCapability parent(std::make_unique<IndexedSubGraph>());
  for (const auto& node : fixture.graph.Nodes()) parent.sub_graph->nodes.push_back(node.Index());
  ComputeCapability selected(std::make_unique<IndexedSubGraph>());
  selected.sub_graph->nodes = parent.sub_graph->nodes;
  const auto anchor = fixture.swish->Index();
  selected.optimization_func = [&](Graph& graph, const ComputeCapability& region, ComputeCapability& result,
                                   const GraphOptimizerRegistry&) -> Status {
    InlinedHashSet<NodeIndex> ids(region.sub_graph->nodes.begin(), region.sub_graph->nodes.end());
    rewrite::Statistics statistics;
    ORT_RETURN_IF_ERROR(rewriter.RunOnce(graph, statistics, ids, anchor));
    ORT_RETURN_IF_NOT(statistics.rewrites == 1 && ids.size() == 1 && graph.GraphResolveNeeded(),
                      "RunOnce did not stage one unresolved target.");
    auto* target = graph.GetNode(*ids.begin());
    ORT_RETURN_IF_NOT(target && target->InputDefs()[2] == fixture.bias &&
                          target->InputDefs()[2]->Shape()->dim_size() == 3,
                      "RunOnce prematurely changed or validated the source bias.");
    auto* flat = fixture.builder.MakeInitializer<BFloat16>({10240}, std::vector<BFloat16>(10240, BFloat16(0.25f)));
    graph_utils::ReplaceNodeInput(*target, 2, *flat);
    ORT_RETURN_IF_ERROR(graph.Resolve());
    result.sub_graph->nodes.assign(ids.begin(), ids.end());
    return Status::OK();
  };
  GraphOptimizerRegistry registry(&options, &cpu, &logger);
  ASSERT_STATUS_OK(selected.optimization_func(fixture.graph, selected, parent, registry));
  ASSERT_EQ(fixture.graph.NumberOfNodes(), 1);
  const auto* target = fixture.graph.GetNode(parent.sub_graph->nodes[0]);
  EXPECT_EQ(target->InputDefs()[2]->Shape()->dim_size(), 1);
  EXPECT_EQ(target->OutputDefs()[0], fixture.y);
  EXPECT_EQ(target->OutputDefs()[1], fixture.state);
}

TEST_F(CUDARuntimeFusionHostTest, FoldedDiscoveryIsReadOnlyAndAcceptanceOwnsIndependentFlatBits) {
  Enable();
  for (bool in_memory : {false, true}) {
    SCOPED_TRACE(in_memory);
    MobiusGraph fixture;
    const auto original_bias_name = fixture.bias->Name();
    fixture.Branch(true);
    const auto bias_name = fixture.bias->Name();
    const auto bits = fixture.graph.GetConstantInitializer(fixture.bias->Name(), false)->raw_data();
    if (in_memory) fixture.OwnBias();
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    ASSERT_EQ(fixture.graph.NumberOfNodes(), 10);
    EXPECT_FALSE(fixture.graph.IsInitializedTensor(original_bias_name));
    const auto count = fixture.graph.GetAllInitializedTensors().size();
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    OrtValue source_owned;
    EXPECT_EQ(fixture.graph.GetOrtValueInitializer(fixture.bias->Name(), source_owned, false), in_memory);
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    for (int repeat = 0; repeat < 3; ++repeat) {
      offers.clear();
      ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
      ASSERT_EQ(offers.size(), 1u);
      EXPECT_EQ(offers[0]->sub_graph->nodes.size(), 10u);
      EXPECT_EQ(offers[0]->nodes_to_optimize[0]->sub_graph->nodes, offers[0]->sub_graph->nodes);
      EXPECT_EQ(fixture.graph.GetAllInitializedTensors().size(), count);
      EXPECT_EQ(fixture.graph.ToGraphProto().SerializeAsString(), before);
      if (in_memory) EXPECT_EQ(source_owned.Get<Tensor>().Shape(), (TensorShape{1, 10240, 1}));
    }
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    auto& selected = *offers[0]->nodes_to_optimize[0];
    ASSERT_STATUS_OK(selected.optimization_func(fixture.graph, selected, *offers[0], registry));
    ASSERT_EQ(offers[0]->sub_graph->nodes.size(), 1u);
    ASSERT_EQ(fixture.graph.NumberOfNodes(), 1);
    const auto* target = fixture.graph.GetNode(offers[0]->sub_graph->nodes[0]);
    EXPECT_EQ(target->OutputDefs()[0], fixture.y);
    EXPECT_EQ(target->OutputDefs()[1], fixture.state);
    const auto* flat = target->InputDefs()[2];
    EXPECT_NE(flat->Name(), bias_name);
    OrtValue flat_owned;
    ASSERT_TRUE(fixture.graph.GetOrtValueInitializer(flat->Name(), flat_owned, false));
    EXPECT_EQ(flat_owned.Get<Tensor>().Shape(), (TensorShape{10240}));
    EXPECT_EQ(std::memcmp(flat_owned.Get<Tensor>().DataRaw(), bits.data(), bits.size()), 0);
    if (in_memory) {
      EXPECT_EQ(source_owned.Get<Tensor>().Shape(), (TensorShape{1, 10240, 1}));
      EXPECT_NE(flat_owned.Get<Tensor>().DataRaw(), source_owned.Get<Tensor>().DataRaw());
    }
  }
}

TEST_F(CUDARuntimeFusionHostTest, FoldedExternalBiasDiscoveryDoesNotOpenFilesAndAcceptanceCopiesValidData) {
  Enable();
  struct FileCleanup {
    std::filesystem::path path{"cuda_runtime_fusion_host_bias.data"};
    ~FileCleanup() {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } cleanup;
  for (bool available : {false, true}) {
    MobiusGraph fixture;
    fixture.Branch(true);
    auto proto = *fixture.graph.GetConstantInitializer(fixture.bias->Name(), false);
    const auto bits = proto.raw_data();
    if (available) {
      std::ofstream file(cleanup.path, std::ios::binary);
      file.write(bits.data(), static_cast<std::streamsize>(bits.size()));
      ASSERT_TRUE(file.good());
    }
    proto.clear_raw_data();
    ExternalDataInfo::SetExternalLocationToProto(available ? cleanup.path : "nonexistent-fusion-bias.data",
                                                 0, bits.size(), proto);
    fixture.graph.RemoveInitializedTensor(proto.name());
    fixture.graph.AddInitializedTensor(proto);
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    ASSERT_EQ(offers.size(), 1u);  // Includes a nonexistent file: discovery cannot load the payload.
    EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
    if (!available) continue;
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    auto& selected = *offers[0]->nodes_to_optimize[0];
    ASSERT_STATUS_OK(selected.optimization_func(fixture.graph, selected, *offers[0], registry));
    const auto* target = fixture.graph.GetNode(offers[0]->sub_graph->nodes[0]);
    OrtValue flat;
    ASSERT_TRUE(fixture.graph.GetOrtValueInitializer(target->InputDefs()[2]->Name(), flat, false));
    EXPECT_EQ(flat.Get<Tensor>().Shape(), (TensorShape{10240}));
    EXPECT_EQ(std::memcmp(flat.Get<Tensor>().DataRaw(), bits.data(), bits.size()), 0);
  }
}

TEST_F(CUDARuntimeFusionHostTest, FoldedOverridableOrInconsistentHeaderAndStorageDecline) {
  Enable();
  for (int scenario = 0; scenario < 13; ++scenario) {
    SCOPED_TRACE(scenario);
    MobiusGraph fixture;
    fixture.Branch(true);
    if (scenario == 0) fixture.graph.SetInputs({fixture.x, fixture.h, fixture.bias});
    if (scenario == 5 || scenario == 6) fixture.OwnBias();
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    auto* proto = const_cast<ONNX_NAMESPACE::TensorProto*>(
        fixture.graph.GetConstantInitializer(fixture.bias->Name(), false));
    if (scenario == 1) proto->mutable_dims()->RemoveLast();
    if (scenario == 2) proto->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    if (scenario == 3) proto->clear_raw_data();
    if (scenario == 4) {
      proto->clear_raw_data();
      ExternalDataInfo::SetExternalLocationToProto(utils::kTensorProtoNativeEndianMemoryAddressTag, 1, 20480, *proto);
    }
    if (scenario == 5) {
      for (auto& entry : *proto->mutable_external_data()) {
        if (entry.key() == "offset") entry.set_value("1");
      }
    }
    if (scenario == 6) {
      OrtValue owned;
      ASSERT_TRUE(fixture.graph.GetOrtValueInitializer(fixture.bias->Name(), owned, false));
      owned.GetMutable<Tensor>()->Reshape(TensorShape({10240}));
    }
    if (scenario >= 7 && scenario <= 9) {
      proto->clear_raw_data();
      ExternalDataInfo::SetExternalLocationToProto("not-opened.data", 0, scenario == 7 ? 2 : 20480, *proto);
      if (scenario == 8) proto->add_int32_data(0);
      if (scenario == 9) {
        for (auto& entry : *proto->mutable_external_data()) {
          if (entry.key() == "offset") entry.set_value("not-a-number");
        }
      }
    }
    if (scenario == 10) proto->add_int32_data(0);  // Conflicting raw and typed payloads.
    if (scenario == 11) proto->add_float_data(0.0f);
    if (scenario == 12) proto->mutable_segment()->set_begin(0);
    const auto count = fixture.graph.GetAllInitializedTensors().size();
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    const auto calls = lookup.calls;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_TRUE(offers.empty());
    EXPECT_EQ(lookup.calls, calls);
    EXPECT_EQ(fixture.graph.GetAllInitializedTensors().size(), count);
  }
}

TEST_F(CUDARuntimeFusionHostTest, FoldedSharedBiasCopiesSurviveSourcePruningWithoutMutatingOtherConsumers) {
  Enable();
  MobiusGraph fixture;
  fixture.Branch(true);
  fixture.Branch(true);
  const auto bits = fixture.graph.GetConstantInitializer(fixture.bias->Name(), false)->raw_data();
  const auto bias_name = fixture.bias->Name();
  fixture.OwnBias();
  OrtValue source;
  ASSERT_TRUE(fixture.graph.GetOrtValueInitializer(bias_name, source, false));
  auto* unrelated_output = fixture.builder.MakeOutput();
  auto& unrelated = fixture.builder.AddNode("Identity", {fixture.bias}, {unrelated_output});
  const auto unrelated_id = unrelated.Index();
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  GraphViewer viewer(fixture.graph);
  ASSERT_STATUS_OK(Select(viewer, offers));
  ASSERT_EQ(offers.size(), 2u);
  size_t offer_index = 0;
  for (auto id : viewer.GetNodesInTopologicalOrder()) {
    if (fixture.graph.GetNode(id)->OpType() != "Swish") continue;
    const auto& ids = offers[offer_index++]->sub_graph->nodes;
    EXPECT_NE(std::find(ids.begin(), ids.end(), id), ids.end());
  }
  EXPECT_EQ(offer_index, offers.size());
  GraphOptimizerRegistry registry(&options, &cpu, &logger);
  InlinedVector<const NodeArg*> outputs;
  std::vector<OrtValue> flat_copies;
  for (auto& offer : offers) {
    EXPECT_EQ(offer->sub_graph->nodes.size(), 10u);
    auto& selected = *offer->nodes_to_optimize[0];
    ASSERT_STATUS_OK(selected.optimization_func(fixture.graph, selected, *offer, registry));
    const auto* target = fixture.graph.GetNode(offer->sub_graph->nodes[0]);
    EXPECT_EQ(target->OpType(), "CausalConvWithState");
    outputs.insert(outputs.end(), target->OutputDefs().begin(), target->OutputDefs().end());
    OrtValue flat;
    ASSERT_TRUE(fixture.graph.GetOrtValueInitializer(target->InputDefs()[2]->Name(), flat, false));
    flat_copies.push_back(flat);
    EXPECT_NE(flat.Get<Tensor>().DataRaw(), source.Get<Tensor>().DataRaw());
    EXPECT_EQ(fixture.graph.GetNode(unrelated_id)->InputDefs()[0], fixture.bias);
    EXPECT_EQ(source.Get<Tensor>().Shape(), (TensorShape{1, 10240, 1}));
  }
  ASSERT_EQ(fixture.graph.NumberOfNodes(), 3);
  EXPECT_NE(flat_copies[0].Get<Tensor>().DataRaw(), flat_copies[1].Get<Tensor>().DataRaw());
  fixture.graph.SetOutputs(outputs);
  fixture.graph.RemoveNode(unrelated_id);
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  EXPECT_FALSE(fixture.graph.IsInitializedTensor(bias_name));
  for (const auto& flat : flat_copies) {
    EXPECT_EQ(flat.Get<Tensor>().Shape(), (TensorShape{10240}));
    EXPECT_EQ(std::memcmp(flat.Get<Tensor>().DataRaw(), bits.data(), bits.size()), 0);
  }
}

TEST_F(CUDARuntimeFusionHostTest, FoldedStaleOrChangedSelectedRegionsNeverMaterializeBias) {
  Enable();
  for (int scenario = 0; scenario < 6; ++scenario) {
    SCOPED_TRACE(scenario);
    MobiusGraph fixture;
    fixture.Branch(true);
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    ASSERT_EQ(offers.size(), 1u);
    auto& selected = *offers[0]->nodes_to_optimize[0];
    if (scenario == 0) selected.sub_graph->nodes.pop_back();
    if (scenario == 1) selected.sub_graph->nodes.push_back(selected.sub_graph->nodes[0]);
    if (scenario == 2) fixture.conv->SetExecutionProviderType(kCpuExecutionProvider);
    if (scenario == 3) fixture.graph.RemoveNode(fixture.swish->Index());
    if (scenario == 4) {
      fixture.graph.SetInputs({fixture.x, fixture.h, fixture.bias});
      ASSERT_STATUS_OK(fixture.graph.Resolve());
    }
    if (scenario == 5) {
      auto* proto = const_cast<ONNX_NAMESPACE::TensorProto*>(
          fixture.graph.GetConstantInitializer(fixture.bias->Name(), false));
      proto->clear_raw_data();
    }
    const auto count = fixture.graph.GetAllInitializedTensors().size();
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    EXPECT_FALSE(selected.optimization_func(fixture.graph, selected, *offers[0], registry).IsOK());
    EXPECT_EQ(fixture.graph.GetAllInitializedTensors().size(), count);
    EXPECT_EQ(fixture.graph.ToGraphProto().SerializeAsString(), before);
  }
}

#if !defined(DISABLE_SPARSE_TENSORS)
TEST_F(CUDARuntimeFusionHostTest, FoldedSparseBiasDeclinesEvenAfterDenseNormalization) {
  Enable();
  MobiusGraph fixture;
  fixture.Branch(true);
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  auto proto = fixture.model.ToProto();
  auto* graph_proto = proto.mutable_graph();
  for (int i = 0; i < graph_proto->initializer_size(); ++i) {
    if (graph_proto->initializer(i).name() != fixture.bias->Name()) continue;
    graph_proto->mutable_initializer()->DeleteSubrange(i, 1);
    break;
  }
  auto* sparse = graph_proto->add_sparse_initializer();
  for (auto dim : {1, 10240, 1}) sparse->add_dims(dim);
  auto* values = sparse->mutable_values();
  values->set_name(fixture.bias->Name());
  values->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
  values->add_dims(1);
  const BFloat16 value(0.25f);
  values->set_raw_data(&value, sizeof(value));
  auto* indices = sparse->mutable_indices();
  indices->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices->add_dims(1);
  indices->add_int64_data(7);
  std::shared_ptr<Model> loaded;
  ASSERT_STATUS_OK(Model::Load(proto, loaded, nullptr, logger));
  ASSERT_TRUE(loaded->MainGraph().IsSparseInitializer(fixture.bias->Name()));
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(loaded->MainGraph()), offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 0u);
}
#endif

TEST_F(CUDARuntimeFusionHostTest, FoldedEarlierAssignmentAndOverlappingSinksDoNotMaterializeBias) {
  Enable();
  for (bool overlap : {false, true}) {
    MobiusGraph fixture;
    fixture.Branch(true);
    if (overlap) {
      fixture.builder.AddNode("Swish", {fixture.swish->MutableInputDefs()[0]}, {fixture.builder.MakeOutput()});
      fixture.builder.SetGraphOutputs();
    }
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    if (!overlap) fixture.conv->SetExecutionProviderType(kCpuExecutionProvider);
    const auto count = fixture.graph.GetAllInitializedTensors().size();
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_TRUE(offers.empty());
    EXPECT_EQ(lookup.calls, 0u);
    EXPECT_EQ(fixture.graph.GetAllInitializedTensors().size(), count);
    EXPECT_EQ(fixture.graph.ToGraphProto().SerializeAsString(), before);
  }
}

TEST_F(CUDARuntimeFusionHostTest, DefaultOffInvalidPolicyAndUnknownRule) {
  MobiusGraph fixture;
  fixture.Branch();
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  GraphViewer viewer(fixture.graph);
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(viewer, offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 0u);
  for (const char* invalid : {"", "true", "01", " 1", "2"}) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kCausalConvConfig, invalid));
    EXPECT_FALSE(Select(viewer, offers).IsOK());
    EXPECT_TRUE(offers.empty());
  }
  EXPECT_FALSE(Select(viewer, offers, "unknown").IsOK());
  offers.push_back(std::make_unique<ComputeCapability>());
  EXPECT_FALSE(Select(viewer, offers).IsOK());
  EXPECT_EQ(offers.size(), 1u);
}

TEST_F(CUDARuntimeFusionHostTest, ReadOnlyRepeatedDiscoveryAndExactAcceptedRegion) {
  Enable();
  MobiusGraph fixture;
  fixture.Branch();
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  const auto before = fixture.graph.ToGraphProto().SerializeAsString();
  const auto* original_y = fixture.y;
  const auto* original_state = fixture.state;
  std::vector<std::unique_ptr<ComputeCapability>> first, second;
  ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), first));
  ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), second));
  ASSERT_EQ(first.size(), 1u);
  ASSERT_EQ(second.size(), 1u);
  EXPECT_EQ(first[0]->sub_graph->nodes, second[0]->sub_graph->nodes);
  EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
  EXPECT_EQ(first[0]->sub_graph->nodes.size(), 11u);
  EXPECT_EQ(first[0]->sub_graph->GetMetaDef(), nullptr);
  ASSERT_EQ(first[0]->nodes_to_optimize.size(), 1u);
  auto& selected = *first[0]->nodes_to_optimize[0];
  GraphOptimizerRegistry registry(&options, &cpu, &logger);
  selected.sub_graph->nodes.pop_back();
  EXPECT_FALSE(selected.optimization_func(fixture.graph, selected, *first[0], registry).IsOK());
  EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
  selected.sub_graph->nodes = first[0]->sub_graph->nodes;
  ASSERT_STATUS_OK(selected.optimization_func(fixture.graph, selected, *first[0], registry));
  ASSERT_EQ(fixture.graph.NumberOfNodes(), 1);
  ASSERT_EQ(first[0]->sub_graph->nodes.size(), 1u);
  const auto* target = fixture.graph.GetNode(first[0]->sub_graph->nodes[0]);
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(target->OutputDefs()[0], original_y);
  EXPECT_EQ(target->OutputDefs()[1], original_state);
  EXPECT_EQ(target->InputDefs()[1], fixture.w);
  EXPECT_EQ(target->InputDefs()[2], fixture.bias);
  EXPECT_FALSE(selected.optimization_func(fixture.graph, selected, *first[0], registry).IsOK());
}

TEST_F(CUDARuntimeFusionHostTest, MissingKernelAndFilteredViewerDeclineWithoutMutation) {
  Enable();
  MobiusGraph fixture;
  fixture.Branch();
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  const auto before = fixture.graph.ToGraphProto().SerializeAsString();
  lookup.supported = false;
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 1u);
  IndexedSubGraph filter;
  filter.nodes.push_back(fixture.swish->Index());
  auto meta = std::make_unique<IndexedSubGraph::MetaDef>();
  meta->name = "filtered_swish";
  meta->domain = "test";
  meta->since_version = 1;
  meta->inputs = {fixture.swish->InputDefs()[0]->Name()};
  meta->outputs = {fixture.y->Name()};
  filter.SetMetaDef(std::move(meta));
  lookup.supported = true;
  ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph, filter), offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 1u);
  EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
}

TEST_F(CUDARuntimeFusionHostTest, NonUnitAndNonFiniteAlphaDecline) {
  Enable();
  for (float alpha : {0.5f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
    MobiusGraph fixture;
    fixture.Branch();
    fixture.swish->AddAttribute("alpha", alpha);
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_TRUE(offers.empty());
  }
  EXPECT_EQ(lookup.calls, 0u);
}

TEST_F(CUDARuntimeFusionHostTest, UnsupportedAttributesAndEarlierAssignmentDecline) {
  Enable();
  for (int scenario = 0; scenario < 5; ++scenario) {
    MobiusGraph fixture;
    fixture.Branch();
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    // Deliberately contradict a resolved graph's attributes without rerunning schema
    // validation: the semantic checker must independently reject unsupported metadata.
    ONNX_NAMESPACE::AttributeProto attr;
    if (scenario == 0) {
      attr.set_name("beta");
      attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
      attr.set_f(1.0f);
      fixture.swish->GetMutableAttributes()["beta"] = attr;
    }
    if (scenario == 1) {
      attr.set_name("dilations");
      attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
      attr.add_ints(2);
      fixture.conv->GetMutableAttributes()["dilations"] = attr;
    }
    if (scenario == 2) {
      attr.set_name("auto_pad");
      attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
      attr.set_s("SAME_UPPER");
      fixture.conv->GetMutableAttributes()["auto_pad"] = attr;
    }
    if (scenario == 3) fixture.conv->SetExecutionProviderType(kCpuExecutionProvider);
    if (scenario == 4) fixture.conv->SetExecutionProviderType(kCudaExecutionProvider);
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_TRUE(offers.empty()) << scenario;
  }
}

TEST_F(CUDARuntimeFusionHostTest, ImmutableInputsAndSmallConstantsAreRequired) {
  Enable();
  struct FileCleanup {
    std::filesystem::path path{"cuda_runtime_fusion_host_small_constant.data"};
    ~FileCleanup() {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } cleanup;
  for (int scenario = 0; scenario < 5; ++scenario) {
    SCOPED_TRACE(scenario);
    MobiusGraph fixture;
    fixture.Branch();
    if (scenario < 2) {
      // Initializers that are graph inputs can be overridden and are not immutable.
      fixture.graph.SetInputs({fixture.x, fixture.h, scenario == 0 ? fixture.w : fixture.bias});
    } else {
      const auto* value = scenario == 3 ? fixture.axes : fixture.index;
      auto tensor = *fixture.graph.GetConstantInitializer(value->Name(), false);
      fixture.graph.RemoveInitializedTensor(value->Name());
      if (scenario != 3) {
        const auto bits = tensor.raw_data();
        if (scenario == 4) {
          std::ofstream file(cleanup.path, std::ios::binary);
          file.write(bits.data(), static_cast<std::streamsize>(bits.size()));
          ASSERT_TRUE(file.good());
        }
        tensor.clear_raw_data();
        ExternalDataInfo::SetExternalLocationToProto(
            scenario == 4 ? cleanup.path : "unavailable-small-constant.data", 0, bits.size(), tensor);
      } else {
        tensor.set_raw_data(std::string(sizeof(int64_t), '\0'));
      }
      fixture.graph.AddInitializedTensor(tensor);
    }
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    const auto before = fixture.graph.ToGraphProto().SerializeAsString();
    const auto calls = lookup.calls;
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_TRUE(offers.empty()) << scenario;
    EXPECT_EQ(lookup.calls, calls);
    EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
  }
}

TEST_F(CUDARuntimeFusionHostTest, IntermediateConsumerRejectsButReturnedOutputConsumerIsPreserved) {
  Enable();
  for (bool intermediate : {true, false}) {
    MobiusGraph fixture;
    fixture.Branch();
    auto* extra = fixture.builder.MakeOutput();
    fixture.builder.AddNode("Identity", {intermediate ? fixture.c : fixture.y}, {extra});
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_EQ(offers.size(), intermediate ? 0u : 1u);
    if (offers.empty()) continue;
    GraphOptimizerRegistry registry(&options, &cpu, &logger);
    auto& selected = *offers[0]->nodes_to_optimize[0];
    ASSERT_STATUS_OK(selected.optimization_func(fixture.graph, selected, *offers[0], registry));
    EXPECT_EQ(fixture.graph.NumberOfNodes(), 2);
    EXPECT_EQ(fixture.graph.GetProducerNode(extra->Name())->InputDefs()[0], fixture.y);
  }
}

TEST_F(CUDARuntimeFusionHostTest, DisjointOffersShareConstantsAndRejectStaleAcceptance) {
  Enable();
  MobiusGraph fixture;
  fixture.Branch();
  fixture.Branch();
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
  ASSERT_EQ(offers.size(), 2u);
  GraphOptimizerRegistry registry(&options, &cpu, &logger);
  auto& first = *offers[0]->nodes_to_optimize[0];
  ASSERT_STATUS_OK(first.optimization_func(fixture.graph, first, *offers[0], registry));
  const auto id = offers[1]->sub_graph->nodes[0];
  fixture.graph.GetNode(id)->SetExecutionProviderType(kCpuExecutionProvider);
  const auto before = fixture.graph.ToGraphProto().SerializeAsString();
  auto& second = *offers[1]->nodes_to_optimize[0];
  EXPECT_FALSE(second.optimization_func(fixture.graph, second, *offers[1], registry).IsOK());
  EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
}

TEST_F(CUDARuntimeFusionHostTest, ContradictoryMetadataAndUnsupportedDtypesDecline) {
  Enable();
  for (int scenario = 0; scenario < 7; ++scenario) {
    MobiusGraph fixture;
    fixture.Branch();
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    auto replace_shape = [](NodeArg* value, std::initializer_list<int64_t> dims) {
      ONNX_NAMESPACE::TensorShapeProto shape;
      for (auto dim : dims) {
        auto* target = shape.add_dim();
        if (dim >= 0) target->set_dim_value(dim);
      }
      value->SetShape(shape);
    };
    if (scenario == 0) replace_shape(fixture.x, {1, 10240, 0});
    if (scenario == 1) replace_shape(fixture.h, {1, 10240, 4});
    if (scenario == 2) replace_shape(fixture.x, {1, 10239, 2});
    if (scenario == 3) {
      replace_shape(fixture.x, {1, 10240, 2});
      replace_shape(fixture.h, {2, 10240, 3});
    }
    if (scenario == 4) {
      replace_shape(fixture.y, {1, 10240, 5});
      replace_shape(fixture.conv->MutableOutputDefs()[0], {1, 10240, 6});
    }
    if (scenario >= 5) {
      auto type = *fixture.x->TypeAsProto();
      type.mutable_tensor_type()->set_elem_type(scenario == 5 ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
                                                              : ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      ASSERT_STATUS_OK(fixture.x->UpdateTypeAndShape(type, false, true, logger));
    }
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_TRUE(offers.empty()) << scenario;
  }
}

TEST_F(CUDARuntimeFusionHostTest, MissingIntermediateShapesAreProvedStructurallyAndExternalWeightsNeedOnlyHeaders) {
  Enable();
  MobiusGraph fixture;
  fixture.Branch();
  for (const auto* value : {fixture.w, fixture.bias}) {
    auto tensor = *fixture.graph.GetConstantInitializer(value->Name(), false);
    fixture.graph.RemoveInitializedTensor(value->Name());
    tensor.clear_raw_data();
    tensor.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
    auto* entry = tensor.add_external_data();
    entry->set_key("location");
    entry->set_value("not-read-by-discovery.data");
    fixture.graph.AddInitializedTensor(tensor);
  }
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  for (auto& node : fixture.graph.Nodes()) {
    for (auto* output : node.MutableOutputDefs()) output->ClearShape();
  }
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
  EXPECT_EQ(offers.size(), 1u);
  EXPECT_EQ(lookup.calls, 1u);
}

TEST_F(CUDARuntimeFusionHostTest, ExplicitSliceStepAndConvBiasAreOutsideV1) {
  Enable();
  for (bool extra_bias : {true, false}) {
    MobiusGraph fixture;
    fixture.Branch();
    if (extra_bias) {
      fixture.conv->MutableInputDefs().push_back(fixture.bias);
      fixture.conv->MutableInputArgsCount().push_back(1);
    } else {
      for (auto& node : fixture.graph.Nodes()) {
        if (node.OpType() == "Slice") {
          node.MutableInputDefs().push_back(fixture.builder.MakeInitializer<int64_t>({1}, {1}));
          node.MutableInputArgsCount().push_back(1);
          break;
        }
      }
    }
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    std::vector<std::unique_ptr<ComputeCapability>> offers;
    ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
    EXPECT_TRUE(offers.empty());
  }
}

TEST_F(CUDARuntimeFusionHostTest, OverlappingSinksDoNotOfferUnsafePartialRegions) {
  Enable();
  MobiusGraph fixture;
  fixture.Branch();
  auto* second_y = fixture.builder.MakeOutput();
  fixture.builder.AddNode("Swish", {fixture.swish->MutableInputDefs()[0]}, {second_y});
  fixture.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  const auto before = fixture.graph.ToGraphProto().SerializeAsString();
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(fixture.graph), offers));
  EXPECT_TRUE(offers.empty());
  EXPECT_EQ(lookup.calls, 0u);
  EXPECT_EQ(before, fixture.graph.ToGraphProto().SerializeAsString());
}

}  // namespace
}  // namespace onnxruntime::test

#endif
