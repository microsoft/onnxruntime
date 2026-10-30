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
  ASSERT_STATUS_OK(Select(viewer, offers));
  EXPECT_TRUE(offers.empty());
  for (const char* value : {"", "true", "01", " 1", "2", "-1"}) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kUnitSwishConfig, value));
    EXPECT_FALSE(Select(viewer, offers, false).IsOK());
    EXPECT_TRUE(offers.empty());
  }
  Enable();
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

TEST_F(UnitSwishHostTest, ChangedInputProducerPreservesEligibleNodeArgSnapshot) {
  Enable();
  UnitSwishGraph f;
  ASSERT_STATUS_OK(f.graph.Resolve());
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  ASSERT_STATUS_OK(Select(GraphViewer(f.graph), offers));
  ASSERT_EQ(offers.size(), 1u);
  auto* upstream = f.builder.MakeInput<float>(std::vector<int64_t>{2, 17}, "upstream");
  f.builder.AddNode("Identity", {upstream}, {f.x});
  f.graph.SetInputs({upstream});
  ASSERT_STATUS_OK(f.graph.Resolve());
  auto& selected = *offers[0]->nodes_to_optimize[0];
  GraphOptimizerRegistry registry(&options, &cpu, &logger);
  ASSERT_STATUS_OK(selected.optimization_func(f.graph, selected, *offers[0], registry));
  ASSERT_EQ(offers[0]->sub_graph->nodes.size(), 1u);
  const auto* target = f.graph.GetNode(offers[0]->sub_graph->nodes[0]);
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(target->OpType(), "ScaledSiLU");
  EXPECT_EQ(target->InputDefs()[0], f.x);
  EXPECT_EQ(target->OutputDefs()[0], f.y);
}

TEST_F(UnitSwishHostTest, UnknownRuleAndNonemptyOutputAreErrors) {
  UnitSwishGraph f;
  ASSERT_STATUS_OK(f.graph.Resolve());
  GraphOptimizerRegistry registry(&options, &cpu, &logger);
  std::vector<std::unique_ptr<ComputeCapability>> offers;
  EXPECT_FALSE(cuda_runtime_fusion::Select(GraphViewer(f.graph), lookup, registry, "unknown", offers).IsOK());
  EXPECT_EQ(lookup.calls, 0u);
  offers.push_back(std::make_unique<ComputeCapability>(std::make_unique<IndexedSubGraph>()));
  EXPECT_FALSE(Select(GraphViewer(f.graph), offers).IsOK());
  EXPECT_EQ(offers.size(), 1u);
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

}  // namespace
}  // namespace onnxruntime::test

#endif
