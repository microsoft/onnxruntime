// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "gtest/gtest.h"

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test/test_environment.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/flatbuffers/ort_format_version.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/selectors_actions/selector_action_transformer.h"
#endif  // !defined(ORT_MINIMAL_BUILD)

namespace onnxruntime::test {

#if !defined(ORT_MINIMAL_BUILD)

namespace {
namespace sat {
class TestTransformer : public SelectorActionTransformer {
 public:
  static constexpr const char* kTransformerName = "test_transformer";
  static constexpr const char* kSelectorActionId = "remove_identity";

  TestTransformer(const SatApplyContextVariant& apply_context)
      : SelectorActionTransformer{kTransformerName, CreateSelectorActionRegistry(), apply_context} {
  }

 private:
  struct SurroundingIdentitySelector : NodeSelector {
    std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer, const Node& node) const override {
      // all inputs are identity
      const auto inputs = graph_utils::FindParentsByType(node, "Identity");
      if (inputs.size() != node.GetInputEdgesCount()) return std::nullopt;

      // does not produce graph output
      if (graph_viewer.NodeProducesGraphOutput(node)) return std::nullopt;

      // all outputs are identity
      const auto outputs = graph_utils::FindChildrenByType(node, "Identity");
      if (outputs.size() != node.GetOutputEdgesCount()) return std::nullopt;

      NodesToOptimizeIndicesBuilder builder;

      builder.target_node = node.Index();

      auto get_node_idx = [&](const Node* n) { return n ? n->Index() : NodesToOptimizeIndices::kEmptyNodeIndex; };
      std::transform(inputs.begin(), inputs.end(), std::back_inserter(builder.input_nodes), get_node_idx);
      std::transform(outputs.begin(), outputs.end(), std::back_inserter(builder.output_nodes), get_node_idx);

      return builder.Build();
    }
  };

  static std::vector<NodeAndMoveInfo> GetBinaryMoves() {
    using NTO = NodesToOptimize;
    NTO::NodeLocation i0{NTO::NodeType::kInput, 0};
    NTO::NodeLocation i1{NTO::NodeType::kInput, 1};
    NTO::NodeLocation o0{NTO::NodeType::kOutput, 0};

    return {
        MoveAll(i0, ArgType::kInput),   // append all inputs from i0
        MoveAll(i1, ArgType::kInput),   // append all inputs from i1
        MoveAll(o0, ArgType::kOutput),  // use outputs from o0
    };
  }

  static SelectorActionRegistry CreateSelectorActionRegistry() {
    SelectorActionRegistry result{};
    auto selector = std::make_unique<SurroundingIdentitySelector>();
    auto action = std::make_unique<ReplaceWithNew>(kOnnxDomain, "Add", GetBinaryMoves());
    result.RegisterSelectorAndAction(kSelectorActionId, {{"Add", {}}}, std::move(selector), std::move(action));
    return result;
  }
};
}  // namespace sat

std::unique_ptr<KernelRegistryManager> CreateKernelRegistryManager() {
  auto krm = std::make_unique<KernelRegistryManager>();
  krm->RegisterKernelRegistry(TestCPUExecutionProvider()->GetKernelRegistry());
  return krm;
}
}  // namespace

TEST(GraphRuntimeOptimizationTest, SaveRuntimeOptimizationToOrtFormat) {
  const auto logger = DefaultLoggingManager().CreateLogger("graph_runtime_optimization_test");
  const auto model_path = ORT_TSTR("testdata/transform/runtime_optimization/add_with_surrounding_identities.onnx");

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_path, model, nullptr, *logger));

  Graph& graph = model->MainGraph();
  const auto loaded_ops = CountOpsInGraph(graph);

  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }

  // run SAT to save runtime optimization
  {
    auto kernel_registry_manager = CreateKernelRegistryManager();
    auto save_context = SatRuntimeOptimizationSaveContext{std::cref(*kernel_registry_manager)};
    auto test_transformer = std::make_unique<sat::TestTransformer>(save_context);
    auto transformer_manager = GraphTransformerManager{/* steps */ 5};
    ASSERT_STATUS_OK(transformer_manager.Register(std::move(test_transformer), TransformerLevel::Level1));
    ASSERT_STATUS_OK(transformer_manager.ApplyTransformers(graph, TransformerLevel::Level1, *logger));
  }

  // check that graph nodes are not updated
  {
    const auto initialized_ops = CountOpsInGraph(graph);
    EXPECT_EQ(initialized_ops, loaded_ops);
  }

  flatbuffers::FlatBufferBuilder builder;

  // write graph to ORT format buffer
  {
    flatbuffers::Offset<fbs::Model> fbs_model_offset;
    ASSERT_STATUS_OK(model->SaveToOrtFormat(builder, fbs_model_offset));

    flatbuffers::Offset<fbs::InferenceSession> fbs_session_offset =
        fbs::CreateInferenceSessionDirect(builder,
                                          kOrtModelVersion,
                                          fbs_model_offset,
                                          0);

    builder.Finish(fbs_session_offset);
  }

  // check that ORT format buffer has expected graph runtime optimization data
  {
    const auto* fbs_buffer = builder.GetBufferPointer();
    const auto* fbs_session = fbs::GetInferenceSession(fbs_buffer);
    ASSERT_NE(fbs_session, nullptr);
    ASSERT_NE(fbs_session->model(), nullptr);
    ASSERT_NE(fbs_session->model()->graph(), nullptr);

    const auto* fbs_runtime_optimizations = fbs_session->model()->graph()->runtime_optimizations();
    ASSERT_NE(fbs_runtime_optimizations, nullptr);
    ASSERT_NE(fbs_runtime_optimizations->records(), nullptr);
    ASSERT_EQ(fbs_runtime_optimizations->records()->size(), 1u);

    const auto* fbs_runtime_optimization_entry = (*fbs_runtime_optimizations->records())[0];
    ASSERT_NE(fbs_runtime_optimization_entry, nullptr);

    auto check_string = [](const flatbuffers::String* fbs_str, const char* cstr) {
      ASSERT_STREQ((fbs_str ? fbs_str->c_str() : nullptr), cstr);
    };

    check_string(fbs_runtime_optimization_entry->optimizer_name(), sat::TestTransformer::kTransformerName);

    const auto* fbs_runtime_optimization_records = fbs_runtime_optimization_entry->runtime_optimization_records();
    ASSERT_NE(fbs_runtime_optimization_records, nullptr);
    ASSERT_EQ(fbs_runtime_optimization_records->size(), 1u);

    const auto* fbs_runtime_optimization_record = (*fbs_runtime_optimization_records)[0];
    ASSERT_NE(fbs_runtime_optimization_record, nullptr);

    check_string(fbs_runtime_optimization_record->action_id(), sat::TestTransformer::kSelectorActionId);
  }
}

#endif  // !defined(ORT_MINIMAL_BUILD)

namespace {
using GraphOpCountsCheckerFn = std::function<void(const OpCountMap& loaded_ops, const OpCountMap& initialized_ops)>;

void LoadAndInitializeSession(const SessionOptions& so, const PathString& input_model_path,
                              const GraphOpCountsCheckerFn& graph_op_count_checker_fn) {
  InferenceSessionWrapper session{so, GetEnvironment()};

  ASSERT_STATUS_OK(session.Load(input_model_path));

  const auto loaded_ops = CountOpsInGraph(session.GetGraph());

  ASSERT_STATUS_OK(session.Initialize());

  const auto initialized_ops = CountOpsInGraph(session.GetGraph());

  graph_op_count_checker_fn(loaded_ops, initialized_ops);
}

void SaveAndLoadRuntimeOptimizationsForQdqConvsModel(bool do_save) {
  constexpr auto model_path = ORT_TSTR("testdata/transform/runtime_optimization/qdq_convs.onnx");

  // the .ort and .generated.ort files should be the same
  // the .ort file is checked in and the .generated.ort file is generated by the test
  const auto saved_runtime_optimizations_model_path =
      do_save
          ? ORT_TSTR("testdata/transform/runtime_optimization/qdq_convs.runtime_optimizations.generated.ort")
          : ORT_TSTR("testdata/transform/runtime_optimization/qdq_convs.runtime_optimizations.ort");

  // save runtime optimizations
  if (do_save) {
    SessionOptions so{};
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsConfigSaveModelFormat, "ORT"));
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsConfigSaveRuntimeOptimizations, "1"));
    so.graph_optimization_level = TransformerLevel::Level2;
    so.optimized_model_filepath = saved_runtime_optimizations_model_path;

    ASSERT_NO_FATAL_FAILURE(LoadAndInitializeSession(
        so, model_path,
        [](const OpCountMap& loaded_ops, const OpCountMap& initialized_ops) {
          EXPECT_EQ(initialized_ops, loaded_ops);
        }));
  }

  // load and replay runtime optimizations
  {
    SessionOptions so{};
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsConfigLoadModelFormat, "ORT"));
    so.graph_optimization_level = TransformerLevel::Level2;

    ASSERT_NO_FATAL_FAILURE(LoadAndInitializeSession(
        so, saved_runtime_optimizations_model_path,
        [](const OpCountMap& loaded_ops, const OpCountMap& initialized_ops) {
          constexpr int n = 3;  // expected number of QDQ Convs to fuse

          EXPECT_EQ(loaded_ops,
                    (OpCountMap{{{"DequantizeLinear", n * 3},
                                 {"QuantizeLinear", n},
                                 {"Conv", n}}}));

          EXPECT_EQ(initialized_ops,
                    (OpCountMap{{{"QLinearConv", n}}}));
        }));
  }
}
}  // namespace

#if !defined(ORT_MINIMAL_BUILD)
TEST(GraphRuntimeOptimizationTest, SaveAndLoadRuntimeOptimizationWithSession) {
  SaveAndLoadRuntimeOptimizationsForQdqConvsModel(/* do_save */ true);
}
#endif  // !defined(ORT_MINIMAL_BUILD)

TEST(GraphRuntimeOptimizationTest, LoadRuntimeOptimizationWithSession) {
  SaveAndLoadRuntimeOptimizationsForQdqConvsModel(/* do_save */ false);
}

}  // namespace onnxruntime::test
