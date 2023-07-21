// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <string>

#include "gtest/gtest.h"

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_environment.h"

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
      : SelectorActionTransformer{kTransformerName,
                                  CreateSelectorActionRegistry(),
                                  apply_context,
                                  {kCpuExecutionProvider}} {
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
    auto action = std::make_unique<ReplaceWithNewFixed>(kOnnxDomain, "Add", GetBinaryMoves());
    result.RegisterSelectorAndAction(kSelectorActionId, {{"Add", {}}}, std::move(selector), std::move(action));
    return result;
  }
};
}  // namespace sat
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
    auto save_context = SatRuntimeOptimizationSaveContext{};
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
                                          std::to_string(kOrtModelVersion).c_str(),
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

#if !defined(DISABLE_CONTRIB_OPS)

namespace {
using GraphOpCountsCheckerFn = std::function<void(const OpCountMap& loaded_ops, const OpCountMap& initialized_ops)>;
using GraphCheckerFn = std::function<void(const Graph& graph)>;

void LoadAndInitializeSession(const SessionOptions& so, const PathString& input_model_path,
                              const GraphOpCountsCheckerFn& graph_op_count_checker_fn,
                              const GraphCheckerFn& graph_checker_fn = {}) {
  InferenceSessionWrapper session{so, GetEnvironment()};

  ASSERT_STATUS_OK(session.Load(input_model_path));

  const auto loaded_ops = CountOpsInGraph(session.GetGraph());

  ASSERT_STATUS_OK(session.Initialize());

  const auto initialized_ops = CountOpsInGraph(session.GetGraph());

  if (graph_op_count_checker_fn) {
    graph_op_count_checker_fn(loaded_ops, initialized_ops);
  }

  if (graph_checker_fn) {
    graph_checker_fn(session.GetGraph());
  }
}

void SaveAndLoadRuntimeOptimizationsForModel(
    const PathString& onnx_model_path,
    const PathString& ort_model_with_runtime_opt_path,
    const GraphOpCountsCheckerFn& graph_op_counts_checker_for_replay) {
  auto run_test = [&](bool do_save) {
    // the two versions of the saved runtime optimizations file should be the same
    // the one with the ".test_output" suffix is generated by the test and the other is checked in
    const PathString saved_runtime_optimizations_model_path =
        do_save ? ort_model_with_runtime_opt_path + ORT_TSTR(".test_output")
                : ort_model_with_runtime_opt_path;

    SCOPED_TRACE(MakeString("ONNX model: '", ToUTF8String(onnx_model_path),
                            "', ORT format model with runtime optimizations: '",
                            ToUTF8String(saved_runtime_optimizations_model_path),
                            "', load only: ", !do_save));

    // save runtime optimizations
    if (do_save) {
      SessionOptions so{};
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsConfigSaveModelFormat, "ORT"));
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsConfigMinimalBuildOptimizations, "save"));
      so.graph_optimization_level = TransformerLevel::Level2;
      so.optimized_model_filepath = saved_runtime_optimizations_model_path;

      ASSERT_NO_FATAL_FAILURE(LoadAndInitializeSession(
          so, onnx_model_path,
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
          graph_op_counts_checker_for_replay));
    }
  };

#if !defined(ORT_MINIMAL_BUILD)
  run_test(/* do_save */ true);
#endif  // !defined(ORT_MINIMAL_BUILD)
  run_test(/* do_save */ false);
}

// if level 3 optimizations are enabled the NHWC transformer should convert the QLinearConv nodes to use channels_last
void CheckNhwcTransformerIsApplied(const PathString& ort_model_path,
                                   const GraphOpCountsCheckerFn& graph_op_counts_checker) {
  SCOPED_TRACE(MakeString("ORT format model: ", ToUTF8String(ort_model_path)));

  // load and replay runtime optimizations
  SessionOptions so{};
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsConfigLoadModelFormat, "ORT"));
  so.graph_optimization_level = TransformerLevel::Level3;

  GraphCheckerFn graph_checker = [](const Graph& graph) {
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "QLinearConv") {
        EXPECT_EQ(node.Domain(), kMSDomain);
        bool has_channels_last_set = false;
        for (const auto& attr : node.GetAttributes()) {
          if (attr.first == "channels_last") {
            EXPECT_EQ(attr.second.i(), 1);
            has_channels_last_set = true;
            break;
          }
        }
        EXPECT_TRUE(has_channels_last_set);
      }
    }
  };

  ASSERT_NO_FATAL_FAILURE(LoadAndInitializeSession(
      so, ort_model_path,
      graph_op_counts_checker,
      graph_checker));
};
}  // namespace

TEST(GraphRuntimeOptimizationTest, QDQConv) {
  SaveAndLoadRuntimeOptimizationsForModel(
      ORT_TSTR("testdata/transform/runtime_optimization/qdq_convs.onnx"),
      ORT_TSTR("testdata/transform/runtime_optimization/qdq_convs.runtime_optimizations.ort"),
      [](const OpCountMap& loaded_ops, const OpCountMap& initialized_ops) {
        constexpr int n = 3;  // expected number of QDQ Convs to fuse

        EXPECT_EQ(loaded_ops,
                  (OpCountMap{{"DequantizeLinear", n * 3},
                              {"QuantizeLinear", n},
                              {"Conv", n}}));

        EXPECT_EQ(initialized_ops,
                  (OpCountMap{{"QLinearConv", n}}));
      });
}

TEST(GraphRuntimeOptimizationTest, ConvActivation) {
  SaveAndLoadRuntimeOptimizationsForModel(
      ORT_TSTR("testdata/transform/fusion/conv_clip11.onnx"),
      ORT_TSTR("testdata/transform/runtime_optimization/conv_clip11.runtime_optimizations.ort"),
      [](const OpCountMap& loaded_ops, const OpCountMap& initialized_ops) {
        constexpr int num_conv_activations = 3;
        constexpr int expected_num_fusions = 2;

        EXPECT_EQ(loaded_ops,
                  (OpCountMap{{"Conv", num_conv_activations},
                              {"Clip", num_conv_activations}}));

        EXPECT_EQ(initialized_ops,
                  (OpCountMap{{"Conv", num_conv_activations - expected_num_fusions},
                              {"Clip", num_conv_activations - expected_num_fusions},
                              {"com.microsoft.FusedConv", expected_num_fusions}}));
      });
}

TEST(GraphRuntimeOptimizationTest, TestNhwcTransformer) {
  CheckNhwcTransformerIsApplied(
      ORT_TSTR("testdata/transform/runtime_optimization/qdq_convs.runtime_optimizations.ort"),
      [](const OpCountMap& loaded_ops, const OpCountMap& initialized_ops) {
        constexpr int n = 3;  // expected number of QDQ Convs to fuse

        EXPECT_EQ(loaded_ops,
                  (OpCountMap{{"DequantizeLinear", n * 3},
                              {"QuantizeLinear", n},
                              {"Conv", n}}));

        // should have internal version of QLinearConv that runs NHWC, and transposes around each of those nodes
        // for the layout conversion.
        EXPECT_EQ(initialized_ops,
                  (OpCountMap{{"Transpose", n * 2},
                              {"com.microsoft.QLinearConv", n}}));
      });
}

TEST(GraphRuntimeOptimizationTest, TestNhwcTransformerDirectlyUpdatesQLinearConv) {
  CheckNhwcTransformerIsApplied(
      // ORT format model that contains QLinearConv nodes
      // to generate:
      // - set environment variable ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL=extended
      // - run:
      //     python -m onnxruntime.tools.convert_onnx_models_to_ort
      //       --optimization_style=Fixed
      //       testdata/transform/runtime_optimization/qdq_convs.onnx
      ORT_TSTR("testdata/transform/runtime_optimization/qdq_convs.extended.ort"),
      [](const OpCountMap& loaded_ops, const OpCountMap& initialized_ops) {
        constexpr int n = 3;  // expected number of QLinearConvs

        EXPECT_EQ(loaded_ops,
                  (OpCountMap{{"QLinearConv", n}}));

        // should have internal version of QLinearConv that runs NHWC, and transposes around each of those nodes
        // for the layout conversion.
        EXPECT_EQ(initialized_ops,
                  (OpCountMap{{"Transpose", n * 2},
                              {"com.microsoft.QLinearConv", n}}));
      });
}

#if !defined(ORT_MINIMAL_BUILD)
TEST(GraphRuntimeOptimizationTest, TestOnlyApplyMinimalBuildOptimizations) {
  // This test assumes that AttentionFusion is not included in the minimal build optimizations.
  // Update it if that changes.

  // When setting the option to only apply minimal build optimizations, verify that AttentionFusion does not run.
  {
    SessionOptions so{};
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsConfigMinimalBuildOptimizations, "apply"));
    so.graph_optimization_level = TransformerLevel::Level2;

    LoadAndInitializeSession(
        so,
        ORT_TSTR("testdata/transform/fusion/attention_int32_mask.onnx"),
        [](const OpCountMap& /*initialized_ops*/, const OpCountMap& loaded_ops) {
          // expect no fused node
          EXPECT_EQ(OpCount(loaded_ops, "com.microsoft.Attention"), 0);
        });
  }

  // Otherwise, it should run.
  {
    SessionOptions so{};
    so.graph_optimization_level = TransformerLevel::Level2;

    LoadAndInitializeSession(
        so,
        ORT_TSTR("testdata/transform/fusion/attention_int32_mask.onnx"),
        [](const OpCountMap& /*initialized_ops*/, const OpCountMap& loaded_ops) {
          // expect fused node
          EXPECT_EQ(OpCount(loaded_ops, "com.microsoft.Attention"), 1);
        });
  }
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace onnxruntime::test
