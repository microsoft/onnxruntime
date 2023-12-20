// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"

#include "gtest/gtest.h"
#include "core/optimizer/utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/util/include/asserts.h"
#include "orttraining/core/optimizer/shape_optimizer.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#ifndef DISABLE_CONTRIB_OPS

TEST(ShapeSharingTests, Shape15CannotFold) {
  /*
          [attention_mask1_dim0,512,1536]
                           |
                        Identity
                           |
           [attention_mask1_dim0,512,1536]
                           |
                        Shape15
                           |
            [2]: (attention_mask1_dim0,512)
                           |
                        Identity
                           |
                          [2]
  */

  std::string identity_output_name;

  auto pre_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Identity"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);

    identity_output_name = "";
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Identity") == 0)
        identity_output_name = node.MutableOutputDefs()[0]->Name();
    }

    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Identity"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);

    TEST_RETURN_IF_NOT(!identity_output_name.empty());
    auto input_arg = graph.GetNodeArg(identity_output_name);
    // Try to parse int64 type constant initializers.
    InlinedVector<int64_t> shape_values;
    TEST_RETURN_IF_NOT(!optimizer_utils::AppendTensorFromInitializer(graph, *input_arg, shape_values, true));

    return Status::OK();
  };

  std::vector<int> opset_candidates{15};
  for (auto opset : opset_candidates) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      std::vector<std::variant<int64_t, std::string>> identity_input_shape;
      identity_input_shape.reserve(3);
      identity_input_shape.push_back("attention_mask1_dim0");
      identity_input_shape.push_back(512);
      identity_input_shape.push_back(1536);

      auto* identity_input_arg = builder.MakeSymbolicInput<float>(identity_input_shape);
      auto* identity_out_arg = builder.MakeIntermediate();
      builder.AddNode("Identity", {identity_input_arg}, {identity_out_arg});

      auto* shape_out_arg = builder.MakeIntermediate();
      Node& shape_node = builder.AddNode("Shape", {identity_out_arg}, {shape_out_arg});
      shape_node.AddAttribute("start", static_cast<int64_t>(0));
      shape_node.AddAttribute("end", static_cast<int64_t>(2));

      auto* identity_out_arg_1 = builder.MakeOutput();
      builder.AddNode("Identity", {shape_out_arg}, {identity_out_arg_1});
    };
    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
    InlinedHashSet<std::string_view> compatible_eps;
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ShapeOptimizer>(compatible_eps);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST(ShapeSharingTests, Shape15) {
  /*
          [attention_mask1_dim0,512,1536]
                           |
                        Identity
                           |
           [attention_mask1_dim0,512,1536]
                           |
                        Shape15
                           |
                      [2]: (512,1536)
                           |
                        Identity
                           |
                          [2]
  */

  std::string shape_output_name;

  auto pre_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Identity"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);

    shape_output_name = "";
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Shape") == 0)
        shape_output_name = node.MutableOutputDefs()[0]->Name();
    }
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Identity"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 0);

    TEST_RETURN_IF_NOT(!shape_output_name.empty());
    auto input_arg = graph.GetNodeArg(shape_output_name);
    // Try to parse int64 type constant initializers.
    InlinedVector<int64_t> shape_values;
    TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *input_arg, shape_values, true));
    TEST_RETURN_IF_NOT(shape_values.size() == 2U);
    TEST_RETURN_IF_NOT(shape_values[0] == 512);
    TEST_RETURN_IF_NOT(shape_values[1] == 1536);
    return Status::OK();
  };

  std::vector<int> opset_candidates{15};
  for (auto opset : opset_candidates) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      std::vector<std::variant<int64_t, std::string>> identity_input_shape;
      identity_input_shape.reserve(3);
      identity_input_shape.push_back("attention_mask1_dim0");
      identity_input_shape.push_back(512);
      identity_input_shape.push_back(1536);

      auto* identity_input_arg = builder.MakeSymbolicInput<float>(identity_input_shape);
      auto* identity_out_arg = builder.MakeIntermediate();
      builder.AddNode("Identity", {identity_input_arg}, {identity_out_arg});

      auto* shape_out_arg = builder.MakeIntermediate();
      builder.AddNode("Shape", {identity_out_arg}, {shape_out_arg})
          .AddAttribute("start", static_cast<int64_t>(-2));

      auto* identity_out_arg_1 = builder.MakeOutput();
      builder.AddNode("Identity", {shape_out_arg}, {identity_out_arg_1});
    };

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
    InlinedHashSet<std::string_view> compatible_eps;
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ShapeOptimizer>(compatible_eps);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST(ShapeSharingTests, Shape15TakesGraphInput) {
  /*
   [attention_mask1_dim0,512,1536]
                  |
                Shape15
                  |
              [2]: (512,1536)
                  |
                Identity
                  |
                  [2]
  */

  std::string shape_output_name;
  auto pre_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Identity"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);

    shape_output_name = "";
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Shape") == 0)
        shape_output_name = node.MutableOutputDefs()[0]->Name();
    }
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Identity"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 0);

    TEST_RETURN_IF_NOT(!shape_output_name.empty());
    auto input_arg = graph.GetNodeArg(shape_output_name);
    // Try to parse int64 type constant initializers.
    InlinedVector<int64_t> shape_values;
    TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *input_arg, shape_values, true));
    TEST_RETURN_IF_NOT(shape_values.size() == 2U);
    TEST_RETURN_IF_NOT(shape_values[0] == 512);
    TEST_RETURN_IF_NOT(shape_values[1] == 1536);
    return Status::OK();
  };

  std::vector<int> opset_candidates{15};
  for (auto opset : opset_candidates) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      std::vector<std::variant<int64_t, std::string>> shape_input_shape;
      shape_input_shape.reserve(3);
      shape_input_shape.push_back("attention_mask1_dim0");
      shape_input_shape.push_back(512);
      shape_input_shape.push_back(1536);

      auto* shape_input_arg = builder.MakeSymbolicInput<float>(shape_input_shape);
      auto* shape_out_arg = builder.MakeIntermediate();
      builder.AddNode("Shape", {shape_input_arg}, {shape_out_arg})
          .AddAttribute("start", static_cast<int64_t>(-2));

      auto* identity_out_arg_1 = builder.MakeOutput();
      builder.AddNode("Identity", {shape_out_arg}, {identity_out_arg_1});
    };

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
    InlinedHashSet<std::string_view> compatible_eps;
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ShapeOptimizer>(compatible_eps);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST(ShapeSharingTests, Shape15GeneratesGraphOutput) {
  /*
          [attention_mask1_dim0,512,1536]
                           |
                        Identity
                           |
        [attention_mask1_dim0,512,1536]
                           |
                        Shape15
                           |
                      [2]: (512,1536)
                           |
                          [2]
  */
  std::string shape_output_name;
  auto pre_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Identity"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);

    shape_output_name = "";
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Shape") == 0)
        shape_output_name = node.MutableOutputDefs()[0]->Name();
    }
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Identity"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 0);

    TEST_RETURN_IF_NOT(!shape_output_name.empty());
    auto input_arg = graph.GetNodeArg(shape_output_name);
    // Try to parse int64 type constant initializers.
    InlinedVector<int64_t> shape_values;
    TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *input_arg, shape_values, true));
    TEST_RETURN_IF_NOT(shape_values.size() == 2U);
    TEST_RETURN_IF_NOT(shape_values[0] == 512);
    TEST_RETURN_IF_NOT(shape_values[1] == 1536);
    return Status::OK();
  };

  std::vector<int> opset_candidates{15};
  for (auto opset : opset_candidates) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      std::vector<std::variant<int64_t, std::string>> identity_input_shape;
      identity_input_shape.reserve(3);
      identity_input_shape.push_back("attention_mask1_dim0");
      identity_input_shape.push_back(512);
      identity_input_shape.push_back(1536);

      auto* identity_input_arg = builder.MakeSymbolicInput<float>(identity_input_shape);
      auto* identity_out_arg = builder.MakeIntermediate();
      builder.AddNode("Identity", {identity_input_arg}, {identity_out_arg});

      auto* shape_out_arg = builder.MakeOutput();
      builder.AddNode("Shape", {identity_out_arg}, {shape_out_arg})
          .AddAttribute("start", static_cast<int64_t>(-2));
    };

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
    InlinedHashSet<std::string_view> compatible_eps;
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ShapeOptimizer>(compatible_eps);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST(ShapeSharingTests, Slice) {
  /*
            [attention_mask1_dim0,512,1536]
                           |
                        Shape
                           |
        [4]: (attention_mask1_dim0,512,1536)
                           |
                         Slice
                           |
                    [2]: (512, 1536)
                           |
                        Identity
                           |
                          [2]
  */

  std::string slice_output_name;

  auto pre_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Identity"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Slice"] == 1);

    slice_output_name = "";
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Slice") == 0)
        slice_output_name = node.MutableOutputDefs()[0]->Name();
    }
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Identity"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Slice"] == 0);

    TEST_RETURN_IF_NOT(!slice_output_name.empty());
    auto input_arg = graph.GetNodeArg(slice_output_name);
    // Try to parse int64 type constant initializers.
    InlinedVector<int64_t> shape_values;
    TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *input_arg, shape_values, true));
    TEST_RETURN_IF_NOT(shape_values.size() == 2U);
    TEST_RETURN_IF_NOT(shape_values[0] == 512);
    TEST_RETURN_IF_NOT(shape_values[1] == 1536);
    return Status::OK();
  };

  std::vector<int> opset_candidates{10, 11, 12, 13, 14, 15};
  for (auto opset : opset_candidates) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      std::vector<std::variant<int64_t, std::string>> shape_input_shape;
      shape_input_shape.reserve(3);
      shape_input_shape.push_back("attention_mask1_dim0");
      shape_input_shape.push_back(512);
      shape_input_shape.push_back(1536);

      auto* shape_input_arg = builder.MakeSymbolicInput<float>(shape_input_shape);
      auto* shape_out_arg = builder.MakeIntermediate();
      builder.AddNode("Shape", {shape_input_arg}, {shape_out_arg});

      // Slice after opset 1 have such schema.
      auto* slice_out_arg = builder.MakeIntermediate();
      auto* starts_input_arg = builder.MakeInitializer<int64_t>({1}, {-2});
      auto* ends_input_arg = builder.MakeInitializer<int64_t>({1}, {3});
      auto* axes_input_arg = builder.MakeInitializer<int64_t>({1}, {0});
      builder.AddNode("Slice", {shape_out_arg, starts_input_arg, ends_input_arg, axes_input_arg}, {slice_out_arg});

      auto* identity_out_arg_1 = builder.MakeOutput();
      builder.AddNode("Identity", {slice_out_arg}, {identity_out_arg_1});
    };

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
    InlinedHashSet<std::string_view> compatible_eps;
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ShapeOptimizer>(compatible_eps);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST(ShapeSharingTests, SliceGeneratesGraphOutput) {
  /*
            [attention_mask1_dim0,512,1536]
                           |
                        Shape
                           |
        [4]: (attention_mask1_dim0,512,1536)
                           |
                         Slice
                           |
                    [2]: (512, 1536)
                           |
                          [2]
    This test also test when axes and step input are missing.
  */

  std::string slice_output_name;
  auto pre_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Slice"] == 1);

    slice_output_name = "";
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Slice") == 0)
        slice_output_name = node.MutableOutputDefs()[0]->Name();
    }

    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Slice"] == 0);

    TEST_RETURN_IF_NOT(!slice_output_name.empty());
    auto input_arg = graph.GetNodeArg(slice_output_name);
    // Try to parse int64 type constant initializers.
    InlinedVector<int64_t> shape_values;
    TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *input_arg, shape_values, true));
    TEST_RETURN_IF_NOT(shape_values.size() == 2U);
    TEST_RETURN_IF_NOT(shape_values[0] == 512);
    TEST_RETURN_IF_NOT(shape_values[1] == 1536);
    return Status::OK();
  };

  std::vector<int> opset_candidates{10, 11, 12, 13, 14, 15};
  for (auto opset : opset_candidates) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      std::vector<std::variant<int64_t, std::string>> shape_input_shape;
      shape_input_shape.reserve(3);
      shape_input_shape.push_back("attention_mask1_dim0");
      shape_input_shape.push_back(512);
      shape_input_shape.push_back(1536);

      auto* shape_input_arg = builder.MakeSymbolicInput<float>(shape_input_shape);
      auto* shape_out_arg = builder.MakeIntermediate();
      builder.AddNode("Shape", {shape_input_arg}, {shape_out_arg});

      // Slice after opset 1 have such schema.
      auto* slice_out_arg = builder.MakeOutput();
      auto* starts_input_arg = builder.MakeInitializer<int64_t>({1}, {-2});
      auto* ends_input_arg = builder.MakeInitializer<int64_t>({1}, {3});
      builder.AddNode("Slice", {shape_out_arg, starts_input_arg, ends_input_arg}, {slice_out_arg});
    };

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
    InlinedHashSet<std::string_view> compatible_eps;
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ShapeOptimizer>(compatible_eps);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST(ShapeSharingTests, Gather) {
  /*
                  [attention_mask1_dim0,512,24,64]
                                  |
                                Shape
                                  |
                                [4]
                              /      \
                         Gather     Gather
                           |           |
     []: (attention_mask1_dim0,)     [1]: (24,)
                           |           |
       [] means a shape for scalar.
  */

  std::vector<std::string> gather_output_names;
  gather_output_names.reserve(2);
  auto pre_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Gather"] == 2);

    gather_output_names.clear();
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Gather") == 0) {
        gather_output_names.push_back(node.MutableOutputDefs()[0]->Name());
      }
    }

    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Gather"] == 1);

    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "Gather") {
        for (auto& gather_output_name : gather_output_names) {
          if (gather_output_name.compare(node.MutableOutputDefs()[0]->Name()) != 0) {
            // Try to parse int64 type constant initializers.
            InlinedVector<int64_t> shape_values;
            auto input_arg = graph.GetNodeArg(gather_output_name);
            TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *input_arg, shape_values, true));
            TEST_RETURN_IF_NOT(shape_values.size() == 1U);
            TEST_RETURN_IF_NOT(shape_values[0] == 24);
          }
        }
      }
    }

    return Status::OK();
  };

  std::vector<int> opset_candidates{10, 11, 12, 13, 14, 15};
  for (auto opset : opset_candidates) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      std::vector<std::variant<int64_t, std::string>> shape_input_shape;
      shape_input_shape.reserve(4);
      shape_input_shape.push_back("attention_mask1_dim0");
      shape_input_shape.push_back(512);
      shape_input_shape.push_back(24);
      shape_input_shape.push_back(64);

      auto* shape_input_arg = builder.MakeSymbolicInput<float>(shape_input_shape);
      auto* shape_out_arg = builder.MakeIntermediate();
      // Shape before opset 15 have such schema, the test schema did not cover opset 15.
      builder.AddNode("Shape", {shape_input_arg}, {shape_out_arg});

      auto* indices_input_arg = builder.MakeScalarInitializer<int64_t>(0);
      auto* gather_out_arg = builder.MakeOutput();
      builder.AddNode("Gather", {shape_out_arg, indices_input_arg}, {gather_out_arg})
          .AddAttribute("axis", static_cast<int64_t>(0));

      auto* indices_input_arg_1 = builder.MakeInitializer<int64_t>({1}, {2});
      auto* gather_out_arg_1 = builder.MakeOutput();
      builder.AddNode("Gather", {shape_out_arg, indices_input_arg_1}, {gather_out_arg_1})
          .AddAttribute("axis", static_cast<int64_t>(0));
    };

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
    InlinedHashSet<std::string_view> compatible_eps;
    std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ShapeOptimizer>(compatible_eps);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST(ShapeSharingTests, ConcreteDimUsedBySlice) {
  /*
                        [attention_mask1_dim0,24,512,512]
                                  |
                                Dropout
                             /            \
 [attention_mask1_dim0,24,512,512]    [attention_mask1_dim0,24,512,512]
                           |               |
                         Shape             |
                           |               |
                          [4]              |
                      /        \           |
                  Slice        Slice       |
                    |            |         |
                 [1]: (512,)  [1]: (512,)  |
                    |             |        |
                Squeeze        Squeeze     |
                    |             |        |
        [1]: -1  Unsqueeze      Unsqueeze  /
            \         \         /         /
                ConcatTraining           /
                          |             /
                          |            /
              [3]: (-1, 512, 512)     /
                            \        /
                              Reshape
                                |
  */

  std::vector<std::string> slice_output_names;
  auto pre_graph_checker = [&slice_output_names](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Dropout"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Slice"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["Squeeze"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["Unsqueeze"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["com.microsoft.ConcatTraining"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 1);

    slice_output_names.clear();
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Slice") == 0) {
        slice_output_names.push_back(node.OutputDefs()[0]->Name());
      }
    }

    return Status::OK();
  };

  auto post_graph_checker = [&slice_output_names](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Dropout"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Slice"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Squeeze"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["Unsqueeze"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["com.microsoft.ConcatTraining"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 1);

    for (auto slice_output_name : slice_output_names) {
      auto input_arg = graph.GetNodeArg(slice_output_name);
      TEST_RETURN_IF_NOT(input_arg != nullptr);
      InlinedVector<int64_t> shape_values;
      // Try to parse int64 type constant initializers.
      TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *input_arg, shape_values, true));
      TEST_RETURN_IF_NOT(shape_values.size() == 1U);
      TEST_RETURN_IF_NOT(shape_values[0] == 512);
    }

    return Status::OK();
  };

  std::vector<int> opset_candidates{10, 11, 12, 13, 14, 15};
  for (auto opset : opset_candidates) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      std::vector<std::variant<int64_t, std::string>> dropout_input_shape;
      dropout_input_shape.reserve(4);
      dropout_input_shape.push_back("attention_mask1_dim0");
      dropout_input_shape.push_back(24);
      dropout_input_shape.push_back(512);
      dropout_input_shape.push_back(512);

      auto* dropout_input_arg = builder.MakeSymbolicInput<float>(dropout_input_shape);
      auto* dropout_out_arg = builder.MakeIntermediate();
      auto* mask_out_arg = builder.MakeIntermediate();
      constexpr float ratio = 0.10000000149011612f;
      if (opset < 12) {
        builder.AddNode("Dropout", {dropout_input_arg}, {dropout_out_arg, mask_out_arg})
            .AddAttribute("ratio", ratio);
      } else {
        auto* ratio_input_arg = builder.MakeScalarInitializer<float>(ratio);
        auto* mode_input_arg = builder.MakeInitializerBool({}, std::vector<bool>{true});
        builder.AddNode("Dropout", {dropout_input_arg, ratio_input_arg, mode_input_arg},
                        {dropout_out_arg, mask_out_arg});
      }

      auto* shape_out_arg = builder.MakeIntermediate();
      // Shape before opset 15 have such schema, the test schema did not cover opset 15.
      builder.AddNode("Shape", {dropout_out_arg}, {shape_out_arg});

      // Slice after opset 1 have such schema.
      auto* slice_out_arg = builder.MakeIntermediate();
      auto* starts_input_arg = builder.MakeInitializer<int64_t>({1}, {-2});
      auto* ends_input_arg = builder.MakeInitializer<int64_t>({1}, {-1});
      auto* axes_input_arg = builder.MakeInitializer<int64_t>({1}, {0});
      builder.AddNode("Slice", {shape_out_arg, starts_input_arg, ends_input_arg, axes_input_arg}, {slice_out_arg});

      auto* starts_input_arg_1 = builder.MakeInitializer<int64_t>({1}, {-1});
      auto* ends_input_arg_1 = builder.MakeInitializer<int64_t>({1}, {9223372036854775807});
      auto* axes_input_arg_1 = builder.MakeInitializer<int64_t>({1}, {0});
      auto* slice_out_arg_1 = builder.MakeIntermediate();
      builder.AddNode("Slice", {shape_out_arg, starts_input_arg_1, ends_input_arg_1, axes_input_arg_1},
                      {slice_out_arg_1});

      auto* squeeze_out_arg = builder.MakeIntermediate();
      auto* squeeze_out_arg_1 = builder.MakeIntermediate();
      const std::vector<int64_t> squeeze_axes{0};
      if (opset < 13) {
        builder.AddNode("Squeeze", {slice_out_arg}, {squeeze_out_arg}).AddAttribute("axes", squeeze_axes);
        builder.AddNode("Squeeze", {slice_out_arg_1}, {squeeze_out_arg_1}).AddAttribute("axes", squeeze_axes);
      } else {
        auto* squeeze_axes_input_arg = builder.MakeInitializer<int64_t>({1}, squeeze_axes);
        builder.AddNode("Squeeze", {slice_out_arg, squeeze_axes_input_arg}, {squeeze_out_arg});
        auto* squeeze_axes_input_arg_1 = builder.MakeInitializer<int64_t>({1}, squeeze_axes);
        builder.AddNode("Squeeze", {slice_out_arg_1, squeeze_axes_input_arg_1}, {squeeze_out_arg_1});
      }

      auto* unsqueeze_out_arg = builder.MakeIntermediate();
      auto* unsqueeze_out_arg_1 = builder.MakeIntermediate();
      const std::vector<int64_t> unsqueeze_axes{0};
      if (opset < 13) {
        builder.AddNode("Unsqueeze", {squeeze_out_arg}, {unsqueeze_out_arg}).AddAttribute("axes", unsqueeze_axes);
        builder.AddNode("Unsqueeze", {squeeze_out_arg_1}, {unsqueeze_out_arg_1}).AddAttribute("axes", unsqueeze_axes);
      } else {
        auto* unsqueeze_axes_input_arg = builder.MakeInitializer<int64_t>({1}, unsqueeze_axes);
        builder.AddNode("Unsqueeze", {squeeze_out_arg, unsqueeze_axes_input_arg}, {unsqueeze_out_arg});
        auto* unsqueeze_axes_input_arg_1 = builder.MakeInitializer<int64_t>({1}, unsqueeze_axes);
        builder.AddNode("Unsqueeze", {squeeze_out_arg_1, unsqueeze_axes_input_arg_1}, {unsqueeze_out_arg_1});
      }

      auto* concat_training_out_arg = builder.MakeIntermediate();
      auto* concat_input_arg = builder.MakeInitializer<int64_t>({1}, {-1});
      builder.AddNode("ConcatTraining", {concat_input_arg, unsqueeze_out_arg, unsqueeze_out_arg_1},
                      {concat_training_out_arg}, kMSDomain)
          .AddAttribute("axis", static_cast<int64_t>(0));

      auto* reshape_out_arg = builder.MakeOutput();
      builder.AddNode("Reshape", {dropout_out_arg, concat_training_out_arg}, {reshape_out_arg});
    };

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
    InlinedHashSet<std::string_view> compatible_eps;
    std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ShapeOptimizer>(compatible_eps);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST(ShapeSharingTests, ConcreteDimUsedByGatherSlice) {
  /*
      [attention_mask1_dim0,512,1536]     [4]: (0, 0, 24, -1)
                            \            /
                               Reshape
                                /
        [attention_mask1_dim0,512,24,64]
                           |          \
                         Shape      Transpose
                           |           |
                          [4]    [attention_mask1_dim0,24,512,64]
                      /        \        \
                  Gather        Slice     \
                    |            |        \
                 []: (512,)  [1]: (64,)   |
                    |             |        |
                    |          Squeeze     |
                    |             |        |
        [1]: -1  Unsqueeze      Unsqueeze  /
            \         \         /         /
                ConcatTraining           /
                          |             /
                          |            /
              [3]: (-1, 512, 64)      /
                            \        /
                              Reshape
                                |
      [] means a shape for scalar.
  */

  std::string gather_output_name, slice_output_name;
  auto pre_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Transpose"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Gather"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Slice"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Squeeze"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Unsqueeze"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["com.microsoft.ConcatTraining"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 2);

    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Gather") == 0) {
        gather_output_name = node.OutputDefs()[0]->Name();
      } else if (node.OpType().compare("Slice") == 0) {
        slice_output_name = node.OutputDefs()[0]->Name();
      }
    }

    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Transpose"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Gather"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Slice"] == 0);
    TEST_RETURN_IF_NOT(op_to_count["Squeeze"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Unsqueeze"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["com.microsoft.ConcatTraining"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 2);

    auto gather_output_arg = graph.GetNodeArg(gather_output_name);
    TEST_RETURN_IF_NOT(gather_output_arg != nullptr);
    // Try to parse int64 type constant initializers.
    InlinedVector<int64_t> gather_out_values;
    TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *gather_output_arg, gather_out_values, true));
    TEST_RETURN_IF_NOT(gather_out_values.size() == 1U);
    TEST_RETURN_IF_NOT(gather_out_values[0] == 512);

    auto slice_out_arg = graph.GetNodeArg(slice_output_name);
    TEST_RETURN_IF_NOT(slice_out_arg != nullptr);
    // Try to parse int64 type constant initializers.
    InlinedVector<int64_t> slice_out_values;
    TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *slice_out_arg, slice_out_values, true));
    TEST_RETURN_IF_NOT(slice_out_values.size() == 1U);
    TEST_RETURN_IF_NOT(slice_out_values[0] == 64);

    return Status::OK();
  };

  std::vector<int> opset_candidates{10, 11, 12, 13, 14, 15};
  for (auto opset : opset_candidates) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      std::vector<std::variant<int64_t, std::string>> reshape_input_shape;
      reshape_input_shape.reserve(3);
      reshape_input_shape.push_back("attention_mask1_dim0");
      reshape_input_shape.push_back(512);
      reshape_input_shape.push_back(1536);

      auto* reshape_input_arg = builder.MakeSymbolicInput<float>(reshape_input_shape);
      auto* target_shape_input_arg = builder.MakeInitializer<int64_t>({4}, {0, 0, 24, -1});
      auto* reshape_out_arg = builder.MakeIntermediate();
      builder.AddNode("Reshape", {reshape_input_arg, target_shape_input_arg}, {reshape_out_arg});

      auto* shape_out_arg = builder.MakeIntermediate();
      // Shape before opset 15 have such schema, the test schema did not cover opset 15.
      builder.AddNode("Shape", {reshape_out_arg}, {shape_out_arg});
      auto* transpose_out_arg = builder.MakeIntermediate();
      builder.AddNode("Transpose", {reshape_out_arg}, {transpose_out_arg})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});

      auto* indices_input_arg = builder.MakeScalarInitializer<int64_t>(1);
      auto* gather_out_arg = builder.MakeIntermediate();
      builder.AddNode("Gather", {shape_out_arg, indices_input_arg}, {gather_out_arg})
          .AddAttribute("axis", static_cast<int64_t>(0));

      auto* starts_input_arg_1 = builder.MakeInitializer<int64_t>({1}, {-1});
      auto* ends_input_arg_1 = builder.MakeInitializer<int64_t>({1}, {9223372036854775807});
      auto* axes_input_arg_1 = builder.MakeInitializer<int64_t>({1}, {0});
      auto* slice_out_arg_1 = builder.MakeIntermediate();
      builder.AddNode("Slice", {shape_out_arg, starts_input_arg_1, ends_input_arg_1, axes_input_arg_1},
                      {slice_out_arg_1});

      auto* squeeze_out_arg_1 = builder.MakeIntermediate();
      const std::vector<int64_t> squeeze_axes{0};
      if (opset < 13) {
        builder.AddNode("Squeeze", {slice_out_arg_1}, {squeeze_out_arg_1}).AddAttribute("axes", squeeze_axes);
      } else {
        auto* squeeze_axes_input_arg_1 = builder.MakeInitializer<int64_t>({1}, squeeze_axes);
        builder.AddNode("Squeeze", {slice_out_arg_1, squeeze_axes_input_arg_1}, {squeeze_out_arg_1});
      }

      auto* unsqueeze_out_arg = builder.MakeIntermediate();
      auto* unsqueeze_out_arg_1 = builder.MakeIntermediate();
      const std::vector<int64_t> unsqueeze_axes{0};
      if (opset < 13) {
        builder.AddNode("Unsqueeze", {gather_out_arg}, {unsqueeze_out_arg}).AddAttribute("axes", unsqueeze_axes);
        builder.AddNode("Unsqueeze", {squeeze_out_arg_1}, {unsqueeze_out_arg_1}).AddAttribute("axes", unsqueeze_axes);
      } else {
        auto* unsqueeze_axes_input_arg = builder.MakeInitializer<int64_t>({1}, unsqueeze_axes);
        builder.AddNode("Unsqueeze", {gather_out_arg, unsqueeze_axes_input_arg}, {unsqueeze_out_arg});
        auto* unsqueeze_axes_input_arg_1 = builder.MakeInitializer<int64_t>({1}, unsqueeze_axes);
        builder.AddNode("Unsqueeze", {squeeze_out_arg_1, unsqueeze_axes_input_arg_1}, {unsqueeze_out_arg_1});
      }

      auto* concat_training_out_arg = builder.MakeIntermediate();
      auto* concat_input_arg = builder.MakeInitializer<int64_t>({1}, {-1});
      builder.AddNode("ConcatTraining", {concat_input_arg, unsqueeze_out_arg, unsqueeze_out_arg_1},
                      {concat_training_out_arg}, kMSDomain)
          .AddAttribute("axis", static_cast<int64_t>(0));

      auto* reshape_out_arg_1 = builder.MakeOutput();
      builder.AddNode("Reshape", {transpose_out_arg, concat_training_out_arg}, {reshape_out_arg_1});
    };

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
    InlinedHashSet<std::string_view> compatible_eps;
    std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ShapeOptimizer>(compatible_eps);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST(ShapeSharingTests, SymbolicDimUsedByGather_ConcreteDimUsedByGather) {
  /*
                [attention_mask1_dim0,512,1536]     [4]: (0, 0, 24, -1)
                                      \            /
                                        Reshape
                                          /
                  [attention_mask1_dim0,512,24,64]
                                  |             \
                                Shape          Transpose
                                  |                 |
                                [4]     [attention_mask1_dim0,24,512,64]
                            /     |                     |
                        Gather   Gather                 |
                          |        |                    |
  []: (attention_mask1_dim0,)  [1]: (24,)               |
                         |          |                   |
                         |          |                   |
                         |          |                  /
                    Unsqueeze       |    [1]: -1      /
                         \          |     /          /
                          ConcatTraining            /
                                    |              /
                                    |             /
      [3]: (attention_mask1_dim0, 24, -1)       /
                                      \        /
                                        Reshape
                                          |
       [] means a shape for scalar.
  */
  auto pre_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Transpose"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Gather"] == 2);
    TEST_RETURN_IF_NOT(op_to_count["Unsqueeze"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["com.microsoft.ConcatTraining"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 2);

    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) -> Status {
    auto op_to_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_to_count["Shape"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Transpose"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Gather"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Unsqueeze"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["com.microsoft.ConcatTraining"] == 1);
    TEST_RETURN_IF_NOT(op_to_count["Reshape"] == 2);

    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "Reshape") {
        NodeArg* shape_input = node.MutableInputDefs()[1];
        auto p_output_node = node.OutputNodesBegin();
        const auto p_output_node_end = node.OutputNodesEnd();
        bool find_transpose = false;
        while (p_output_node != p_output_node_end) {
          const auto& output_node = *p_output_node;
          if (output_node.OpType().compare("Transpose") == 0) {
            find_transpose = true;
            break;
          }
          ++p_output_node;
        }

        if (find_transpose) {
          // Ignore the first Reshape node.
          continue;
        }

        // Try to parse int64 type constant initializers.
        InlinedVector<int64_t> shape_values;
        TEST_RETURN_IF_NOT(!optimizer_utils::AppendTensorFromInitializer(graph, *shape_input, shape_values, true));
        TEST_RETURN_IF_NOT(graph.GetProducerNode(
                                    node.MutableInputDefs()[1]->Name())
                               ->OpType()
                               .compare("ConcatTraining") == 0);
      } else if (node.OpType() == "ConcatTraining") {
        NodeArg* shape_input = node.MutableInputDefs()[1];

        // Try to parse int64 type constant initializers.
        InlinedVector<int64_t> shape_values;
        TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *shape_input, shape_values, true));
        TEST_RETURN_IF_NOT(shape_values.size() == 1U);
        TEST_RETURN_IF_NOT(shape_values[0] == 24);
      }
    }

    return Status::OK();
  };

  std::vector<int> opset_candidates{10, 11, 12, 13, 14, 15};
  for (auto opset : opset_candidates) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      std::vector<std::variant<int64_t, std::string>> reshape_input_shape;
      reshape_input_shape.reserve(3);
      reshape_input_shape.push_back("attention_mask1_dim0");
      reshape_input_shape.push_back(512);
      reshape_input_shape.push_back(1536);

      auto* reshape_input_arg = builder.MakeSymbolicInput<float>(reshape_input_shape);
      auto* target_shape_input_arg = builder.MakeInitializer<int64_t>({4}, {0, 0, 24, -1});
      auto* reshape_out_arg = builder.MakeIntermediate();
      builder.AddNode("Reshape", {reshape_input_arg, target_shape_input_arg}, {reshape_out_arg});

      auto* shape_out_arg = builder.MakeIntermediate();
      // Shape before opset 15 have such schema, the test schema did not cover opset 15.
      builder.AddNode("Shape", {reshape_out_arg}, {shape_out_arg});
      auto* transpose_out_arg = builder.MakeIntermediate();
      builder.AddNode("Transpose", {reshape_out_arg}, {transpose_out_arg})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});

      auto* indices_input_arg = builder.MakeScalarInitializer<int64_t>(0);
      auto* gather_out_arg = builder.MakeIntermediate();
      builder.AddNode("Gather", {shape_out_arg, indices_input_arg}, {gather_out_arg})
          .AddAttribute("axis", static_cast<int64_t>(0));

      auto* indices_input_arg_1 = builder.MakeInitializer<int64_t>({1}, {2});
      auto* gather_out_arg_1 = builder.MakeIntermediate();
      builder.AddNode("Gather", {shape_out_arg, indices_input_arg_1}, {gather_out_arg_1})
          .AddAttribute("axis", static_cast<int64_t>(0));

      auto* unsqueeze_out_arg = builder.MakeIntermediate();
      const std::vector<int64_t> unsqueeze_axes{0};
      if (opset < 13) {
        builder.AddNode("Unsqueeze", {gather_out_arg}, {unsqueeze_out_arg}).AddAttribute("axes", unsqueeze_axes);
      } else {
        auto* unsqueeze_axes_input_arg = builder.MakeInitializer<int64_t>({1}, unsqueeze_axes);
        builder.AddNode("Unsqueeze", {gather_out_arg, unsqueeze_axes_input_arg}, {unsqueeze_out_arg});
      }

      auto* concat_training_out_arg = builder.MakeIntermediate();
      auto* concat_input_arg = builder.MakeInitializer<int64_t>({1}, {-1});
      builder.AddNode("ConcatTraining", {unsqueeze_out_arg, gather_out_arg_1, concat_input_arg},
                      {concat_training_out_arg}, kMSDomain)
          .AddAttribute("axis", static_cast<int64_t>(0));

      auto* reshape_out_arg_1 = builder.MakeOutput();
      builder.AddNode("Reshape", {transpose_out_arg, concat_training_out_arg}, {reshape_out_arg_1});
    };

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
    InlinedHashSet<std::string_view> compatible_eps;
    std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ShapeOptimizer>(compatible_eps);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

// end of DISABLE_CONTRIB_OPS
#endif

}  // namespace test
}  // namespace onnxruntime
