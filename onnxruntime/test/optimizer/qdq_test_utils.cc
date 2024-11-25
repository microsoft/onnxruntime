// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qdq_test_utils.h"
#include <type_traits>
#include <utility>
#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/common/span_utils.h"

namespace onnxruntime {
namespace test {

GetQDQTestCaseFn BuildQDQReshapeTestCase(const std::vector<int64_t>& input_shape,
                                         const std::vector<int64_t>& reshape_shape) {
  return [input_shape, reshape_shape](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                 std::numeric_limits<uint8_t>::min(),
                                                 std::numeric_limits<uint8_t>::max());
    auto* output_arg = builder.MakeOutput();

    // add DQ
    auto* dq_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<uint8_t>(input_arg, .003f, 1, dq_output);

    // add Reshape
    auto* reshape_output = builder.MakeIntermediate();
    auto* shape = builder.Make1DInitializer<int64_t>(reshape_shape);
    builder.AddNode("Reshape", {dq_output, shape}, {reshape_output});

    // add Q
    builder.AddQuantizeLinearNode<uint8_t>(reshape_output, .003f, 1, output_arg);
  };
}

GetQDQTestCaseFn BuildQDQConcatTestCase(const std::vector<std::vector<int64_t>>& input_shapes,
                                        int64_t axis,
                                        bool has_input_float,
                                        bool has_input_int8,
                                        bool has_output_int8,
                                        bool use_contrib_qdq) {
  return [input_shapes, axis, has_input_float, has_input_int8,
          has_output_int8, use_contrib_qdq](ModelTestBuilder& builder) {
    auto input_count = input_shapes.size();
    std::vector<NodeArg*> input_args;
    std::vector<NodeArg*> q_input_args;
    for (size_t i = 0; i < input_count; i++) {
      input_args.push_back(builder.MakeInput<float>(input_shapes[i], -1.f, 1.f));
      if (i == 0 && has_input_float) {
        q_input_args.push_back(input_args.back());
      } else if (i == 0 && has_input_int8) {
        q_input_args.push_back(AddQDQNodePair<int8_t>(builder, input_args.back(), 0.05f, 1, use_contrib_qdq));
      } else {
        q_input_args.push_back(AddQDQNodePair<uint8_t>(builder, input_args.back(), 0.05f, 128, use_contrib_qdq));
      }
    }
    auto* concat_output = builder.MakeIntermediate();
    Node& concat_node = builder.AddNode("Concat", q_input_args, {concat_output});
    concat_node.AddAttribute("axis", axis);

    auto* q_concat_output = builder.MakeIntermediate();
    if (has_output_int8) {
      builder.AddQuantizeLinearNode<int8_t>(concat_output, 0.05f, 1, q_concat_output, use_contrib_qdq);

      auto* output_arg = builder.MakeOutput();
      builder.AddDequantizeLinearNode<int8_t>(q_concat_output, 0.05f, 1, output_arg, use_contrib_qdq);
    } else {
      builder.AddQuantizeLinearNode<uint8_t>(concat_output, 0.05f, 128, q_concat_output, use_contrib_qdq);

      auto* output_arg = builder.MakeOutput();
      builder.AddDequantizeLinearNode<uint8_t>(q_concat_output, 0.05f, 128, output_arg, use_contrib_qdq);
    }
  };
}

GetQDQTestCaseFn BuildQDQConcatTestCaseUnsupportedInputScaleZp() {
  return [](ModelTestBuilder& builder) {
    const std::vector<std::vector<int64_t>> input_shapes = {
        {1, 6, 36},
        {1, 6, 8},
        {1, 6, 2},
    };
    int64_t axis = 2;

    std::vector<NodeArg*> input_args;
    std::vector<NodeArg*> q_input_args;

    // set unmatched input scales/zp for test purpose
    input_args.push_back(builder.MakeInput<float>(input_shapes[0], -1.f, 1.f));
    q_input_args.push_back(AddQDQNodePair<uint8_t>(builder, input_args.back(), 0.05f, 128));
    input_args.push_back(builder.MakeInput<float>(input_shapes[1], -1.f, 1.f));
    q_input_args.push_back(AddQDQNodePair<uint8_t>(builder, input_args.back(), 0.04f, 127));
    input_args.push_back(builder.MakeInput<float>(input_shapes[2], -1.f, 1.f));
    q_input_args.push_back(AddQDQNodePair<uint8_t>(builder, input_args.back(), 0.03f, 126));

    auto* concat_output = builder.MakeIntermediate();
    Node& concat_node = builder.AddNode("Concat", q_input_args, {concat_output});
    concat_node.AddAttribute("axis", axis);

    auto* q_concat_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(concat_output, 0.05f, 128, q_concat_output);
    auto* output_arg = builder.MakeOutput();
    builder.AddDequantizeLinearNode<uint8_t>(q_concat_output, 0.05f, 128, output_arg);
  };
}

GetQDQTestCaseFn BuildQDQMatMulTestCase(const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape) {
  return [input1_shape, input2_shape](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    using InputLimits = std::numeric_limits<uint8_t>;
    using OutputTypeLimits = std::numeric_limits<uint8_t>;

    // add QDQ input
    auto* q1_output = builder.MakeIntermediate();
    auto* dq1_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input_arg,
                                           .039f,
                                           (InputLimits::max() + InputLimits::min()) / 2 + 1,
                                           q1_output);
    builder.AddDequantizeLinearNode<uint8_t>(q1_output,
                                             .039f,
                                             (InputLimits::max() + InputLimits::min()) / 2 + 1,
                                             dq1_output);

    // add input b initializer (NNAPI only supports case of MatMul A*B - B is an initializer)
    auto* dq_2_output = builder.MakeIntermediate();
    auto* input_b = builder.MakeInitializer<uint8_t>(input2_shape, InputLimits::min(), InputLimits::max());
    builder.AddDequantizeLinearNode<uint8_t>(input_b, .04f, 0, dq_2_output);

    // add MatMul operator
    auto* matmul_op_output = builder.MakeIntermediate();
    builder.AddNode("MatMul", {dq1_output, dq_2_output}, {matmul_op_output});

    // add QDQ output
    auto* q3_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(matmul_op_output,
                                           .039f,
                                           (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                           q3_output);
    builder.AddDequantizeLinearNode<uint8_t>(q3_output,
                                             .039f,
                                             (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                             output_arg);
  };
}

std::vector<std::string> GetNodeOpTypesInTopologicalOrder(const Graph& graph, bool include_domain) {
  std::vector<std::string> op_types{};
  GraphViewer graph_viewer{graph};
  const auto& ordering = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : ordering) {
    const auto* node = graph.GetNode(node_idx);
    std::string full_op_type;

    if (include_domain) {
      const std::string& domain = node->Domain();
      full_op_type = domain.empty() ? node->OpType() : domain + "." + node->OpType();
    } else {
      full_op_type = node->OpType();
    }

    op_types.push_back(std::move(full_op_type));
  }
  return op_types;
}

GetQDQTestCaseFn BuildDoubleQDQTestCaseWithDuplicateLastDQs(
    gsl::span<const int64_t> input_shape,
    gsl::span<const float> input_data,
    gsl::span<const int64_t> zero_points,
    gsl::span<const ONNX_NAMESPACE::TensorProto_DataType> zero_point_types,
    gsl::span<const float> scales,
    size_t graph_output_index,
    bool use_contrib_qdq) {
  const size_t num_nodes = zero_points.size();
  bool valid_inputs = (num_nodes >= 4) &&
                      (zero_point_types.size() == num_nodes) &&
                      (scales.size() == num_nodes) &&
                      (graph_output_index < 4);
  if (!valid_inputs) {
    ORT_THROW("Invalid inputs for call to BuildDoubleQDQTestCaseWithDuplicateLastDQs()");
  }

  return [=](ModelTestBuilder& builder) {
    // TODO(adrianlizarraga): Clean up ModelTestBuilder functions (like MakeInput) to work with gsl::span inputs.
    // For now, we have to copy data into a std::vector if we want this outer function to take in span inputs.
    std::vector<int64_t> input_shape_copy(input_shape.begin(), input_shape.end());
    std::vector<float> input_data_copy(input_data.begin(), input_data.end());
    auto* input_arg = builder.MakeInput<float>(input_shape_copy, input_data_copy);
    InlinedVector<NodeArg*> node_outputs(num_nodes);

    for (size_t i = 0; i < num_nodes; i++) {
      if (i == graph_output_index || i >= 3) {
        node_outputs[i] = builder.MakeOutput();
      } else {
        node_outputs[i] = builder.MakeIntermediate();
      }
    }

    builder.AddQuantizeLinearNode(input_arg, scales[0], zero_points[0], zero_point_types[0], node_outputs[0],
                                  use_contrib_qdq);
    builder.AddDequantizeLinearNode(node_outputs[0], scales[1], zero_points[1], zero_point_types[1], node_outputs[1],
                                    use_contrib_qdq);
    builder.AddQuantizeLinearNode(node_outputs[1], scales[2], zero_points[2], zero_point_types[2], node_outputs[2],
                                  use_contrib_qdq);

    for (size_t i = 3; i < num_nodes; i++) {
      builder.AddDequantizeLinearNode(node_outputs[2], scales[i], zero_points[i], zero_point_types[i],
                                      node_outputs[i], use_contrib_qdq);
    }
  };
}

}  // namespace test
}  // namespace onnxruntime
