// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qdq_test_utils.h"

namespace onnxruntime {
namespace test {

GetQDQTestCaseFn BuildQDQResizeTestCase(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& sizes_data,
    const std::string& mode,
    const std::string& coordinate_transformation_mode) {
  return [input_shape, sizes_data, mode, coordinate_transformation_mode](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<uint8_t>(input_shape,
                                                  std::numeric_limits<uint8_t>::min(),
                                                  std::numeric_limits<uint8_t>::max());
    auto* roi = builder.MakeInitializer<float>({0}, {});
    auto* scales = builder.MakeInitializer<float>({0}, {});
    auto* sizes = builder.Make1DInitializer<int64_t>(sizes_data);
    auto* output_arg = builder.MakeOutput();

    // add DQ
    auto* dq_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<uint8_t>(input1_arg, .003f, 1, dq_output);

    // add Resize
    auto* resize_output = builder.MakeIntermediate();
    Node& resize_node = builder.AddNode("Resize", {dq_output, roi, scales, sizes}, {resize_output});

    resize_node.AddAttribute("mode", mode);
    resize_node.AddAttribute("coordinate_transformation_mode", coordinate_transformation_mode);

    // add Q
    builder.AddQuantizeLinearNode<uint8_t>(resize_output, .003f, 1, output_arg);
  };
}

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
                                        bool has_output_int8) {
  return [input_shapes, axis,
          has_input_float, has_input_int8, has_output_int8](
             ModelTestBuilder& builder) {
    auto input_count = input_shapes.size();
    std::vector<NodeArg*> input_args;
    std::vector<NodeArg*> q_input_args;
    for (size_t i = 0; i < input_count; i++) {
      input_args.push_back(builder.MakeInput<float>(input_shapes[i], -1.f, 1.f));
      if (i == 0 && has_input_float) {
        q_input_args.push_back(input_args.back());
      } else if (i == 0 && has_input_int8) {
        q_input_args.push_back(AddQDQNodePair<int8_t>(builder, input_args.back(), 0.05f, 1));
      } else {
        q_input_args.push_back(AddQDQNodePair<uint8_t>(builder, input_args.back(), 0.05f, 128));
      }
    }
    auto* concat_output = builder.MakeIntermediate();
    Node& concat_node = builder.AddNode("Concat", q_input_args, {concat_output});
    concat_node.AddAttribute("axis", axis);

    auto* q_concat_output = builder.MakeIntermediate();
    if (has_output_int8) {
      builder.AddQuantizeLinearNode<int8_t>(concat_output, 0.05f, 1, q_concat_output);

      auto* output_arg = builder.MakeOutput();
      builder.AddDequantizeLinearNode<int8_t>(q_concat_output, 0.05f, 1, output_arg);
    } else {
      builder.AddQuantizeLinearNode<uint8_t>(concat_output, 0.05f, 128, q_concat_output);

      auto* output_arg = builder.MakeOutput();
      builder.AddDequantizeLinearNode<uint8_t>(q_concat_output, 0.05f, 128, output_arg);
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

}  // namespace test
}  // namespace onnxruntime