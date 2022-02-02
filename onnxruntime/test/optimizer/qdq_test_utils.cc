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

}  // namespace test
}  // namespace onnxruntime