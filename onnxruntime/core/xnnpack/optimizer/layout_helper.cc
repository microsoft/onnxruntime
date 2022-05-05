// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/xnnpack/optimizer/layout_helper.h"

#include "core/graph/constants.h"

namespace onnxruntime {
Status CreateTransposeNode(::ONNX_NAMESPACE::NodeProto& node, const std::string& node_name,
                           const std::string& input_name, const std::string& output_name,
                           const std::vector<int64_t>& perm) {
  node.set_name(node_name);
  node.set_domain(kOnnxDomain);
  node.set_op_type("Transpose");
  ::ONNX_NAMESPACE::AttributeProto* attr = node.add_attribute();
  attr->set_name("perm");
  for (int64_t i : perm) attr->add_ints(i);
  attr->set_type(::onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  node.add_input(input_name);
  node.add_output(output_name);
  return Status::OK();
}
}  // namespace onnxruntime