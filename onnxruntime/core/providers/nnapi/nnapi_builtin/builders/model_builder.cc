// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "android_log_helper.h"

namespace onnxruntime {
namespace nnapi {

ModelBuilder::ModelBuilder(ONNX_NAMESPACE::ModelProto& model_proto, const NnApi* nnapi)
    : nnapi_(nnapi), model_proto_(model_proto) {}

std::pair<bool, std::string> ModelBuilder::IsNodeSupported(
    const ONNX_NAMESPACE::NodeProto& node) {
  (void)node;
  int32_t android_sdk_ver = nnapi_ ? nnapi_->android_sdk_version : 0;

#ifdef __ANDROID__
  if (android_sdk_ver < 27) {
    LOGI("Android API level %d is lower than 27", android_sdk_ver);
    return {false, "Android API level is lower than 27"};
  }
#endif

  const auto& op = node.op_type();
  std::map<std::string, int> supported_ops{{"Add", 27}};

  if (supported_ops.find(op) == supported_ops.end()) {
    LOGI("Unsupported operator %s", op.c_str());
    return {false, "Unsupported operator " + op};
  }

#ifdef __ANDROID__
  if (supported_ops[op] > android_sdk_ver) {
    LOGI("Android API level %d is lower than %d", android_sdk_ver, supported_ops[op]);
    return {false, "Operator " + op + " is only supported on API > " + std::to_string(supported_ops[op])};
  }
#endif

  LOGI("Supported operator %s", op.c_str());
  return {true, ""};
}

bool IsValidSupportedNodesVec(const std::vector<int>& supported_node_vec,
                              const ONNX_NAMESPACE::ModelProto& model_proto) {
  if (!supported_node_vec.empty()) {
    if (supported_node_vec.size() == 1) {
      const auto& node = model_proto.graph().node(supported_node_vec[0]);
      // It is not worth it to perform a single Reshape/Dropout/Identity operator
      // which is only copying the data in NNAPI
      // If this is the case, let it fall back
      if (node.op_type() == "Reshape" || node.op_type() == "Dropout" ||
          node.op_type() == "Identity") {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::vector<std::vector<int>> ModelBuilder::GetSupportedNodes() {
  // ONNX_NAMESPACE::shape_inference::InferShapes(model_proto_);

  std::vector<std::vector<int>> supported_node_vecs;
  std::vector<int> supported_node_vec;

  for (int i = 0; i < model_proto_.graph().node_size(); i++) {
    bool supported;
    std::string error_msg;
    std::tie(supported, error_msg) = IsNodeSupported(model_proto_.graph().node(i));
    if (supported) {
      supported_node_vec.push_back(i);
    } else {
      if (IsValidSupportedNodesVec(supported_node_vec, model_proto_)) {
        supported_node_vecs.push_back(supported_node_vec);
        supported_node_vec.clear();
      }
    }
  }

  if (IsValidSupportedNodesVec(supported_node_vec, model_proto_))
    supported_node_vecs.push_back(supported_node_vec);

  return supported_node_vecs;
}

}  // namespace nnapi
}  // namespace onnxruntime