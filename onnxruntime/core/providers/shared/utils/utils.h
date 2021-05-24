
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/graph/basic_types.h"

namespace onnxruntime {

namespace logging {
class Logger;
}

class Node;
class NodeArg;

// Get initialize tensort float/int32/int64 data without unpacking
// NOTE!!! This will not work when the initializer has external data
const float* GetTensorFloatData(const ONNX_NAMESPACE::TensorProto& tensor);
const int32_t* GetTensorInt32Data(const ONNX_NAMESPACE::TensorProto& tensor);
const int64_t* GetTensorInt64Data(const ONNX_NAMESPACE::TensorProto& tensor);

// Get the min/max of a Clip operator.
// If min/max are not known initializer tensors, will return false
// For now we only support getting float min/max,
// since in most cases, Clip(0,6)[Relu6] will be fused by quantization tool
bool GetClipMinMax(const InitializedTensorSet& initializers, const Node& node,
                   float& min, float& max, const logging::Logger& logger);

// Get the type of the given NodeArg
// Will return false if the given NodeArg has no type
bool GetType(const NodeArg& node_arg, int32_t& type, const logging::Logger& logger);

/**
 * Wrapping onnxruntime::Node for retrieving attribute values
 */
class NodeAttrHelper {
 public:
  NodeAttrHelper(const onnxruntime::Node& node);

  float Get(const std::string& key, float def_val) const;

  int64_t Get(const std::string& key, int64_t def_val) const;

  std::string Get(const std::string& key, const std::string& def_val) const;

  std::vector<int64_t> Get(const std::string& key, const std::vector<int64_t>& def_val) const;
  std::vector<float> Get(const std::string& key, const std::vector<float>& def_val) const;

  // Convert the i() or ints() of the attribute from int64_t to int32_t
  int32_t Get(const std::string& key, int32_t def_val) const;
  std::vector<int32_t> Get(const std::string& key, const std::vector<int32_t>& def_val) const;

  bool HasAttr(const std::string& key) const;

 private:
  const onnxruntime::NodeAttributes& node_attributes_;
};

}  // namespace onnxruntime
