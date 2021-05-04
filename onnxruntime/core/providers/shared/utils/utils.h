
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/graph/basic_types.h"

namespace onnxruntime {

class Node;

// Get initialize tensort float/int32/int64 data without unpacking
// NOTE!!! This will not work when the initializer has external data
const float* GetTensorFloatData(const ONNX_NAMESPACE::TensorProto& tensor);
const int32_t* GetTensorInt32Data(const ONNX_NAMESPACE::TensorProto& tensor);
const int64_t* GetTensorInt64Data(const ONNX_NAMESPACE::TensorProto& tensor);

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
