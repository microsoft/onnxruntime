
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <optional>

#include "core/graph/basic_types.h"

namespace onnxruntime {

namespace logging {
class Logger;
}

class GraphViewer;
class Node;
class NodeArg;
class NodeUnit;

// Get the min/max of a Clip operator. Reads values from attributes for opset < 11 and inputs after that.
// For opset 11+, if min/max are not constant initializers, will return false.
// For now we only support getting float min/max.
bool GetClipMinMax(const GraphViewer& graph_viewer, const Node& node,
                   float& min, float& max, const logging::Logger& logger);

/// <deprecated>GraphViewer GetConstantInitializer/IsConstantInitializer should be used to ensure the initializer is
/// constant. Low risk for Clip min/max but in general the infrastructure to check if an operator is supported needs
/// to be updated to not use InitializedTensorSet which may contain non-constant initializers.</deprecated>
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
  explicit NodeAttrHelper(const Node& node);

  // Get the attributes from the target node of the node_unit
  explicit NodeAttrHelper(const NodeUnit& node_unit);

  /*
   * Get with default
   */
  float Get(const std::string& key, float def_val) const;
  std::vector<float> Get(const std::string& key, const std::vector<float>& def_val) const;

  int64_t Get(const std::string& key, int64_t def_val) const;
  std::vector<int64_t> Get(const std::string& key, const std::vector<int64_t>& def_val) const;

  const std::string& Get(const std::string& key, const std::string& def_val) const;

  // Convert the i() or ints() of the attribute from int64_t to int32_t
  int32_t Get(const std::string& key, int32_t def_val) const;
  std::vector<int32_t> Get(const std::string& key, const std::vector<int32_t>& def_val) const;

  // Convert the i() or ints() of the attribute from int64_t to uint32_t
  uint32_t Get(const std::string& key, uint32_t def_val) const;
  std::vector<uint32_t> Get(const std::string& key, const std::vector<uint32_t>& def_val) const;

  /*
   * Get without default.
   */
  std::optional<float> GetFloat(const std::string& key) const;
  std::optional<std::vector<float>> GetFloats(const std::string& key) const;

  std::optional<int64_t> GetInt64(const std::string& key) const;
  std::optional<std::vector<int64_t>> GetInt64s(const std::string& key) const;

  std::optional<std::string> GetString(const std::string& key) const;

  bool HasAttr(const std::string& key) const;

 private:
  const NodeAttributes& node_attributes_;
};

}  // namespace onnxruntime
