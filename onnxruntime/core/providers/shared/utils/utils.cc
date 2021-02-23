
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"

#include <core/common/safeint.h>
#include <core/graph/graph.h>

#include "core/providers/common.h"

namespace onnxruntime {

NodeAttrHelper::NodeAttrHelper(const onnxruntime::Node& node)
    : node_attributes_(node.GetAttributes()) {}

float NodeAttrHelper::Get(const std::string& key, float def_val) const {
  if (!HasAttr(key))
    return def_val;

  return node_attributes_.at(key).f();
}

int32_t NodeAttrHelper::Get(const std::string& key, int32_t def_val) const {
  if (!HasAttr(key))
    return def_val;

  return SafeInt<int32_t>(node_attributes_.at(key).i());
}

int64_t NodeAttrHelper::Get(const std::string& key, int64_t def_val) const {
  if (!HasAttr(key))
    return def_val;

  return node_attributes_.at(key).i();
}

std::string NodeAttrHelper::Get(const std::string& key, const std::string& def_val) const {
  if (!HasAttr(key))
    return def_val;

  return node_attributes_.at(key).s();
}

std::vector<int32_t> NodeAttrHelper::Get(const std::string& key, const std::vector<int32_t>& def_val) const {
  if (!HasAttr(key))
    return def_val;

  const auto& attr(node_attributes_.at(key));
  std::vector<int32_t> v;
  v.reserve(static_cast<size_t>(attr.ints_size()));
  std::transform(attr.ints().cbegin(), attr.ints().cend(), std::back_inserter(v),
                 [](int64_t val) -> int32_t { return SafeInt<int32_t>(val); });
  return v;
}

std::vector<int64_t> NodeAttrHelper::Get(const std::string& key, const std::vector<int64_t>& def_val) const {
  if (!HasAttr(key))
    return def_val;

  const auto& source(node_attributes_.at(key).ints());
  return std::vector<int64_t>{source.cbegin(), source.cend()};
}

std::vector<float> NodeAttrHelper::Get(const std::string& key, const std::vector<float>& def_val) const {
  if (!HasAttr(key))
    return def_val;

  const auto& source(node_attributes_.at(key).floats());
  return std::vector<float>{source.cbegin(), source.cend()};
}

bool NodeAttrHelper::HasAttr(const std::string& key) const {
  return Contains(node_attributes_, key);
}

}  // namespace onnxruntime
