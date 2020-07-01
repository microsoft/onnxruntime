//
// Created by daquexian on 8/3/18.
//

#include <core/common/safeint.h>
#include <core/graph/graph.h>
#include <iostream>
#include <string>
#include <vector>

#include "helper.h"

using std::string;
using std::vector;

NodeAttrHelper::NodeAttrHelper(const onnxruntime::Node& node)
    : node_attributes_(node.GetAttributes()) {}

float NodeAttrHelper::Get(const std::string& key, float def_val) const {
  if (HasAttr(key))
    return node_attributes_.at(key).f();

  return def_val;
}

int32_t NodeAttrHelper::Get(const std::string& key, int32_t def_val) const {
  if (HasAttr(key))
    return SafeInt<int32_t>(node_attributes_.at(key).i());

  return def_val;
}

string NodeAttrHelper::Get(const std::string& key, const string& def_val) const {
  if (HasAttr(key))
    return node_attributes_.at(key).s();

  return def_val;
}

vector<int32_t> NodeAttrHelper::Get(const std::string& key, const vector<int32_t>& def_val) const {
  if (HasAttr(key)) {
    const auto& attr(node_attributes_.at(key));
    std::vector<int32_t> v;
    v.reserve(static_cast<size_t>(attr.ints_size()));
    for (int j = 0; j < attr.ints_size(); j++) {
      int64_t val = attr.ints(j);
      v.push_back(SafeInt<int32_t>(val));
    }
    return v;
  }

  return def_val;
}

vector<float> NodeAttrHelper::Get(const std::string& key, const vector<float>& def_val) const {
  if (HasAttr(key)) {
    const auto& attr(node_attributes_.at(key));
    std::vector<float> v;
    v.reserve(static_cast<size_t>(attr.ints_size()));
    for (int j = 0; j < attr.ints_size(); j++) {
      v.push_back(attr.floats(j));
    }

    return v;
  }

  return def_val;
}

bool NodeAttrHelper::HasAttr(const std::string& key) const {
  return Contains(node_attributes_, key);
}
