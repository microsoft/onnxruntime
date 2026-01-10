//
// Created by daquexian on 8/3/18.
//

#include "node_attr_helper.h"

#include <iostream>
#include <string>
#include <vector>

namespace onnxruntime {
namespace rknpu {

NodeAttrHelper::NodeAttrHelper(ONNX_NAMESPACE::NodeProto proto)
    : node_(proto) {
}

float NodeAttrHelper::get(const std::string& key,
                          const float def_val) const {
  for (const auto& attr : node_.attribute()) {
    if (attr.name() == key) {
      return attr.f();
    }
  }

  return def_val;
}

int NodeAttrHelper::get(const std::string& key,
                        const int def_val) const {
  for (const auto& attr : node_.attribute()) {
    if (attr.name() == key) {
      return static_cast<int>(attr.i());
    }
  }

  return def_val;
}

std::string NodeAttrHelper::get(const std::string& key,
                                const std::string& def_val) const {
  for (const auto& attr : node_.attribute()) {
    if (attr.name() == key) {
      return attr.s();
    }
  }

  return def_val;
}

std::vector<int> NodeAttrHelper::get(const std::string& key,
                                     const std::vector<int>& def_val) const {
  if (!has_attr(key)) {
    return def_val;
  }
  std::vector<int> v;

  for (const auto& attr : node_.attribute()) {
    if (attr.name() == key) {
      v.reserve(static_cast<size_t>(attr.ints_size()));
      for (int j = 0; j < attr.ints_size(); j++) {
        v.push_back(static_cast<int>(attr.ints(j)));
      }

      break;
    }
  }

  if (v.empty()) {
    return def_val;
  }

  return v;
}

std::vector<float> NodeAttrHelper::get(const std::string& key,
                                       const std::vector<float>& def_val) const {
  if (!has_attr(key)) {
    return def_val;
  }
  std::vector<float> v;

  for (const auto& attr : node_.attribute()) {
    if (attr.name() == key) {
      v.reserve(static_cast<size_t>(attr.floats_size()));
      for (int j = 0; j < attr.floats_size(); j++) {
        v.push_back(attr.floats(j));
      }

      break;
    }
  }

  if (v.empty()) {
    return def_val;
  }

  return v;
}

bool NodeAttrHelper::has_attr(const std::string& key) const {
  for (const auto& attr : node_.attribute()) {
    if (attr.name() == key) {
      return true;
    }
  }

  return false;
}

}  // namespace rknpu
}  // namespace onnxruntime