//
// Created by daquexian on 8/3/18.
//

#include "node_attr_helper.h"

#include <iostream>
#include <string>
#include <vector>

using std::string;
using std::vector;

namespace onnxruntime {

NodeAttrHelper::NodeAttrHelper(ONNX_NAMESPACE::NodeProto proto) : node_(proto) {
}

float NodeAttrHelper::get(const std::string& key, float def_val) {
  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      return attr.f();
    }
  }

  return def_val;
}

int NodeAttrHelper::get(const std::string& key, int def_val) {
  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      return static_cast<int>(attr.i());
    }
  }

  return def_val;
}

string NodeAttrHelper::get(const std::string& key, string def_val) {
  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      return attr.s();
    }
  }

  return def_val;
}

vector<int> NodeAttrHelper::get(const std::string& key, vector<int> def_val) {
  if (!has_attr(key)) {
    return def_val;
  }
  std::vector<int> v;

  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
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

vector<float> NodeAttrHelper::get(const std::string& key,
                                  vector<float> def_val) {
  if (!has_attr(key)) {
    return def_val;
  }
  std::vector<float> v;

  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
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

bool NodeAttrHelper::has_attr(const std::string& key) {
  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      return true;
    }
  }

  return false;
}
}  // namespace onnxruntime