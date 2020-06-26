//
// Created by daquexian on 8/3/18.
//

#include <iostream>
#include <string>
#include <vector>

#include "core/common/safeint.h"
#include "node_attr_helper.h"

using std::string;
using std::vector;

NodeAttrHelper::NodeAttrHelper(const ONNX_NAMESPACE::NodeProto& proto) : node_(proto) {
}

float NodeAttrHelper::Get(const std::string& key, float def_val) {
  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      return attr.f();
    }
  }

  return def_val;
}

int32_t NodeAttrHelper::Get(const std::string& key, int32_t def_val) {
  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      int64_t val = attr.i();
      return SafeInt<int32_t>(val);
    }
  }

  return def_val;
}

string NodeAttrHelper::Get(const std::string& key, const string& def_val) {
  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      return attr.s();
    }
  }

  return def_val;
}

vector<int32_t> NodeAttrHelper::Get(const std::string& key, const vector<int32_t>& def_val) {
  if (!HasAttr(key)) {
    return def_val;
  }

  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      std::vector<int32_t> v;
      v.reserve(static_cast<size_t>(attr.ints_size()));
      for (int j = 0; j < attr.ints_size(); j++) {
        int64_t val = attr.ints(j);
        v.push_back(SafeInt<int32_t>(val));
      }
      return v;
    }
  }

  return def_val;
}

vector<float> NodeAttrHelper::Get(const std::string& key,
                                  const vector<float>& def_val) {
  if (!HasAttr(key)) {
    return def_val;
  }

  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      std::vector<float> v;
      v.reserve(static_cast<size_t>(attr.floats_size()));
      for (int j = 0; j < attr.floats_size(); j++) {
        v.push_back(attr.floats(j));
      }

      return v;
    }
  }

  return def_val;
}

bool NodeAttrHelper::HasAttr(const std::string& key) {
  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      return true;
    }
  }

  return false;
}
