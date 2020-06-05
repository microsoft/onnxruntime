//
// Created by daquexian on 8/3/18.
//

#include "NodeAttrHelper.h"

#include <iostream>
#include <string>
#include <vector>

using std::string;
using std::vector;

void CheckValidIntCast(int64_t val) {
  if (val < 0x80000000 || val > 0x7FFFFFFF)
    throw std::invalid_argument(
        "The int64_t cannot be casted to int32_t " + std::to_string(val));
}

NodeAttrHelper::NodeAttrHelper(const ONNX_NAMESPACE::NodeProto& proto) : node_(proto) {
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

int32_t NodeAttrHelper::get(const std::string& key, int32_t def_val) {
  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      int64_t val = attr.i();
      CheckValidIntCast(val);
      return static_cast<int>(val);
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

vector<int32_t> NodeAttrHelper::get(const std::string& key, vector<int32_t> def_val) {
  if (!has_attr(key)) {
    return def_val;
  }

  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      std::vector<int32_t> v;
      v.reserve(static_cast<size_t>(attr.ints_size()));
      for (int j = 0; j < attr.ints_size(); j++) {
        int64_t val = attr.ints(j);
        CheckValidIntCast(val);
        v.push_back(static_cast<int>(val));
      }
      return v;
    }
  }

  return def_val;
}

vector<float> NodeAttrHelper::get(const std::string& key,
                                  vector<float> def_val) {
  if (!has_attr(key)) {
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

bool NodeAttrHelper::has_attr(const std::string& key) {
  for (int i = 0; i < node_.attribute_size(); i++) {
    const ONNX_NAMESPACE::AttributeProto& attr = node_.attribute(i);
    if (attr.name() == key) {
      return true;
    }
  }

  return false;
}
