//
// Created by daquexian on 8/3/18.
//

#pragma once

#include <onnx/onnx_pb.h>
#include <string>

/**
 * Wrapping onnx::NodeProto for retrieving attribute values
 */
class NodeAttrHelper {
 public:
  NodeAttrHelper(const ONNX_NAMESPACE::NodeProto& proto);

  float get(const std::string& key, float def_val);
  int32_t get(const std::string& key, int32_t def_val);
  std::vector<float> get(const std::string& key, std::vector<float> def_val);
  std::vector<int32_t> get(const std::string& key, std::vector<int32_t> def_val);
  std::string get(const std::string& key, std::string def_val);

  bool has_attr(const std::string& key);

 private:
  const ONNX_NAMESPACE::NodeProto& node_;
};
