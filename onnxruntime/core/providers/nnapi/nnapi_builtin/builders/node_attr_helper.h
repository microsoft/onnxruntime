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

  float Get(const std::string& key, float def_val);
  int32_t Get(const std::string& key, int32_t def_val);
  std::vector<float> Get(const std::string& key, const std::vector<float>& def_val);
  std::vector<int32_t> Get(const std::string& key, const std::vector<int32_t>& def_val);
  std::string Get(const std::string& key, const std::string& def_val);

  bool HasAttr(const std::string& key);

 private:
  const ONNX_NAMESPACE::NodeProto& node_;
};
