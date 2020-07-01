//
// Created by daquexian on 5/21/18.
//
#pragma once

#include <core/graph/graph.h>
#include <string>

#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksTypes.h"

#define THROW_ON_ERROR(val)                                               \
  {                                                                       \
    const auto ret = (val);                                               \
    ORT_ENFORCE(                                                          \
        ret == ANEURALNETWORKS_NO_ERROR,                                  \
        std::string("Error in ") + __FILE__ + std::string(":") +          \
            std::to_string(__LINE__) + std::string(", function name: ") + \
            std::string(__func__) + "error, ret: " + GetErrorCause(ret)); \
  }

#define THROW_ON_ERROR_WITH_NOTE(val, note)                               \
  {                                                                       \
    const auto ret = (val);                                               \
    ORT_ENFORCE(                                                          \
        ret == ANEURALNETWORKS_NO_ERROR,                                  \
        std::string("Error in ") + __FILE__ + std::string(":") +          \
            std::to_string(__LINE__) + std::string(", function name: ") + \
            std::string(__func__) + "error, ret: " + GetErrorCause(ret) + \
            std::string(", ") + (note));                                  \
  }

template <class Map, class Key>
inline bool Contains(const Map& map, const Key& key) {
  return map.find(key) != map.end();
}

std::string GetErrorCause(int error_code);

/**
 * Wrapping onnxruntime::Node for retrieving attribute values
 */
class NodeAttrHelper {
 public:
  NodeAttrHelper(const onnxruntime::Node& proto);

  float Get(const std::string& key, float def_val) const;
  int32_t Get(const std::string& key, int32_t def_val) const;
  std::vector<float> Get(const std::string& key, const std::vector<float>& def_val) const;
  std::vector<int32_t> Get(const std::string& key, const std::vector<int32_t>& def_val) const;
  std::string Get(const std::string& key, const std::string& def_val) const;

  bool HasAttr(const std::string& key) const;

 private:
  const onnxruntime::NodeAttributes& node_attributes_;
};
