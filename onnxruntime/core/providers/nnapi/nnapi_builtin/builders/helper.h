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

inline std::string GetErrorCause(int error_code) {
  switch (error_code) {
    case ANEURALNETWORKS_NO_ERROR:
      return "ANEURALNETWORKS_NO_ERROR";
    case ANEURALNETWORKS_OUT_OF_MEMORY:
      return "ANEURALNETWORKS_OUT_OF_MEMORY";
    case ANEURALNETWORKS_INCOMPLETE:
      return "ANEURALNETWORKS_INCOMPLETE";
    case ANEURALNETWORKS_UNEXPECTED_NULL:
      return "ANEURALNETWORKS_UNEXPECTED_NULL";
    case ANEURALNETWORKS_BAD_DATA:
      return "ANEURALNETWORKS_BAD_DATA";
    case ANEURALNETWORKS_OP_FAILED:
      return "ANEURALNETWORKS_OP_FAILED";
    case ANEURALNETWORKS_BAD_STATE:
      return "ANEURALNETWORKS_BAD_STATE";
    case ANEURALNETWORKS_UNMAPPABLE:
      return "ANEURALNETWORKS_UNMAPPABLE";
    case ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE:
      return "ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE";
    case ANEURALNETWORKS_UNAVAILABLE_DEVICE:
      return "ANEURALNETWORKS_UNAVAILABLE_DEVICE";

    default:
      return "Unknown error code: " + std::to_string(error_code);
  }
}

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
