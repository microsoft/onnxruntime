// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {

enum class MLStatus : uint32_t {
  OK = 0,
  FAIL = 1,
  INVALID_ARGUMENT = 2,
  NO_SUCHFILE = 3,
  NO_MODEL = 4,
  ENGINE_ERROR = 5,
  RUNTIME_EXCEPTION = 6,
  INVALID_PROTOBUF = 7,
  MODEL_LOADED = 8,
  NOT_IMPLEMENTED = 9,
  INVALID_GRAPH = 10,
  SHAPE_INFERENCE_NOT_REGISTERED = 11,
  REQUIREMENT_NOT_REGISTERED = 12
};

inline const char* MLStatusToString(MLStatus status) noexcept {
  switch (status) {
    case MLStatus::OK:
      return "SUCCESS";
    case MLStatus::INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case MLStatus::NO_SUCHFILE:
      return "NO_SUCHFILE";
    case MLStatus::NO_MODEL:
      return "NO_MODEL";
    case MLStatus::ENGINE_ERROR:
      return "ENGINE_ERROR";
    case MLStatus::RUNTIME_EXCEPTION:
      return "RUNTIME_EXCEPTION";
    case MLStatus::INVALID_PROTOBUF:
      return "INVALID_PROTOBUF";
    case MLStatus::MODEL_LOADED:
      return "MODEL_LOADED";
    case MLStatus::NOT_IMPLEMENTED:
      return "NOT_IMPLEMENTED";
    case MLStatus::INVALID_GRAPH:
      return "INVALID_GRAPH";
    case MLStatus::SHAPE_INFERENCE_NOT_REGISTERED:
      return "SHAPE_INFERENCE_NOT_REGISTERED";
    case MLStatus::REQUIREMENT_NOT_REGISTERED:
      return "REQUIREMENT_NOT_REGISTERED";
    default:
      return "GENERAL ERROR";
  }
}

}  // namespace onnxruntime
