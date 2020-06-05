//
// Created by daquexian on 5/21/18.
//
#pragma once

#include <android/log.h>
#include <string>
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksTypes.h"
#include <vector>

#define LOG_TAG "ORT NNAPI"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)

#define THROW_ON_ERROR(val)                                             \
  {                                                                     \
    const auto ret = (val);                                             \
    if (ret != ANEURALNETWORKS_NO_ERROR) {                              \
      throw std::invalid_argument(                                      \
          std::string("Error in ") + __FILE__ + std::string(":") +      \
          std::to_string(__LINE__) + std::string(", function name: ") + \
          std::string(__func__) + "error, ret: " + GetErrorCause(ret)); \
    }                                                                   \
  }

#define THROW_ON_ERROR_WITH_NOTE(val, note)                             \
  {                                                                     \
    const auto ret = (val);                                             \
    if (ret != ANEURALNETWORKS_NO_ERROR) {                              \
      throw std::invalid_argument(                                      \
          std::string("Error in ") + __FILE__ + std::string(":") +      \
          std::to_string(__LINE__) + std::string(", function name: ") + \
          std::string(__func__) + "error, ret: " + GetErrorCause(ret) + \
          std::string(", ") + (note));                                  \
    }                                                                   \
  }

inline std::string GetErrorCause(int errorCode) {
  switch (errorCode) {
    case ANEURALNETWORKS_OUT_OF_MEMORY:
      return "Out of memory";
    case ANEURALNETWORKS_BAD_DATA:
      return "Bad data";
    case ANEURALNETWORKS_BAD_STATE:
      return "Bad state";
    case ANEURALNETWORKS_INCOMPLETE:
      return "Incomplete";
    case ANEURALNETWORKS_UNEXPECTED_NULL:
      return "Unexpected null";
    case ANEURALNETWORKS_OP_FAILED:
      return "Op failed";
    case ANEURALNETWORKS_UNMAPPABLE:
      return "Unmappable";
    case ANEURALNETWORKS_NO_ERROR:
      return "No error";
    default:
      return "Unknown error code";
  }
}
