// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <iomanip>
#include "core/providers/cann/cann_call.h"
#include "core/providers/cann/cann_inc.h"
#include "core/framework/float16.h"

namespace onnxruntime {
namespace cann {

struct CannPreparation {
  CannPreparation() {
    opAttr_ = aclopCreateAttr();
  }

  virtual ~CannPreparation() {
    for (auto desc : inputDesc_) {
      aclDestroyTensorDesc(desc);
    }

    for (auto desc : outputDesc_) {
      aclDestroyTensorDesc(desc);
    }

    for (auto buf : inputBuffers_) {
      aclDestroyDataBuffer(buf);
    }

    for (auto buf : outputBuffers_) {
      aclDestroyDataBuffer(buf);
    }

    aclopDestroyAttr(opAttr_);
  }

  std::vector<aclDataBuffer*> inputBuffers_;
  std::vector<aclDataBuffer*> outputBuffers_;
  std::vector<aclTensorDesc*> inputDesc_;
  std::vector<aclTensorDesc*> outputDesc_;
  aclopAttr* opAttr_;
};

template <typename T>
aclDataType getACLType();

#define CANN_PREPARE_INPUTDESC(var, ...)           \
  do {                                             \
    auto _rPtr = aclCreateTensorDesc(__VA_ARGS__); \
    if (_rPtr == nullptr)                          \
      ORT_THROW("aclCreateTensorDesc run failed"); \
    else                                           \
      var.inputDesc_.push_back(_rPtr);             \
  } while (0)

#define CANN_PREPARE_OUTPUTDESC(var, ...)          \
  do {                                             \
    auto _rPtr = aclCreateTensorDesc(__VA_ARGS__); \
    if (_rPtr == nullptr)                          \
      ORT_THROW("aclCreateTensorDesc run failed"); \
    else                                           \
      var.outputDesc_.push_back(_rPtr);            \
  } while (0)

#define CANN_PREPARE_INPUTBUFFER(var, ...)         \
  do {                                             \
    auto _rPtr = aclCreateDataBuffer(__VA_ARGS__); \
    if (_rPtr == nullptr)                          \
      ORT_THROW("aclCreateDataBuffer run failed"); \
    else                                           \
      var.inputBuffers_.push_back(_rPtr);          \
  } while (0)

#define CANN_PREPARE_OUTPUTBUFFER(var, ...)        \
  do {                                             \
    auto _rPtr = aclCreateDataBuffer(__VA_ARGS__); \
    if (_rPtr == nullptr)                          \
      ORT_THROW("aclCreateDataBuffer run failed"); \
    else                                           \
      var.outputBuffers_.push_back(_rPtr);         \
  } while (0)

#define CANN_RETURN_IF_ERROR(expr)               \
  ORT_RETURN_IF_ERROR(CANN_CALL(expr)            \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CANN error executing ", #expr))

}  // namespace cann
}  // namespace onnxruntime
