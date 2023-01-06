// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <utility>
#include <iomanip>
#include <string>
#include <memory>

#include "core/framework/murmurhash3.h"
#include "core/providers/cann/cann_common.h"
#include "core/providers/cann/cann_inc.h"

namespace onnxruntime {
namespace cann {

struct CannPreparation {
  CannPreparation() {
    opAttr_ = aclopCreateAttr();
    ORT_ENFORCE(opAttr_ != nullptr, "aclopCreateAttr run failed");
  }

  virtual ~CannPreparation() {
    for (auto desc : inputDesc_) {
      aclDestroyTensorDesc(desc);
    }

    for (auto desc : outputDesc_) {
      aclDestroyTensorDesc(desc);
    }

    for (auto buf : inputBuffers_) {
      CANN_CALL_THROW(aclDestroyDataBuffer(buf));
    }

    for (auto buf : outputBuffers_) {
      CANN_CALL_THROW(aclDestroyDataBuffer(buf));
    }

    aclopDestroyAttr(opAttr_);
  }

  std::vector<aclDataBuffer*> inputBuffers_;
  std::vector<aclDataBuffer*> outputBuffers_;
  std::vector<aclTensorDesc*> inputDesc_;
  std::vector<aclTensorDesc*> outputDesc_;
  aclopAttr* opAttr_;
};

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

#define CANN_CONST_INPUTDESC(var, index, ...)                           \
  do {                                                                  \
    auto _rPtr = aclSetTensorConst(var.inputDesc_[index], __VA_ARGS__); \
    if (_rPtr != ACL_SUCCESS)                                           \
      ORT_THROW("aclSetTensorConst run failed");                        \
  } while (0)

template <typename T>
aclDataType getACLType();

template <typename T>
Status Fill(Tensor* y, void* addr, aclrtStream stream);

template <typename T>
Status Broadcast(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);

Status aclrtblasGemmEx(aclTransType transA,
                       aclTransType transB,
                       aclTransType transC,
                       int m,
                       int n,
                       int k,
                       const void* alpha,
                       const void* matrixA,
                       int lda,
                       aclDataType dataTypeA,
                       const void* matrixB,
                       int ldb,
                       aclDataType dataTypeB,
                       const void* beta,
                       void* matrixC,
                       int ldc,
                       aclDataType dataTypeC,
                       aclComputeType type,
                       aclrtStream stream);

bool FileExist(const std::string& file_name);
void GenerateHashValue(const std::string string, HashValue& hash_value);
std::unique_ptr<Model> CreateModel(const GraphViewer& graph_viewer, const logging::Logger& logger);

}  // namespace cann
}  // namespace onnxruntime
