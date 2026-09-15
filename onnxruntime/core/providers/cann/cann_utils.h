// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <utility>
#include <iomanip>
#include <string>
#include <memory>
#include <filesystem>
#include <regex>
#include <exception>
#include <gsl/util>
#include <random>
#include <system_error>

#include "core/framework/murmurhash3.h"
#include "core/providers/cann/cann_common.h"
#include "core/providers/cann/cann_inc.h"

namespace onnxruntime {
namespace cann {
namespace detail {

template <typename F>
Status SaveFileAtomically(const std::string& file_name, F&& save_fn) {
  namespace fs = std::filesystem;

  try {
    const auto file_dir = fs::absolute(file_name).parent_path();

    fs::path tmp_dir;
    std::random_device random;
    bool created = false;

    // MatchFile does not perform subdirectory lookup
    for (int attempt = 0; attempt < 64 && !created; ++attempt) {
      tmp_dir = file_dir / (".ort-cann-tmp-" + std::to_string(random()));
      created = fs::create_directory(tmp_dir);
    }

    ORT_RETURN_IF_NOT(created, "Could not create a temporary directory in ", file_dir.string());

    auto cleanup = gsl::finally([&tmp_dir] {
      std::error_code ec;
      fs::remove_all(tmp_dir, ec);
    });

    const auto tmp_file_name = (tmp_dir / fs::path(file_name).filename()).string();
    ORT_RETURN_IF_ERROR(std::forward<F>(save_fn)(tmp_file_name));

    for (const auto& entry : fs::directory_iterator(tmp_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".om") {
        fs::rename(entry.path(), file_dir / entry.path().filename());
      }
    }
  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to save file: ", e.what());
  }

  return Status::OK();
}

}  // namespace detail

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
bool is_dynamic_shape(const aclmdlIODims& dims);

inline std::string MatchFile(const std::string& file_name) {
  namespace fs = std::filesystem;
  fs::path current_dir = fs::current_path();

  for (const auto& entry : fs::directory_iterator(current_dir)) {
    if (entry.is_regular_file()) {
      std::string name = entry.path().filename().string();
      if (name.find(file_name) != std::string::npos && entry.path().extension() == ".om") {
        return name;
      }
    }
  }
  return "";
}

Status SaveFile(const std::string& file_name, const ge::ModelBufferData& model);
std::unique_ptr<Model> CreateModel(const GraphViewer& graph_viewer, const logging::Logger& logger);
bool GetRepeatInitFlag();
void SetRepeatInitFlag(bool val);
}  // namespace cann
}  // namespace onnxruntime
