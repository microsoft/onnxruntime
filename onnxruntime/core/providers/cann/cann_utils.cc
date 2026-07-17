// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <string>
#include <filesystem>
#include <system_error>

#ifndef _WIN32
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#include "core/providers/cann/cann_utils.h"

namespace onnxruntime {
namespace cann {

template <typename T>
aclDataType getACLType() {
  return ACL_DT_UNDEFINED;
}

#define GET_ACL_TYPE(onnx_type, cann_type) \
  template <>                              \
  aclDataType getACLType<onnx_type>() {    \
    return cann_type;                      \
  }

GET_ACL_TYPE(int8_t, ACL_INT8);
GET_ACL_TYPE(int16_t, ACL_INT16);
GET_ACL_TYPE(int32_t, ACL_INT32);
GET_ACL_TYPE(int64_t, ACL_INT64);
GET_ACL_TYPE(uint8_t, ACL_UINT8);
GET_ACL_TYPE(uint16_t, ACL_UINT16);
GET_ACL_TYPE(uint32_t, ACL_UINT32);
GET_ACL_TYPE(uint64_t, ACL_UINT64);
GET_ACL_TYPE(float, ACL_FLOAT);
GET_ACL_TYPE(MLFloat16, ACL_FLOAT16);
GET_ACL_TYPE(BFloat16, ACL_BF16);
GET_ACL_TYPE(double, ACL_DOUBLE);
GET_ACL_TYPE(bool, ACL_BOOL);

template <typename T>
Status Fill(Tensor* y, void* addr, aclrtStream stream) {
  int64_t one = 1;
  int64_t dims = static_cast<int64_t>(y->Shape().NumDimensions());

  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  CannPreparation prepare;

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, ACL_INT64, 1, &dims, format);
    CANN_CONST_INPUTDESC(prepare, 0, const_cast<int64_t*>(y->Shape().GetDims().data()), dims * sizeof(int64_t));
    CANN_PREPARE_INPUTDESC(prepare, aclType, 1, &one, format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, dims, y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<int64_t*>(y->Shape().GetDims().data()), dims * sizeof(int64_t));
    CANN_PREPARE_INPUTBUFFER(prepare, addr, sizeof(T));
    CANN_PREPARE_OUTPUTBUFFER(prepare, y->MutableDataRaw(), y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("Fill",
                                              prepare.inputDesc_.size(),
                                              prepare.inputDesc_.data(),
                                              prepare.inputBuffers_.data(),
                                              prepare.outputDesc_.size(),
                                              prepare.outputDesc_.data(),
                                              prepare.outputBuffers_.data(),
                                              prepare.opAttr_,
                                              ACL_ENGINE_SYS,
                                              ACL_COMPILE_SYS,
                                              NULL,
                                              stream));

  return Status::OK();
}

template Status Fill<uint8_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<uint16_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<uint32_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<uint64_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<int8_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<int16_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<int32_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<int64_t>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<MLFloat16>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<float>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<double>(Tensor* y, void* addr, aclrtStream stream);
template Status Fill<bool>(Tensor* y, void* addr, aclrtStream stream);

template <typename T>
Status Broadcast(const Tensor* x, Tensor* y, void* addr, aclrtStream stream) {
  const aclDataType aclType = getACLType<T>();
  aclFormat format = ACL_FORMAT_ND;

  CannPreparation prepare;

  CANN_RETURN_IF_ERROR(aclopSetAttrListInt(prepare.opAttr_, "shape", static_cast<int64_t>(y->Shape().NumDimensions()),
                                           y->Shape().GetDims().data()));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, aclType, x->Shape().NumDimensions(), x->Shape().GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, aclType, y->Shape().NumDimensions(), y->Shape().GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(x->DataRaw()), x->SizeInBytes());
    CANN_PREPARE_OUTPUTBUFFER(prepare, addr, y->SizeInBytes());
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("BroadcastToD",
                                              prepare.inputDesc_.size(),
                                              prepare.inputDesc_.data(),
                                              prepare.inputBuffers_.data(),
                                              prepare.outputDesc_.size(),
                                              prepare.outputDesc_.data(),
                                              prepare.outputBuffers_.data(),
                                              prepare.opAttr_,
                                              ACL_ENGINE_SYS,
                                              ACL_COMPILE_SYS,
                                              NULL,
                                              stream));

  return Status::OK();
}

template Status Broadcast<uint8_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<uint16_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<uint32_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<uint64_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<int8_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<int16_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<int32_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<int64_t>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<MLFloat16>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<BFloat16>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<float>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<double>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);
template Status Broadcast<bool>(const Tensor* x, Tensor* y, void* addr, aclrtStream stream);

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
                       aclrtStream stream) {
  ORT_UNUSED_PARAMETER(transC);
  ORT_UNUSED_PARAMETER(lda);
  ORT_UNUSED_PARAMETER(ldb);
  ORT_UNUSED_PARAMETER(ldc);
  ORT_UNUSED_PARAMETER(type);

  TensorShape A;
  TensorShape B;
  TensorShape C{m, n};
  TensorShape shape{1};

  A = transA ? TensorShape{k, m} : TensorShape{m, k};
  B = transB ? TensorShape{n, k} : TensorShape{k, n};

  aclFormat format = ACL_FORMAT_ND;

  CannPreparation prepare;

  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "transpose_a", transA));
  CANN_RETURN_IF_ERROR(aclopSetAttrBool(prepare.opAttr_, "transpose_b", transB));

  ORT_TRY {
    CANN_PREPARE_INPUTDESC(prepare, dataTypeA, A.NumDimensions(), A.GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, dataTypeB, B.NumDimensions(), B.GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, dataTypeC, C.NumDimensions(), C.GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, dataTypeC, shape.NumDimensions(), shape.GetDims().data(), format);
    CANN_PREPARE_INPUTDESC(prepare, dataTypeC, shape.NumDimensions(), shape.GetDims().data(), format);
    CANN_PREPARE_OUTPUTDESC(prepare, dataTypeC, C.NumDimensions(), C.GetDims().data(), format);

    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(matrixA), A.Size() * aclDataTypeSize(dataTypeA));
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(matrixB), B.Size() * aclDataTypeSize(dataTypeB));
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(matrixC), C.Size() * aclDataTypeSize(dataTypeC));
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(alpha), aclDataTypeSize(dataTypeC));
    CANN_PREPARE_INPUTBUFFER(prepare, const_cast<void*>(beta), aclDataTypeSize(dataTypeC));
    CANN_PREPARE_OUTPUTBUFFER(prepare, const_cast<void*>(matrixC), C.Size() * aclDataTypeSize(dataTypeC));
  }
  ORT_CATCH(const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  CANN_RETURN_IF_ERROR(aclopCompileAndExecute("GEMM",
                                              prepare.inputDesc_.size(),
                                              prepare.inputDesc_.data(),
                                              prepare.inputBuffers_.data(),
                                              prepare.outputDesc_.size(),
                                              prepare.outputDesc_.data(),
                                              prepare.outputBuffers_.data(),
                                              prepare.opAttr_,
                                              ACL_ENGINE_SYS,
                                              ACL_COMPILE_SYS,
                                              NULL,
                                              stream));

  return Status::OK();
}

bool FileExist(const std::string& file_name) {
  return (access(file_name.c_str(), F_OK) != -1);
}

void GenerateHashValue(const std::string string, HashValue& hash_value) {
  uint32_t hash[4] = {0, 0, 0, 0};
  MurmurHash3::x86_128(string.data(), string.size(), hash[0], &hash);
  hash_value = hash[0] | (uint64_t(hash[1]) << 32);
}

bool is_dynamic_shape(const aclmdlIODims& dims) {
  return std::find(dims.dims, dims.dims + dims.dimCount, -1) != dims.dims + dims.dimCount;
}

namespace fs = std::filesystem;
std::string MatchFile(const std::string& file_name) {
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

static bool repeat_acl_init_flag = false;

bool GetRepeatInitFlag() {
  return repeat_acl_init_flag;
}

void SetRepeatInitFlag(bool val) {
  repeat_acl_init_flag = val;
}

InterprocessFileLock::InterprocessFileLock(const std::string& name, bool temp)
    : filepath_{name + std::string(".lock")}, is_locked_{false} {
  if (temp) {
    filepath_ = (std::filesystem::temp_directory_path() / filepath_).string();
  }

#ifdef _WIN32
  handle_ = CreateFileA(
      filepath_.c_str(),
      GENERIC_READ | GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE,
      NULL,
      OPEN_ALWAYS,
      FILE_ATTRIBUTE_NORMAL,
      NULL);

  if (handle_ == INVALID_HANDLE_VALUE) {
    throw std::system_error(GetLastError(), std::system_category());
  }
#else

#ifdef O_CLOEXEC
  fd_ = open(filepath_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
#else
  fd_ = open(filepath_.c_str(), O_CREAT | O_RDWR, 0666);
#endif

  if (fd_ == -1) {
    throw std::system_error(errno, std::generic_category());
  }
#endif
}

InterprocessFileLock::~InterprocessFileLock() {
  if (is_locked_) {
    unlock();
  }

#ifdef _WIN32
  if (handle_ != INVALID_HANDLE_VALUE) {
    CloseHandle(handle_);
    handle_ = INVALID_HANDLE_VALUE;
  }
#else
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
#endif
}

void InterprocessFileLock::lock() {
  if (is_locked_) {
    return;
  }

#ifdef _WIN32
  OVERLAPPED overlapped = {0};
  auto result = LockFileEx(
      handle_,
      LOCKFILE_EXCLUSIVE_LOCK,
      0,
      MAXDWORD,
      MAXDWORD,
      &overlapped);

  if (!result) {
    throw std::system_error(GetLastError(), std::system_category());
  }
#else
  if (flock(fd_, LOCK_EX) == -1) {
    throw std::system_error(errno, std::generic_category());
  }
#endif

  is_locked_ = true;
}

bool InterprocessFileLock::try_lock() {
  if (is_locked_) {
    return false;
  }

#ifdef _WIN32
  OVERLAPPED overlapped = {0};
  auto result = LockFileEx(
      handle_,
      LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
      0,
      MAXDWORD,
      MAXDWORD,
      &overlapped);

  if (result) {
    is_locked_ = true;
    return true;
  }

  if (auto err = GetLastError(); err != ERROR_LOCK_VIOLATION) {
    throw std::system_error(err, std::system_category());
  }

  return false;
#else
  if (flock(fd_, LOCK_EX | LOCK_NB) == -1) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return false;
    }

    throw std::system_error(errno, std::generic_category());
  }
#endif

  is_locked_ = true;
  return true;
}

void InterprocessFileLock::unlock() {
  if (!is_locked_) {
    return;
  }

#ifdef _WIN32
  OVERLAPPED overlapped = {0};
  UnlockFileEx(handle_, 0, MAXDWORD, MAXDWORD, &overlapped);
#else
  flock(fd_, LOCK_UN);
#endif

  is_locked_ = false;
}
}  // namespace cann
}  // namespace onnxruntime
