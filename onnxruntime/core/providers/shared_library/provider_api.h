// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Provider implementations include this file

// NOTE: This is still in development so there are many parts that will be fixed in the future. This is just the first version of
//       switching providers to be runnable as shared libraries. The interfaces will become more tightly integrated into the core code.

#pragma once

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include "onnx/common/stl_backports.h"
#include "core/common/common.h"
#include "core/common/const_pointer_container.h"
#include "core/session/onnxruntime_c_api.h"
#include "provider_interfaces.h"

namespace ONNX_NAMESPACE {

// These are exact duplicates of the real protobuf types, defined here since we can't include the protobuf headers
enum AttributeProto_AttributeType : int {
  AttributeProto_AttributeType_UNDEFINED = 0,
  AttributeProto_AttributeType_FLOAT = 1,
  AttributeProto_AttributeType_INT = 2,
  AttributeProto_AttributeType_STRING = 3,
  AttributeProto_AttributeType_TENSOR = 4,
  AttributeProto_AttributeType_GRAPH = 5,
  AttributeProto_AttributeType_SPARSE_TENSOR = 11,
  AttributeProto_AttributeType_FLOATS = 6,
  AttributeProto_AttributeType_INTS = 7,
  AttributeProto_AttributeType_STRINGS = 8,
  AttributeProto_AttributeType_TENSORS = 9,
  AttributeProto_AttributeType_GRAPHS = 10,
  AttributeProto_AttributeType_SPARSE_TENSORS = 12
};

enum Version : int {
  _START_VERSION = 0,
  IR_VERSION_2017_10_10 = 1,
  IR_VERSION_2017_10_30 = 2,
  IR_VERSION_2017_11_3 = 3,
  IR_VERSION_2019_1_22 = 4,
  IR_VERSION_2019_3_18 = 5,
  IR_VERSION_2019_9_19 = 6,
  IR_VERSION = 7
};

enum OperatorStatus : int {
  EXPERIMENTAL = 0,
  STABLE = 1
};

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

void SetProviderHost(ProviderHost& host);

// The function passed in will be run on provider DLL unload. This is used to free thread_local variables that are in threads we don't own
// Since these are not destroyed when the DLL unloads we have to do it manually. Search for usage for an example.
void RunOnUnload(std::function<void()> function);

// A pointer stored in here will be deleted when the DLL gets unloaded, this is really only useful for thread_locals which don't get cleaned up properly otherwise
template <typename T>
struct DeleteOnUnloadPtr {
  DeleteOnUnloadPtr(T* p) : p_(p) {
    RunOnUnload([p = p_]() {
      delete p;
    });
  }

  operator T*() {
    return p_;
  }

 private:
  T* p_;
};

constexpr const char* kOnnxDomain = "";
constexpr const char* kMSDomain = "com.microsoft";
constexpr const char* kDnnlExecutionProvider = "DnnlExecutionProvider";
constexpr const char* kTensorrtExecutionProvider = "TensorrtExecutionProvider";

enum CUDAStreamType : int {
  kCudaStreamDefault = 0,
  kCudaStreamCopyIn,
  kCudaStreamCopyOut,
  kTotalCudaStreams,
};

class DataTypeImpl {
 public:
  virtual ~DataTypeImpl() = default;

  template <typename T>
  static MLDataType GetType();
  template <typename elemT>
  static MLDataType GetTensorType();

  static const std::vector<MLDataType>& AllFixedSizeTensorTypes();
};

class TensorShape : private std::vector<int64_t> {
 public:
  TensorShape() = default;

  TensorShape(const TensorShape& /*other*/) = default;
  TensorShape& operator=(const TensorShape& /*other*/) = default;

  TensorShape(TensorShape&& /*other*/) = default;
  TensorShape& operator=(TensorShape&& /*other*/) = default;

  TensorShape(const std::vector<int64_t>& dims) : std::vector<int64_t>{dims} {}
  TensorShape(std::vector<int64_t>&& dims) : std::vector<int64_t>{dims} {}
  TensorShape(const std::initializer_list<int64_t>& dims) : std::vector<int64_t>{dims} {}

  TensorShape(const int64_t* dimension_sizes, size_t dimension_count);
  TensorShape(const std::vector<int64_t>& dims, size_t start, size_t end);

  using std::vector<int64_t>::operator[];

  size_t NumDimensions() const noexcept {
    return size();
  }

  const std::vector<int64_t>& GetDims() const { return *this; }

  int64_t Size() const;

  /**
     Return a new TensorShape of the dimensions from dimstart to dimend.
  */
  TensorShape Slice(size_t dimstart, size_t dimend) const;

  /**
     Return a new TensorShape of the dimensions from dimstart to end.
  */
  TensorShape Slice(size_t dimstart) const;

  /**
     output dimensions nicely formatted
  */
  std::string ToString() const;

  /**
     Calculate size between start and end.
     Assumes start and end are between 0 and this->NumDimensions(), inclusive, and that
     start < end.
  */
  int64_t SizeHelper(size_t start, size_t end) const;
};

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

std::unique_ptr<Provider_IDeviceAllocator> Provider_CreateCPUAllocator(std::unique_ptr<Provider_OrtMemoryInfo> memory_info);
std::unique_ptr<Provider_IDeviceAllocator> Provider_CreateCUDAAllocator(int16_t device_id, const char* name);
std::unique_ptr<Provider_IDeviceAllocator> Provider_CreateCUDAPinnedAllocator(int16_t device_id, const char* name);
Provider_AllocatorPtr CreateAllocator(const Provider_DeviceAllocatorRegistrationInfo& info, int16_t device_id = 0, bool use_arena = true);

std::unique_ptr<Provider_IDataTransfer> Provider_CreateGPUDataTransfer();

std::string GetEnvironmentVar(const std::string& var_name);

class CPUIDInfo {
 public:
  static const CPUIDInfo& GetCPUIDInfo();

  bool HasAVX2() const;
  bool HasAVX512f() const;
};

namespace logging {

enum class Severity {
  kVERBOSE = 0,
  kINFO = 1,
  kWARNING = 2,
  kERROR = 3,
  kFATAL = 4
};

enum class DataType {
  SYSTEM = 0,  ///< System data.
  USER = 1     ///< Contains potentially sensitive user data.
};

struct Category {
  static const char* onnxruntime;  ///< General output
  static const char* System;       ///< Log output regarding interactions with the host system
  // TODO: What other high level categories are meaningful? Model? Optimizer? Execution?
};

constexpr const char* SEVERITY_PREFIX = "VIWEF";

class Logger {
 public:
  bool OutputIsEnabled(Severity severity, DataType data_type) const noexcept;
};

class LoggingManager {
 public:
  static const Logger& DefaultLogger();
};

class Capture {
 public:
  Capture(const Logger& logger, logging::Severity severity, const char* category,
          logging::DataType dataType, const CodeLocation& location);

  std::ostream& Stream() noexcept;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Capture);
};

}  // namespace logging

enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

// TODO(RyanHill): Move this to a host function
inline AutoPadType StringToAutoPadType(const std::string& str) {
  if (str.empty()) {
    return AutoPadType::NOTSET;
  }
  if (str == "NOTSET") {  // in onnx spec, default value is "NOTSET"
    return AutoPadType::NOTSET;
  }
  if (str == "VALID") {
    return AutoPadType::VALID;
  }
  if (str == "SAME_UPPER") {
    return AutoPadType::SAME_UPPER;
  }
  if (str == "SAME_LOWER") {
    return AutoPadType::SAME_LOWER;
  }
  ORT_ENFORCE(false, "Unknown AutoPadType String");
}

namespace math {

// Rounds a up to the next highest multiple of b, which is power-of-2. User must be careful
// to ensure that there is no overflow or underflow in the calculation
// of divUp.
template <typename T, T b>
constexpr T roundUpPow2(T a) {
  return (a + (b - 1)) & (~(b - 1));
}
}  // namespace math

}  // namespace onnxruntime

#define ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name) \
  provider##_##name##_##domain##_ver##ver

#define ONNX_OPERATOR_KERNEL_EX(name, domain, ver, provider, builder, ...)                                                                       \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name);                                                                            \
  template <>                                                                                                                                    \
  Provider_KernelCreateInfo                                                                                                                      \
  BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)>() {                                                        \
    return Provider_KernelCreateInfo(                                                                                                            \
        builder.SetName(#name)                                                                                                                   \
            .SetDomain(domain)                                                                                                                   \
            .SinceVersion(ver)                                                                                                                   \
            .Provider(provider)                                                                                                                  \
            .Build(),                                                                                                                            \
        static_cast<Provider_KernelCreatePtrFn>([](const Provider_OpKernelInfo& info) -> Provider_OpKernel* { return new __VA_ARGS__(info); })); \
  }

#define CREATE_MESSAGE(logger, severity, category, datatype) \
  ::onnxruntime::logging::Capture(logger, ::onnxruntime::logging::Severity::k##severity, category, datatype, ORT_WHERE)

// iostream style logging. Capture log info in Message, and push to the logger in ~Message.
#define LOGS_CATEGORY(logger, severity, category)                                                                        \
  if ((logger).OutputIsEnabled(::onnxruntime::logging::Severity::k##severity, ::onnxruntime::logging::DataType::SYSTEM)) \
  CREATE_MESSAGE(logger, severity, category, ::onnxruntime::logging::DataType::SYSTEM).Stream()

#define LOGS_DEFAULT_CATEGORY(severity, category) \
  LOGS_CATEGORY(::onnxruntime::logging::LoggingManager::DefaultLogger(), severity, category)

#define LOGS_DEFAULT(severity) \
  LOGS_DEFAULT_CATEGORY(severity, ::onnxruntime::logging::Category::onnxruntime)
