// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Provider authors include this file

#pragma once

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include "core/common/common.h"
#include "core/common/const_pointer_container.h"
#include "core/session/onnxruntime_c_api.h"
#include "provider_interfaces.h"

namespace ONNX_NAMESPACE {

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

enum OperatorStatus : int {
  EXPERIMENTAL = 0,
  STABLE = 1
};

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {

struct RunOnUnload {
  RunOnUnload(std::function<void()> run);
};

constexpr const char* kOnnxDomain = "";
constexpr const char* kDnnlExecutionProvider = "DnnlExecutionProvider";

class DataTypeImpl {
 public:
  virtual ~DataTypeImpl() = default;

  template <typename T>
  static MLDataType GetType();
  template <typename elemT>
  static MLDataType GetTensorType();
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

#if 0
  bool operator==(const TensorShape& other) const noexcept;
  bool operator!=(const TensorShape& other) const noexcept;
#endif
  size_t NumDimensions() const noexcept {
    return size();
  }

#if 0

  void CopyDims(int64_t* dims, size_t num_dims) const;
#endif

  const std::vector<int64_t>& GetDims() const { return *this; }

  int64_t Size() const;

#if 0

  /**
     Return the total number of elements up to the specified dimension.
     If the dimension interval is empty (dimension == 0), return 1.
     @param dimension Return size up to this dimension. Value must be between 0 and this->NumDimensions(), inclusive.
  */
  int64_t SizeToDimension(size_t dimension) const;

  /**
     Return the total number of elements from the specified dimension to the end of the tensor shape.
     If the dimension interval is empty (dimension == this->NumDimensions()), return 1.
     @param dimension Return size from this dimension to the end. Value must be between 0 and this->NumDimensions(),
                      inclusive.
  */
  int64_t SizeFromDimension(size_t dimension) const;
#endif

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
#if 0

  /**
     empty shape or 1D shape (1) is regarded as scalar tensor
  */
  bool IsScalar() const;

  static const TensorShape& ReinterpretBaseType(const std::vector<int64_t>& dimensions);
#endif
};

class Tensor final {
 public:
  Tensor() = default;

  const TensorShape& Shape() const noexcept;

  template <typename T>
  T* MutableData() {
    PROVIDER_NOT_IMPLEMENTED
    return nullptr;
  }

  template <typename T>
  const T* Data() const {
    PROVIDER_NOT_IMPLEMENTED
    return nullptr;
  }
};

class OpKernelInfo {
 public:
  template <typename T>
  Status GetAttr(const std::string& name, T* value) const {
    PROVIDER_NOT_IMPLEMENTED
    name;
    value;
    return Status::OK();
  }
};

class OpKernel {
 public:
  explicit OpKernel(const OpKernelInfo& info);
};

class OpKernelContext {
 public:
  template <typename T>
  const T* Input(int index) const {
    PROVIDER_NOT_IMPLEMENTED
    index;
    return nullptr;
  }
  template <typename T>
  T* Output(int index);
  Tensor* Output(int index, const TensorShape& shape);
};

constexpr const char* kMSDomain = "com.microsoft";
constexpr const char* kMklDnnExecutionProvider = "MKLDNNExecutionProvider";

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

std::unique_ptr<Prov_IDeviceAllocator> CreateCPUAllocator(std::unique_ptr<Prov_OrtMemoryInfo> memory_info);
Prov_AllocatorPtr CreateDummyArenaAllocator(Prov_AllocatorPtr resource_allocator);
Prov_AllocatorPtr CreateAllocator(Prov_DeviceAllocatorRegistrationInfo& info, int16_t device_id = 0);

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
};
}  // namespace logging

enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

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

#define ONNX_OPERATOR_KERNEL_EX(name, domain, ver, provider, builder, ...)                                                 \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name);                                                      \
  template <>                                                                                                              \
  Prov_KernelCreateInfo                                                                                                    \
  BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)>() {                                  \
    return Prov_KernelCreateInfo(                                                                                          \
        builder.SetName(#name)                                                                                             \
            .SetDomain(domain)                                                                                             \
            .SinceVersion(ver)                                                                                             \
            .Provider(provider)                                                                                            \
            .Build(),                                                                                                      \
        static_cast<Prov_KernelCreatePtrFn>([](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); })); \
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
