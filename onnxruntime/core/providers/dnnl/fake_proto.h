// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#pragma once
#define NO_PROTOBUF
#define ONNX_NAMESPACE onnx

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include "core/common/common.h"
#include "core/common/const_pointer_container.h"
#include "core/session/onnxruntime_c_api.h"
#include "gsl/gsl-lite.hpp"
//#include "core/framework/op_node_proto_helper.h"
//#include "core/graph/graph.h"
//#include "core/providers/providers.h"
#include "../shared_library/bridge.h"

#if 0
namespace google {
namespace protobuf {
template <typename T>
struct RepeatedPtrField {};
}  // namespace protobuf
}  // namespace google
#endif

namespace onnx {
using DataType = const std::string*;
using OperatorSetVersion = int;

enum AttributeProto_AttributeType {
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

enum Version {
  _START_VERSION = 0,
  IR_VERSION_2017_10_10 = 1,
  IR_VERSION_2017_10_30 = 2,
  IR_VERSION_2017_11_3 = 3,
  IR_VERSION_2019_1_22 = 4,
  IR_VERSION_2019_3_18 = 5,
  IR_VERSION = 6
};

enum OperatorStatus {
  EXPERIMENTAL = 0,
  STABLE = 1
};

class ValueInfoProto {};
class TensorProto {
};

class TypeProto;
class OpSchema {
 public:
  OperatorSetVersion SinceVersion() const;
};

class AttributeProto {
 public:
  ::onnx::AttributeProto_AttributeType type() const;
  int ints_size() const;
  int64_t ints(int i) const;
  int64_t i() const;
  float f() const;
  void set_s(const ::std::string& value);
  const ::std::string& s() const;
  void set_name(const ::std::string& value);
  void set_type(::onnx::AttributeProto_AttributeType value);
  ::onnx::TensorProto* add_tensors();
};

class GraphProto {
};

class SparseTensorProto {
};

class NodeProto {};

class FunctionProto {};
}  // namespace onnx

namespace onnxruntime {

constexpr const char* kOnnxDomain = "";
constexpr const char* kDnnlExecutionProvider = "DnnlExecutionProvider";

class Graph;

/**
 * \brief Base class for MLDataType
 *
 */
class DataTypeImpl {
 public:
  virtual ~DataTypeImpl() = default;

  template <typename T>
  static MLDataType GetType();
  template <typename elemT>
  static MLDataType GetTensorType();
};

struct IExecutionProviderFactory;

struct IndexedSubGraph {
  struct MetaDef {
    std::string name;    ///< Name of customized SubGraph/FunctionProto
    std::string domain;  ///< Domain of customized SubGraph/FunctionProto
    int since_version;   ///< Since version of customized SubGraph/FunctionProto.

    ONNX_NAMESPACE::OperatorStatus status;  ///< Status of customized SubGraph/FunctionProto.

    std::vector<std::string> inputs;   ///< Inputs of customized SubGraph/FunctionProto.
    std::vector<std::string> outputs;  ///< Outputs of customized SubGraph/FunctionProto.
    Prov_NodeAttributes attributes;    ///< Attributes of customized SubGraph/FunctionProto.

    std::string doc_string;  ///< Doc string of customized SubGraph/FunctionProto.
  };

  /** Nodes covered by this subgraph. The NodeIndex values are from the parent Graph.*/
  std::vector<onnxruntime::NodeIndex> nodes;

  void SetMetaDef(std::unique_ptr<MetaDef>& meta_def_);
};

class GraphNodes;

class TensorShape {
 public:
  TensorShape();

#if 0
  TensorShape(const TensorShape& /*other*/) = default;
  TensorShape& operator=(const TensorShape& /*other*/) = default;

  TensorShape(TensorShape&& /*other*/) = default;
  TensorShape& operator=(TensorShape&& /*other*/) = default;
#endif
  TensorShape(const std::vector<int64_t>& dims);
  TensorShape(std::vector<int64_t>&& dims);

  TensorShape(const std::initializer_list<int64_t>& dims);

  TensorShape(const int64_t* dimension_sizes, size_t dimension_count);
#if 0
  TensorShape(const std::vector<int64_t>& dims, size_t start, size_t end);
#endif

  const int64_t& operator[](size_t idx) const;
  int64_t& operator[](size_t idx);

#if 0
  bool operator==(const TensorShape& other) const noexcept;
  bool operator!=(const TensorShape& other) const noexcept;
#endif
  size_t NumDimensions() const noexcept;

#if 0

  void CopyDims(int64_t* dims, size_t num_dims) const;
#endif

  const std::vector<int64_t>& GetDims() const;

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

#if 0
  /**
     Calculate size between start and end.
     Assumes start and end are between 0 and this->NumDimensions(), inclusive, and that
     start < end.
  */
  int64_t SizeHelper(size_t start, size_t end) const;

  /**
     empty shape or 1D shape (1) is regarded as scalar tensor
  */
  bool IsScalar() const;

  static const TensorShape& ReinterpretBaseType(const std::vector<int64_t>& dimensions);
#endif

  void* this_;
};

class Tensor final {
 public:
  Tensor() = default;

  const TensorShape& Shape() const noexcept;

  template <typename T>
  T* MutableData() {
    __debugbreak();
    return nullptr;
  }

  template <typename T>
  const T* Data() const {
    __debugbreak();
    return nullptr;
  }
};

class OpKernelInfo {
 public:
  template <typename T>
  Status GetAttr(const std::string& name, T* value) const {
    __debugbreak();
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
    __debugbreak();
    index;
    return nullptr;
  }
  template <typename T>
  T* Output(int index);
  Tensor* Output(int index, const TensorShape& shape);
};

constexpr const char* kMSDomain = "com.microsoft";
constexpr const char* kMklDnnExecutionProvider = "MKLDNNExecutionProvider";

using AllocateFunc = void* (*)(void*, size_t, size_t);
using DestroyFunc = void (*)(void*, void*);
using AllocatorHandle = void*;

typedef struct {
  //right now we only include allocation for host memory
  AllocateFunc allocate_func;
  DestroyFunc release_func;
  AllocatorHandle allocator_handle;
  const char* node_name;
} ComputeContext;

using FunctionState = void*;

// if we are export the fused function to dll, the function will still in the same binary as lotus
// use std function to give execution provider some chance to capture some state.
using CreateFunctionStateFunc = std::function<int(ComputeContext*, FunctionState*)>;
using ComputeFunc = std::function<Status(FunctionState, const OrtApi*, OrtKernelContext*)>;
using DestroyFunctionStateFunc = std::function<void(FunctionState)>;

struct NodeComputeInfo {
  CreateFunctionStateFunc create_state_func;
  ComputeFunc compute_func;
  DestroyFunctionStateFunc release_state_func;
};

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

std::unique_ptr<Prov_IDeviceAllocator> CreateCPUAllocator(std::unique_ptr<Prov_OrtMemoryInfo> memory_info);
Prov_AllocatorPtr CreateDummyArenaAllocator(Prov_AllocatorPtr resource_allocator);
Prov_AllocatorPtr CreateAllocator(Prov_DeviceAllocatorRegistrationInfo& info, int device_id = 0);

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
