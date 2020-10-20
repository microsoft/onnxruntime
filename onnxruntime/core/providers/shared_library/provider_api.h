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
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {
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

}  // namespace logging

enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

// onnx Protobuf types (all of these are actually just Provider_<type> -> ONNX_NAMESPACE::<type>)
struct Provider_AttributeProto;
struct Provider_GraphProto;
struct Provider_ModelProto;
struct Provider_NodeProto;
struct Provider_TensorProto;
struct Provider_TensorProtos;
struct Provider_TensorShapeProto_Dimension;
struct Provider_TensorShapeProto_Dimensions;
struct Provider_TensorShapeProto;
struct Provider_TypeProto_Tensor;
struct Provider_TypeProto;
struct Provider_ValueInfoProto;
struct Provider_ValueInfoProtos;

// OnnxRuntime Types (all of these are actually just Provider_<type> -> <type>)
struct CPUIDInfo;
namespace logging {
struct Logger;
struct Capture;
}  // namespace logging
struct Provider_ComputeCapability;
struct Provider_DataTransferManager;
struct Provider_IDataTransfer;
struct Provider_IndexedSubGraph;
struct Provider_IndexedSubGraph_MetaDef;
struct Provider_KernelDef;
struct Provider_KernelDefBuilder;
struct Provider_KernelRegistry;
struct Provider_Function;
struct Provider_Graph;
struct Provider_GraphViewer;
struct Provider_Model;
struct Provider_Node;
struct Provider_NodeArg;
struct Provider_NodeAttributes;
struct Provider_OpKernelContext;
struct Provider_OpKernelInfo;
struct Provider_Tensor;
}

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

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

std::unique_ptr<Provider_IAllocator> Provider_CreateCPUAllocator(const OrtMemoryInfo& memory_info);
std::unique_ptr<Provider_IAllocator> Provider_CreateCUDAAllocator(int16_t device_id, const char* name);
std::unique_ptr<Provider_IAllocator> Provider_CreateCUDAPinnedAllocator(int16_t device_id, const char* name);
Provider_AllocatorPtr CreateAllocator(const Provider_AllocatorCreationInfo& info);

std::unique_ptr<Provider_IDataTransfer> Provider_CreateGPUDataTransfer();

std::string GetEnvironmentVar(const std::string& var_name);

inline AutoPadType StringToAutoPadType(const std::string& str) { return g_host->StringToAutoPadType(str); }

namespace logging {

struct Category {
  static const char* onnxruntime;  ///< General output
};

constexpr const char* SEVERITY_PREFIX = "VIWEF";

}  // namespace logging

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
  ::onnxruntime::logging::Capture::Create(logger, ::onnxruntime::logging::Severity::k##severity, category, datatype, ORT_WHERE)

// iostream style logging. Capture log info in Message, and push to the logger in ~Message.
#define LOGS_CATEGORY(logger, severity, category)                                                                        \
  if ((logger).OutputIsEnabled(::onnxruntime::logging::Severity::k##severity, ::onnxruntime::logging::DataType::SYSTEM)) \
  CREATE_MESSAGE(logger, severity, category, ::onnxruntime::logging::DataType::SYSTEM)->Stream()

#define LOGS_DEFAULT_CATEGORY(severity, category) \
  LOGS_CATEGORY(::onnxruntime::logging::LoggingManager::DefaultLogger(), severity, category)

#define LOGS_DEFAULT(severity) \
  LOGS_DEFAULT_CATEGORY(severity, ::onnxruntime::logging::Category::onnxruntime)
