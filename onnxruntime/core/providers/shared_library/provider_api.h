// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Provider implementations include this file

// NOTE: This is still in development so there are many parts that will be fixed in the future. This is just the first version of
//       switching providers to be runnable as shared libraries. The interfaces will become more tightly integrated into the core code.

#pragma once
#define SHARED_PROVIDER 1

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include "onnx/common/stl_backports.h"
#include "core/common/common.h"
#include "core/common/const_pointer_container.h"
#include "core/common/type_list.h"
#include "core/common/logging/severity.h"
#include "core/framework/allocator.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/float16.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/providers.h"
#include "core/common/path_string.h"

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

enum TensorProto_DataType : int {
  TensorProto_DataType_UNDEFINED = 0,
  TensorProto_DataType_FLOAT = 1,
  TensorProto_DataType_UINT8 = 2,
  TensorProto_DataType_INT8 = 3,
  TensorProto_DataType_UINT16 = 4,
  TensorProto_DataType_INT16 = 5,
  TensorProto_DataType_INT32 = 6,
  TensorProto_DataType_INT64 = 7,
  TensorProto_DataType_STRING = 8,
  TensorProto_DataType_BOOL = 9,
  TensorProto_DataType_FLOAT16 = 10,
  TensorProto_DataType_DOUBLE = 11,
  TensorProto_DataType_UINT32 = 12,
  TensorProto_DataType_UINT64 = 13,
  TensorProto_DataType_COMPLEX64 = 14,
  TensorProto_DataType_COMPLEX128 = 15,
  TensorProto_DataType_BFLOAT16 = 16
};

enum TensorProto_DataLocation : int {
  TensorProto_DataLocation_DEFAULT = 0,
  TensorProto_DataLocation_EXTERNAL = 1
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

// onnx Protobuf types (All of these are direct mappings to the onnx types except for the Repeated*Field ones which map to a Repeated*Field type)
struct int64s;  // RepeatedField
struct AttributeProto;
struct GraphProto;
struct ModelProto;
struct NodeProto;
struct TensorProto;
struct TensorProtos;  // RepeatedPtrField
struct TensorShapeProto_Dimension;
struct TensorShapeProto_Dimensions;  // RepeatedPtrField
struct TensorShapeProto;
struct TypeProto_Tensor;
struct TypeProto;
struct ValueInfoProto;
struct ValueInfoProtos;  // RepeatedPtrField
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace logging {

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

// OnnxRuntime Types (these are the internal types)
struct CPUIDInfo;
namespace logging {
struct Logger;
struct Capture;
}  // namespace logging
struct ComputeCapability;
struct DataTransferManager;
struct IndexedSubGraph;
struct IndexedSubGraph_MetaDef;
struct KernelCreateInfo;
struct KernelDef;
struct KernelDefBuilder;
struct KernelRegistry;
struct Function;
struct Graph;
struct GraphViewer;
struct Model;
struct Path;
struct Node;
struct NodeArg;
struct NodeAttributes;
struct OpKernelContext;
struct OpKernelInfo;
struct PrimitiveDataTypeBase;
struct Tensor;

class DataTypeImpl;
using MLDataType = const DataTypeImpl*;
using NodeArgInfo = ONNX_NAMESPACE::ValueInfoProto;
}  // namespace onnxruntime

#include "core/framework/data_transfer.h"
#include "core/framework/execution_provider.h"
#include "provider_interfaces.h"
#include "core/framework/op_kernel.h"
#include "core/framework/data_types_internal.h"

namespace onnxruntime {

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
constexpr const char* kNGraphDomain = "com.intel.ai";
constexpr const char* kDnnlExecutionProvider = "DnnlExecutionProvider";
constexpr const char* kOpenVINOExecutionProvider = "OpenVINOExecutionProvider";
constexpr const char* kTensorrtExecutionProvider = "TensorrtExecutionProvider";

enum CUDAStreamType : int {
  kCudaStreamDefault = 0,
  kCudaStreamCopyIn,
  kCudaStreamCopyOut,
  kTotalCudaStreams,
};

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

std::unique_ptr<IAllocator> CreateCPUAllocator(const OrtMemoryInfo& memory_info);
std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name);
std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name);

std::unique_ptr<IDataTransfer> CreateGPUDataTransfer(void* stream);

std::string GetEnvironmentVar(const std::string& var_name);

inline AutoPadType StringToAutoPadType(const std::string& str) { return g_host->StringToAutoPadType(str); }

void AllocatorManager__InsertAllocator(AllocatorManager* p, AllocatorPtr allocator);
AllocatorPtr AllocatorManager__GetAllocator(AllocatorManager* p, int id, OrtMemType mem_type);

namespace logging {

struct Category {
  static const char* onnxruntime;  ///< General output
};

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
