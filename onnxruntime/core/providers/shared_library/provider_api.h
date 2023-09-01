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
#include "core/common/gsl.h"
#include <unordered_map>
#include <unordered_set>
#include <stddef.h>
#include "core/common/common.h"
#include "core/common/const_pointer_container.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/common/type_list.h"
#include "core/common/logging/severity.h"
#include "core/framework/allocator.h"
#include "core/framework/float8.h"
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
  TensorProto_DataType_BFLOAT16 = 16,
  TensorProto_DataType_FLOAT8E4M3FN = 17,
  TensorProto_DataType_FLOAT8E4M3FNUZ = 18,
  TensorProto_DataType_FLOAT8E5M2 = 19,
  TensorProto_DataType_FLOAT8E5M2FNUZ = 20
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
  IR_VERSION_2020_5_8 = 7,
  IR_VERSION_2021_7_31 = 8,
  IR_VERSION = 9
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
struct SparseTensorProto;
struct TensorProto;
struct TensorProtos;  // RepeatedPtrField
struct TensorShapeProto_Dimension;
struct TensorShapeProto_Dimensions;  // RepeatedPtrField
struct TensorShapeProto;
struct TypeProto_Optional;
struct TypeProto_Tensor;
struct TypeProto_SparseTensor;
struct TypeProto_Sequence;
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
enum class DataLayout;
struct Model;
struct Path;
struct Node;
struct NodeArg;
struct NodeAttributes;
class OpKernel;
struct OpKernelContext;
struct OpKernelInfo;
struct PrimitiveDataTypeBase;
struct Tensor;
struct SparseTensor;
class TensorSeq;
class SessionState;

class If;
class Loop;
class ConcatBase;
class Einsum;
class GatherBase;
class Size;
class SliceBase;
class SplitBase;
class TensorShape;
struct Prepare;
struct PrepareContext;
enum class Mode : int;
struct EinsumComputePreprocessor;
template <typename T>
struct EinsumTypedComputeProcessor;
struct SessionOptions;

namespace contrib {
class ATen;
class Group;
class PassThrough;
class YieldOp;
class TritonOp;
class AdamWOptimizerBase;
class SGDOptimizerV2Base;
class ShrunkenGatherCommon;
}  // namespace contrib

class UnsqueezeBase;
template <int OpSet>
class Scan;

class DataTypeImpl;
using MLDataType = const DataTypeImpl*;
// be used with class MLValue
using DeleteFunc = void (*)(void*);
using NodeArgInfo = ONNX_NAMESPACE::ValueInfoProto;

using NameMLValMap = std::unordered_map<std::string, OrtValue>;
}  // namespace onnxruntime

#include "core/platform/threadpool.h"
#include "core/providers/cpu/math/einsum_utils/einsum_compute_preprocessor.h"
#include "core/providers/cpu/cpu_provider_shared.h"
#include "core/framework/data_transfer.h"
#include "core/framework/execution_provider.h"
#include "provider_interfaces.h"
#include "provider_wrappedtypes.h"
#include "core/framework/op_kernel.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/providers/common.h"
#include "core/util/math.h"

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

  T& operator*() { return *p_; }
  const T& operator*() const { return *p_; }

  operator T*() {
    return p_;
  }

 private:
  T* p_;
};

constexpr const char* kOnnxDomain = "";
constexpr const char* kMSDomain = "com.microsoft";
constexpr const char* kPytorchAtenDomain = "org.pytorch.aten";
constexpr const char* kNGraphDomain = "com.intel.ai";
constexpr const char* kCudaExecutionProvider = "CUDAExecutionProvider";
constexpr const char* kCannExecutionProvider = "CANNExecutionProvider";
constexpr const char* kDnnlExecutionProvider = "DnnlExecutionProvider";
constexpr const char* kOpenVINOExecutionProvider = "OpenVINOExecutionProvider";
constexpr const char* kRocmExecutionProvider = "ROCMExecutionProvider";
constexpr const char* kTensorrtExecutionProvider = "TensorrtExecutionProvider";
constexpr const char* kMIGraphXExecutionProvider = "MIGraphXExecutionProvider";
constexpr const char* kQnnExecutionProvider = "QNNExecutionProvider";
constexpr const char* kCpuExecutionProvider = "CPUExecutionProvider";
constexpr const char* kAzureExecutionProvider = "AzureExecutionProvider";

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)> >;

inline OrtStatus* CreateStatus(OrtErrorCode code, _In_ const char* msg) noexcept { return g_host->CreateStatus(code, msg); }

std::unique_ptr<IAllocator> CreateCPUAllocator(const OrtMemoryInfo& memory_info);
std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name);
std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(const char* name);

std::unique_ptr<IAllocator> CreateROCMAllocator(int16_t device_id, const char* name);
std::unique_ptr<IAllocator> CreateROCMPinnedAllocator(const char* name);

std::unique_ptr<IDataTransfer> CreateGPUDataTransfer();

std::unordered_set<NodeIndex> GetCpuPreferredNodes(const onnxruntime::GraphViewer& graph,
                                                   const IExecutionProvider::IKernelLookup& kernel_lookup,
                                                   gsl::span<const NodeIndex> tentative_nodes);

std::string GetEnvironmentVar(const std::string& var_name);

namespace profiling {

std::string demangle(const char* name);
std::string demangle(const std::string& name);

}  // namespace profiling

namespace logging {

unsigned int GetThreadId();
unsigned int GetProcessId();

struct Category {
  static const char* onnxruntime;  ///< General output
};

}  // namespace logging

namespace utils {

template <typename T>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<bool>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<std::string>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<float>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<double>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<MLFloat16>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<BFloat16>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<int8_t>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<uint8_t>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<int16_t>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<uint16_t>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<int32_t>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<uint32_t>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<int64_t>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<uint64_t>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64; }

#if !defined(DISABLE_FLOAT8_TYPES)
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Float8E4M3FN>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Float8E4M3FNUZ>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Float8E5M2>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2; }
template <>
constexpr ONNXTensorElementDataType GetONNXTensorElementDataType<Float8E5M2FNUZ>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ; }
#endif
}  // namespace utils

// This is a replacement for Ort::InitApi() to be called before any other onnxruntime API calls.
// So the C API (and C++) becomes available when ORT_API_MANUAL_INIT is used.
void InitProviderOrtApi();

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
