// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This is the Onnxruntime side of the bridge to allow providers to be built as a DLL
// It implements onnxruntime::ProviderHost

#include "core/common/inlined_containers.h"
#include "core/framework/allocator_utils.h"
#include "core/framework/compute_capability.h"
#include "core/framework/data_types.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/provider_shutdown.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/provider_options.h"
#include "core/framework/fallback_cpu_capability.h"
#include "core/framework/random_generator.h"
#include "core/graph/model.h"
#include "core/platform/env.h"
#include "core/providers/common.h"
#include "core/session/inference_session.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"
#include "core/session/provider_bridge_ort.h"
#include "core/util/math.h"
#include "core/framework/sparse_utils.h"
#include "core/graph/graph_proto_serializer.h"
#include "core/framework/murmurhash3.h"

#include "core/session/onnxruntime_c_api.h"
#include "core/common/string_helper.h"

#ifdef ENABLE_TRAINING
#ifdef ENABLE_TRAINING_TORCH_INTEROP
#include "orttraining/training_ops/cpu/torch/torch_custom_function_kernel_base.h"
#include "orttraining/core/framework/torch/refcount_tracker.h"
#endif
#endif
#ifdef ENABLE_NVTX_PROFILE
#include "core/providers/cuda/nvtx_profile.h"
#endif
#if defined(ORT_USE_NCCL) && defined(ENABLE_TRAINING)
#include "orttraining/training_ops/cuda/communication/nccl_service.h"
#include "orttraining/core/framework/distributed_run_context.h"
#endif

#if defined(USE_ROCM) && defined(ORT_USE_NCCL) && defined(ENABLE_TRAINING)
#include "orttraining/training_ops/rocm/communication/nccl_service.h"
#include "orttraining/core/framework/distributed_run_context.h"
#endif

namespace ONNX_NAMESPACE {
// We use these names in the provider API because we don't have the protobuf definitions of the RepeatedField* types
using int64s = google::protobuf::RepeatedField<int64_t>;
using TensorProtos = google::protobuf::RepeatedPtrField<TensorProto>;
using TensorShapeProto_Dimensions = google::protobuf::RepeatedPtrField<TensorShapeProto_Dimension>;
using ValueInfoProtos = google::protobuf::RepeatedPtrField<ValueInfoProto>;
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
using IndexedSubGraph_MetaDef = IndexedSubGraph::MetaDef;
}  // namespace onnxruntime

#include "core/common/cpuid_info.h"
#include "core/common/logging/logging.h"
#include "core/providers/shared_library/provider_interfaces.h"

#include "core/providers/cuda/cuda_provider_factory_creator.h"
#include "core/providers/cann/cann_provider_factory_creator.h"
#include "core/providers/rocm/rocm_provider_factory_creator.h"
#include "core/providers/dnnl/dnnl_provider_factory_creator.h"
#include "core/providers/migraphx/migraphx_provider_factory_creator.h"
#include "core/providers/openvino/openvino_provider_factory_creator.h"
#include "core/providers/tensorrt/tensorrt_provider_factory_creator.h"

#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cann/cann_provider_factory.h"
#include "core/providers/rocm/rocm_provider_factory.h"
#include "core/providers/dnnl/dnnl_provider_factory.h"
#include "core/providers/migraphx/migraphx_provider_factory.h"
#include "core/providers/openvino/openvino_provider_factory.h"
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/cann/cann_provider_options.h"
#include "core/providers/dnnl/dnnl_provider_options.h"

// The filename extension for a shared library is different per platform
#ifdef _WIN32
#define LIBRARY_PREFIX
#define LIBRARY_EXTENSION ORT_TSTR(".dll")
#elif defined(__APPLE__)
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".dylib"
#else
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".so"
#endif

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

namespace onnxruntime {

ProviderInfo_CUDA* TryGetProviderInfo_CUDA();
ProviderInfo_CUDA& GetProviderInfo_CUDA();
ProviderInfo_TensorRT* TryGetProviderInfo_TensorRT();
ProviderInfo_TensorRT& GetProviderInfo_TensorRT();
ProviderInfo_CANN* TryGetProviderInfo_CANN();
ProviderInfo_CANN& GetProviderInfo_CANN();
ProviderInfo_Dnnl* TryGetProviderInfo_Dnnl();
ProviderInfo_Dnnl& GetProviderInfo_Dnnl();
ProviderInfo_ROCM* TryGetProviderInfo_ROCM();
ProviderInfo_ROCM& GetProviderInfo_ROCM();
ProviderHostCPU& GetProviderHostCPU();
struct TensorShapeProto_Dimension_Iterator_Impl : TensorShapeProto_Dimension_Iterator {
  TensorShapeProto_Dimension_Iterator_Impl(google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorShapeProto_Dimension>&& v) : v_{std::move(v)} {}

  bool operator!=(const TensorShapeProto_Dimension_Iterator& p) const override { return v_ != static_cast<const TensorShapeProto_Dimension_Iterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const ONNX_NAMESPACE::TensorShapeProto_Dimension& operator*() override { return *v_; }

  google::protobuf::internal::RepeatedPtrIterator<const onnx::TensorShapeProto_Dimension> v_;
};

struct NodeAttributes_Iterator_Impl : NodeAttributes_Iterator {
  NodeAttributes_Iterator_Impl(NodeAttributes::const_iterator&& v) : v_{std::move(v)} {}

  bool operator!=(const NodeAttributes_Iterator& p) const override { return v_ != static_cast<const NodeAttributes_Iterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const std::pair<const std::string, ONNX_NAMESPACE::AttributeProto>& operator*() const override { return v_.operator*(); }

  const std::string& first() const override { return v_->first; }
  const ONNX_NAMESPACE::AttributeProto& second() const override { return v_->second; }

  NodeAttributes::const_iterator v_;
};

struct Node__NodeIterator_Impl : Node__NodeIterator {
  Node__NodeIterator_Impl(Node::NodeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Node__NodeIterator& p) const override { return v_ != static_cast<const Node__NodeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Node& operator*() override { return *v_; }

  Node::NodeConstIterator v_;
};

struct Node__EdgeIterator_Impl : Node__EdgeIterator {
  Node__EdgeIterator_Impl(Node::EdgeConstIterator&& v) : v_{std::move(v)} {}

  bool operator!=(const Node__EdgeIterator& p) const override { return v_ != static_cast<const Node__EdgeIterator_Impl*>(&p)->v_; }

  void operator++() override { v_.operator++(); }
  const Node& GetNode() const override { return v_->GetNode(); }
  int GetSrcArgIndex() const override { return v_->GetSrcArgIndex(); }
  int GetDstArgIndex() const override { return v_->GetDstArgIndex(); }

  Node::EdgeConstIterator v_;
};

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
common::Status LoadDynamicLibraryFromProvider(onnxruntime::PathString library_name) {
  const auto& platform_env = onnxruntime::Env::Default();
  void* library_handle = nullptr;

  ORT_RETURN_IF_ERROR(platform_env.LoadDynamicLibrary(library_name, false, &library_handle));
  if (!library_handle) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load dynamic library ",
                           onnxruntime::PathToUTF8String(library_name));
  }

  return onnxruntime::Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26436)
#pragma warning(disable : 26409)
#endif
// wrapped = The internal object is exposed as an opaque pointer, so we wrap it in a class that forwards every call to the real calls. No members are ever directly accessed
// direct = Same implementation is used for shared providers & core code, but some of the methods need to be routed through here to make the linker happy
struct ProviderHostImpl : ProviderHost {
  const OrtApiBase* OrtGetApiBase() override { return ::OrtGetApiBase(); }

  void* HeapAllocate(size_t size) override { return new uint8_t[size]; }
  void HeapFree(void* p) override { delete[] reinterpret_cast<uint8_t*>(p); }

  logging::Logger* LoggingManager_GetDefaultLogger() override {
    return const_cast<logging::Logger*>(&logging::LoggingManager::DefaultLogger());
  }

  void LogRuntimeError(uint32_t session_id, const common::Status& status, const char* file, const char* function, uint32_t line) override {
    return ::onnxruntime::LogRuntimeError(session_id, status, file, function, line);
  }

  std::vector<std::string> GetStackTrace() override { return onnxruntime::GetStackTrace(); }

  OrtStatus* CreateStatus(OrtErrorCode code, _In_ const char* msg) noexcept override { return OrtApis::CreateStatus(code, msg); }

  AllocatorPtr CreateAllocator(const AllocatorCreationInfo& info) override { return onnxruntime::CreateAllocator(info); }
  std::unique_ptr<IAllocator> CreateCPUAllocator(const OrtMemoryInfo& memory_info) override { return std::make_unique<CPUAllocator>(memory_info); };

  void* CPUAllocator__Alloc(CPUAllocator* p, size_t size) override { return p->CPUAllocator::Alloc(size); }
  void CPUAllocator__Free(CPUAllocator* p, void* allocation) override { return p->CPUAllocator::Free(allocation); }

#ifdef USE_CUDA
  std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) override { return GetProviderInfo_CUDA().CreateCUDAAllocator(device_id, name); }
  std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(const char* name) override { return GetProviderInfo_CUDA().CreateCUDAPinnedAllocator(name); }
  std::unique_ptr<IDataTransfer> CreateGPUDataTransfer() override { return GetProviderInfo_CUDA().CreateGPUDataTransfer(); }

  void cuda__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) override { return GetProviderInfo_CUDA().cuda__Impl_Cast(stream, input_data, output_data, count); }
  void cuda__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) override { return GetProviderInfo_CUDA().cuda__Impl_Cast(stream, input_data, output_data, count); }

  void cuda__Impl_Cast(void* stream, const double* input_data, float* output_data, size_t count) override { return GetProviderInfo_CUDA().cuda__Impl_Cast(stream, input_data, output_data, count); }
  void cuda__Impl_Cast(void* stream, const float* input_data, double* output_data, size_t count) override { return GetProviderInfo_CUDA().cuda__Impl_Cast(stream, input_data, output_data, count); }

  Status CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) override { return GetProviderInfo_CUDA().CudaCall_false(retCode, exprString, libName, successCode, msg, file, line); }
  void CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) override { GetProviderInfo_CUDA().CudaCall_true(retCode, exprString, libName, successCode, msg, file, line); }
#endif

#ifdef USE_ROCM
  std::unique_ptr<IAllocator> CreateROCMAllocator(int16_t device_id, const char* name) override { return GetProviderInfo_ROCM().CreateROCMAllocator(device_id, name); }
  std::unique_ptr<IAllocator> CreateROCMPinnedAllocator(const char* name) override { return GetProviderInfo_ROCM().CreateROCMPinnedAllocator(name); }
  std::unique_ptr<IDataTransfer> CreateGPUDataTransfer() override { return GetProviderInfo_ROCM().CreateGPUDataTransfer(); }

  void rocm__Impl_Cast(void* stream, const int64_t* input_data, int32_t* output_data, size_t count) override { return GetProviderInfo_ROCM().rocm__Impl_Cast(stream, input_data, output_data, count); }
  void rocm__Impl_Cast(void* stream, const int32_t* input_data, int64_t* output_data, size_t count) override { return GetProviderInfo_ROCM().rocm__Impl_Cast(stream, input_data, output_data, count); }

  void rocm__Impl_Cast(void* stream, const double* input_data, float* output_data, size_t count) override { return GetProviderInfo_ROCM().rocm__Impl_Cast(stream, input_data, output_data, count); }
  void rocm__Impl_Cast(void* stream, const float* input_data, double* output_data, size_t count) override { return GetProviderInfo_ROCM().rocm__Impl_Cast(stream, input_data, output_data, count); }

  Status RocmCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) override { return GetProviderInfo_ROCM().RocmCall_false(retCode, exprString, libName, successCode, msg, file, line); }
  void RocmCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) override { GetProviderInfo_ROCM().RocmCall_true(retCode, exprString, libName, successCode, msg, file, line); }
#endif

  std::string GetEnvironmentVar(const std::string& var_name) override { return Env::Default().GetEnvironmentVar(var_name); }

  unsigned int GetThreadId() override { return onnxruntime::logging::GetThreadId(); }
  unsigned int GetProcessId() override { return onnxruntime::logging::GetProcessId(); }

  std::string demangle(const char* name) override { return onnxruntime::profiling::demangle(name); }
  std::string demangle(const std::string& name) override { return onnxruntime::profiling::demangle(name); }

  std::unordered_set<NodeIndex> GetCpuPreferredNodes(const onnxruntime::GraphViewer& graph,
                                                     const IExecutionProvider::IKernelLookup& kernel_lookup,
                                                     gsl::span<const NodeIndex> tentative_nodes) override {
    return onnxruntime::GetCpuPreferredNodes(graph, kernel_lookup, tentative_nodes);
  }

  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ bool* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ float* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ double* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ MLFloat16* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int8_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint8_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int16_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint16_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int32_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint32_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ int64_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, /*out*/ uint64_t* p_data, size_t expected_size) override { return utils::UnpackTensor(tensor, raw_data, raw_data_len, p_data, expected_size); }

  uint16_t math__floatToHalf(float f) override { return math::floatToHalf(f); }
  float math__halfToFloat(uint16_t h) override { return math::halfToFloat(h); }

  // sparse_utils
#if !defined(DISABLE_SPARSE_TENSORS)
#if !defined(ORT_MINIMAL_BUILD)
  Status sparse_utils__DenseTensorToSparseCsr(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                                              const AllocatorPtr& dst_allocator, SparseTensor& dst) override {
    return sparse_utils::DenseTensorToSparseCsr(data_manager, src, cpu_allocator, dst_allocator, dst);
  }

  Status sparse_utils__SparseCsrToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                                              const AllocatorPtr& dst_allocator, Tensor& dst) override {
    return sparse_utils::SparseCsrToDenseTensor(data_manager, src, cpu_allocator, dst_allocator, dst);
  }

  Status sparse_utils__SparseCooToDenseTensor(const DataTransferManager& data_manager, const SparseTensor& src, const AllocatorPtr& cpu_allocator,
                                              const AllocatorPtr& dst_allocator, Tensor& dst) override {
    return sparse_utils::SparseCooToDenseTensor(data_manager, src, cpu_allocator, dst_allocator, dst);
  }

#endif  // ORT_MINIMAL_BUILD
  Status sparse_utils__DenseTensorToSparseCoo(const DataTransferManager& data_manager, const Tensor& src, const AllocatorPtr& cpu_allocator,
                                              const AllocatorPtr& dst_allocator, bool linear_indexs, SparseTensor& dst) override {
    return sparse_utils::DenseTensorToSparseCoo(data_manager, src, cpu_allocator, dst_allocator, linear_indexs, dst);
  }

#endif  // !defined(DISABLE_SPARSE_TENSORS)

  // IAllocator (direct)
  bool IAllocator__CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) override { return IAllocator::CalcMemSizeForArrayWithAlignment(nmemb, size, alignment, out); }

  // IExecutionProvider (direct)
  std::vector<std::unique_ptr<ComputeCapability>> IExecutionProvider__GetCapability(
      const IExecutionProvider* p, const onnxruntime::GraphViewer& graph_viewer,
      const IExecutionProvider::IKernelLookup& kernel_lookup) override {
    return p->IExecutionProvider::GetCapability(graph_viewer, kernel_lookup);
  }

  common::Status IExecutionProvider__Compile(IExecutionProvider* p, const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    return p->IExecutionProvider::Compile(fused_nodes_and_graphs, node_compute_funcs);
  }

  int IExecutionProvider__GenerateMetaDefId(const IExecutionProvider* p, const onnxruntime::GraphViewer& graph_viewer, HashValue& model_hash) override {
    return p->IExecutionProvider::GenerateMetaDefId(graph_viewer, model_hash);
  }

  // Status (direct)
  std::string Status__ToString(const Status* p) override { return p->Status::ToString(); }

  // TensorShape (direct)
  void TensorShape__operator_assign(TensorShape* p, const TensorShape& other) override { p->TensorShape::operator=(other); }
  void TensorShape__operator_move_assign(TensorShape* p, TensorShape&& other) noexcept override { p->TensorShape::operator=(std::move(other)); }
  void TensorShape__Allocate(TensorShape* p, size_t size) override { p->TensorShape::Allocate(size); }
  int64_t TensorShape__SizeHelper(const TensorShape* p, size_t start, size_t end) override { return p->TensorShape::SizeHelper(start, end); }
  std::string TensorShape__ToString(const TensorShape* p) override { return p->TensorShape::ToString(); }
  int64_t TensorShape__SizeToDimension(const TensorShape* p, size_t dimension) override { return p->TensorShape::SizeToDimension(dimension); }
  int64_t TensorShape__SizeFromDimension(const TensorShape* p, size_t dimension) override { return p->TensorShape::SizeFromDimension(dimension); }
  std::ostream& operator_left_shift(std::ostream& out, const TensorShape& shape) override { return out << shape; }

  // CPUIDInfo (wrapped)
  const CPUIDInfo& CPUIDInfo__GetCPUIDInfo() override { return CPUIDInfo::GetCPUIDInfo(); }
  bool CPUIDInfo__HasAVX2(const CPUIDInfo* p) override { return p->HasAVX2(); }
  bool CPUIDInfo__HasAVX512f(const CPUIDInfo* p) override { return p->HasAVX512f(); }
  bool CPUIDInfo__HasAVX512_BF16(const CPUIDInfo* p) override { return p->HasAVX512_BF16(); }
  bool CPUIDInfo__HasAMX_BF16(const CPUIDInfo* p) override { return p->HasAMX_BF16(); }
  bool CPUIDInfo__HasAVX512Skylake(const CPUIDInfo* p) override { return p->HasAVX512Skylake(); }

  // logging::Logger (wrapped)
  bool logging__Logger__OutputIsEnabled(const logging::Logger* p, logging::Severity severity, logging::DataType data_type) override { return p->OutputIsEnabled(severity, data_type); }

  // logging::LoggingManager (wrapped)
  const logging::Logger& logging__LoggingManager__DefaultLogger() override { return logging::LoggingManager::DefaultLogger(); }

  // logging::Capture (wrapped)
  std::unique_ptr<logging::Capture> logging__Capture__construct(const logging::Logger& logger, logging::Severity severity, const char* category, logging::DataType dataType, const CodeLocation& location) override {
    return std::make_unique<logging::Capture>(logger, severity, category, dataType, location);
  }
  void logging__Capture__operator_delete(logging::Capture* p) noexcept override { delete p; }
  std::ostream& logging__Capture__Stream(logging::Capture* p) noexcept override { return p->Stream(); }

  // Utils::DataTypeUtils (wrapped)
  const std::string* Utils__DataTypeUtils__ToType(const ONNX_NAMESPACE::TypeProto& type_proto) override { return ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(type_proto); }

  // int64s (wrapped)
  int int64s__size(const ONNX_NAMESPACE::int64s* p) override { return p->size(); }
  const int64_t& int64s__Get(const ONNX_NAMESPACE::int64s* p, int index) override { return p->Get(index); }

#if !defined(DISABLE_OPTIONAL_TYPE)
  // TypeProto_Optional (wrapped)
  const ONNX_NAMESPACE::TypeProto& TypeProto_Optional__elem_type(const ONNX_NAMESPACE::TypeProto_Optional* p) override { return p->elem_type(); }
  ONNX_NAMESPACE::TypeProto* TypeProto_Optional__mutable_elem_type(ONNX_NAMESPACE::TypeProto_Optional* p) override { return p->mutable_elem_type(); }
#endif

  // TypeProto_Sequence (wrapped)
  const ONNX_NAMESPACE::TypeProto& TypeProto_Sequence__elem_type(const ONNX_NAMESPACE::TypeProto_Sequence* p) override { return p->elem_type(); }
  ONNX_NAMESPACE::TypeProto* TypeProto_Sequence__mutable_elem_type(ONNX_NAMESPACE::TypeProto_Sequence* p) override { return p->mutable_elem_type(); }

  // TypeProto_Tensor (wrapped)
  bool TypeProto_Tensor__has_shape(const ONNX_NAMESPACE::TypeProto_Tensor* p) override { return p->has_shape(); }
  const ONNX_NAMESPACE::TensorShapeProto& TypeProto_Tensor__shape(const ONNX_NAMESPACE::TypeProto_Tensor* p) override { return p->shape(); }
  ONNX_NAMESPACE::TensorShapeProto* TypeProto_Tensor__mutable_shape(ONNX_NAMESPACE::TypeProto_Tensor* p) override { return p->mutable_shape(); }
  int32_t TypeProto_Tensor__elem_type(const ONNX_NAMESPACE::TypeProto_Tensor* p) override { return p->elem_type(); }

  // TypeProto_SparseTensor (wrapped)
#if !defined(DISABLE_SPARSE_TENSORS)
  bool TypeProto_SparseTensor__has_shape(const ONNX_NAMESPACE::TypeProto_SparseTensor* p) override { return p->has_shape(); }
  const ONNX_NAMESPACE::TensorShapeProto& TypeProto_SparseTensor__shape(const ONNX_NAMESPACE::TypeProto_SparseTensor* p) override {
    return p->shape();
  }
  ONNX_NAMESPACE::TensorShapeProto* TypeProto_SparseTensor__mutable_shape(ONNX_NAMESPACE::TypeProto_SparseTensor* p) override {
    return p->mutable_shape();
  }
  int32_t TypeProto_SparseTensor__elem_type(const ONNX_NAMESPACE::TypeProto_SparseTensor* p) override {
    return p->elem_type();
  }
#endif

  // TypeProto (wrapped)
  std::unique_ptr<ONNX_NAMESPACE::TypeProto> TypeProto__construct() override { return std::make_unique<ONNX_NAMESPACE::TypeProto>(); }
  void TypeProto__CopyFrom(ONNX_NAMESPACE::TypeProto* p, const ONNX_NAMESPACE::TypeProto* other) override { p->CopyFrom(*other); }
  const ONNX_NAMESPACE::TypeProto_Tensor& TypeProto__tensor_type(const ONNX_NAMESPACE::TypeProto* p) override { return p->tensor_type(); }
  ONNX_NAMESPACE::TypeProto_Tensor* TypeProto__mutable_tensor_type(ONNX_NAMESPACE::TypeProto* p) override { return p->mutable_tensor_type(); }
  int TypeProto__value_case(const ONNX_NAMESPACE::TypeProto* p) override { return p->value_case(); }
#if !defined(DISABLE_SPARSE_TENSORS)
  const ONNX_NAMESPACE::TypeProto_SparseTensor& TypeProto__sparse_tensor_type(const ONNX_NAMESPACE::TypeProto* p) override {
    return p->sparse_tensor_type();
  }
  ONNX_NAMESPACE::TypeProto_SparseTensor* TypeProto__mutable_sparse_tensor_type(ONNX_NAMESPACE::TypeProto* p) override {
    return p->mutable_sparse_tensor_type();
  }
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
  const ONNX_NAMESPACE::TypeProto_Optional& TypeProto__optional_type(const ONNX_NAMESPACE::TypeProto* p) override { return p->optional_type(); }
  ONNX_NAMESPACE::TypeProto_Optional* TypeProto__mutable_optional_type(ONNX_NAMESPACE::TypeProto* p) override { return p->mutable_optional_type(); }
#endif

  const ONNX_NAMESPACE::TypeProto_Sequence& TypeProto__sequence_type(const ONNX_NAMESPACE::TypeProto* p) override { return p->sequence_type(); }
  ONNX_NAMESPACE::TypeProto_Sequence* TypeProto__mutable_sequence_type(ONNX_NAMESPACE::TypeProto* p) override { return p->mutable_sequence_type(); }

  // AttributeProto (wrapped)
  std::unique_ptr<ONNX_NAMESPACE::AttributeProto> AttributeProto__construct() override { return std::make_unique<ONNX_NAMESPACE::AttributeProto>(); }
  void AttributeProto__operator_delete(ONNX_NAMESPACE::AttributeProto* p) override { delete p; }
  void AttributeProto__operator_assign(ONNX_NAMESPACE::AttributeProto* p, const ONNX_NAMESPACE::AttributeProto& v) override { *p = v; }

  const std::string& AttributeProto__name(const ONNX_NAMESPACE::AttributeProto* p) const override { return p->name(); }
  ONNX_NAMESPACE::AttributeProto_AttributeType AttributeProto__type(const ONNX_NAMESPACE::AttributeProto* p) override { return p->type(); }
  int AttributeProto__ints_size(const ONNX_NAMESPACE::AttributeProto* p) override { return p->ints_size(); }
  int AttributeProto__floats_size(const ONNX_NAMESPACE::AttributeProto* p) override { return p->floats_size(); }
  int AttributeProto__strings_size(const ONNX_NAMESPACE::AttributeProto* p) override { return p->strings_size(); }
  int64_t AttributeProto__ints(const ONNX_NAMESPACE::AttributeProto* p, int i) override { return p->ints(i); }
  float AttributeProto__floats(const ONNX_NAMESPACE::AttributeProto* p, int i) override { return p->floats(i); }
  const std::string& AttributeProto__strings(const ONNX_NAMESPACE::AttributeProto* p, int i) override { return p->strings(i); }
  const ONNX_NAMESPACE::int64s& AttributeProto__ints(const ONNX_NAMESPACE::AttributeProto* p) override { return p->ints(); }
  int64_t AttributeProto__i(const ONNX_NAMESPACE::AttributeProto* p) override { return p->i(); }
  float AttributeProto__f(const ONNX_NAMESPACE::AttributeProto* p) override { return p->f(); }
  void AttributeProto__set_s(ONNX_NAMESPACE::AttributeProto* p, const ::std::string& value) override { return p->set_s(value); }
  const ::std::string& AttributeProto__s(const ONNX_NAMESPACE::AttributeProto* p) override { return p->s(); }
  void AttributeProto__set_name(ONNX_NAMESPACE::AttributeProto* p, const ::std::string& value) override { return p->set_name(value); }
  void AttributeProto__set_type(ONNX_NAMESPACE::AttributeProto* p, ONNX_NAMESPACE::AttributeProto_AttributeType value) override { return p->set_type(value); }
  ONNX_NAMESPACE::TensorProto* AttributeProto__add_tensors(ONNX_NAMESPACE::AttributeProto* p) override { return p->add_tensors(); }

  // GraphProto (wrapped)
  void GraphProto__operator_delete(ONNX_NAMESPACE::GraphProto* p) override { delete p; }

  const ONNX_NAMESPACE::ValueInfoProto& GraphProto__input(const ONNX_NAMESPACE::GraphProto* p, int index) override { return p->input(index); }
  ONNX_NAMESPACE::ValueInfoProto* GraphProto__mutable_input(ONNX_NAMESPACE::GraphProto* p, int index) override { return p->mutable_input(index); }
  ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_input(ONNX_NAMESPACE::GraphProto* p) override { return p->mutable_input(); }
  int GraphProto__input_size(const ONNX_NAMESPACE::GraphProto* p) override { return p->input_size(); }

  const ONNX_NAMESPACE::ValueInfoProtos& GraphProto__output(const ONNX_NAMESPACE::GraphProto* p) override { return p->output(); }
  const ONNX_NAMESPACE::ValueInfoProto& GraphProto__output(const ONNX_NAMESPACE::GraphProto* p, int index) override { return p->output(index); }
  ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_output(ONNX_NAMESPACE::GraphProto* p) override { return p->mutable_output(); }

  ONNX_NAMESPACE::ValueInfoProtos* GraphProto__mutable_value_info(ONNX_NAMESPACE::GraphProto* p) override { return p->mutable_value_info(); }
  ONNX_NAMESPACE::TensorProtos* GraphProto__mutable_initializer(ONNX_NAMESPACE::GraphProto* p) override { return p->mutable_initializer(); }
  ONNX_NAMESPACE::NodeProto* GraphProto__add_node(ONNX_NAMESPACE::GraphProto* p) override { return p->add_node(); }

  void GraphProto__operator_assign(ONNX_NAMESPACE::GraphProto* p, const ONNX_NAMESPACE::GraphProto& v) override { *p = v; }

  // ModelProto (wrapped)
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> ModelProto__construct() override { return std::make_unique<ONNX_NAMESPACE::ModelProto>(); }
  void ModelProto__operator_delete(ONNX_NAMESPACE::ModelProto* p) override { delete p; }

  bool ModelProto__SerializeToString(const ONNX_NAMESPACE::ModelProto* p, std::string& string) override { return p->SerializeToString(&string); }
  bool ModelProto__SerializeToOstream(const ONNX_NAMESPACE::ModelProto* p, std::ostream& output) override { return p->SerializeToOstream(&output); }
  bool ModelProto__ParseFromString(ONNX_NAMESPACE::ModelProto* p, const std::string& data) override { return p->ParseFromString(data); }
  std::string ModelProto__SerializeAsString(const ONNX_NAMESPACE::ModelProto* p) override { return p->SerializeAsString(); }

  const ONNX_NAMESPACE::GraphProto& ModelProto__graph(const ONNX_NAMESPACE::ModelProto* p) override { return p->graph(); }
  ONNX_NAMESPACE::GraphProto* ModelProto__mutable_graph(ONNX_NAMESPACE::ModelProto* p) override { return p->mutable_graph(); }

  void ModelProto__set_ir_version(ONNX_NAMESPACE::ModelProto* p, int64_t value) override { p->set_ir_version(value); }

  // NodeProto (wrapped)
  std::unique_ptr<ONNX_NAMESPACE::NodeProto> NodeProto__construct() override { return std::make_unique<ONNX_NAMESPACE::NodeProto>(); }
  void NodeProto__operator_delete(ONNX_NAMESPACE::NodeProto* p) override { delete p; }
  void NodeProto__operator_assign(ONNX_NAMESPACE::NodeProto* p, const ONNX_NAMESPACE::NodeProto& v) override { *p = v; }
  int NodeProto__attribute_size(ONNX_NAMESPACE::NodeProto* p) override { return p->attribute_size(); }
  const ONNX_NAMESPACE::AttributeProto& NodeProto__attribute(const ONNX_NAMESPACE::NodeProto* p, int index) const override { return p->attribute(index); }

  // TensorProto (wrapped)
  std::unique_ptr<ONNX_NAMESPACE::TensorProto> TensorProto__construct() override { return std::make_unique<ONNX_NAMESPACE::TensorProto>(); }
  void TensorProto__operator_delete(ONNX_NAMESPACE::TensorProto* p) override { delete p; }
  void TensorProto__operator_assign(ONNX_NAMESPACE::TensorProto* p, const ONNX_NAMESPACE::TensorProto& v) override { *p = v; }
  bool TensorProto__has_name(const ONNX_NAMESPACE::TensorProto* p) override { return p->has_name(); }
  int TensorProto__dims_size(const ONNX_NAMESPACE::TensorProto* p) override { return p->dims_size(); }
  const ONNX_NAMESPACE::int64s& TensorProto__dims(const ONNX_NAMESPACE::TensorProto* p) override { return p->dims(); }
  bool TensorProto__has_data_location(const ONNX_NAMESPACE::TensorProto* p) override { return p->has_data_location(); }
  int TensorProto__data_location(const ONNX_NAMESPACE::TensorProto* p) override { return p->data_location(); }
  bool TensorProto__has_raw_data(const ONNX_NAMESPACE::TensorProto* p) override { return p->has_raw_data(); }
  const std::string& TensorProto__raw_data(const ONNX_NAMESPACE::TensorProto* p) override { return p->raw_data(); }
  int32_t TensorProto__data_type(const ONNX_NAMESPACE::TensorProto* p) override { return p->data_type(); }

  bool TensorProto_DataType_IsValid(int value) override { return ONNX_NAMESPACE::TensorProto::DataType_IsValid(value); }
  void TensorProto__CopyFrom(ONNX_NAMESPACE::TensorProto* p, const ONNX_NAMESPACE::TensorProto* other) override { p->CopyFrom(*other); }

  // TensorProtos (wrapped)
  ONNX_NAMESPACE::TensorProto* TensorProtos__Add(ONNX_NAMESPACE::TensorProtos* p) override { return p->Add(); }

  // TensorShapeProto_Dimension (wrapped)
  int TensorShapeProto_Dimension__value_case(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) override { return p->value_case(); }
  const std::string& TensorShapeProto_Dimension__dim_param(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) override { return p->dim_param(); }
  int64_t TensorShapeProto_Dimension__dim_value(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) override { return p->dim_value(); }
  void TensorShapeProto_Dimension__clear_dim_value(ONNX_NAMESPACE::TensorShapeProto_Dimension* p) override { return p->clear_dim_value(); }
  void TensorShapeProto_Dimension__set_dim_value(ONNX_NAMESPACE::TensorShapeProto_Dimension* p, int64_t value) override { return p->set_dim_value(value); }
  bool TensorShapeProto_Dimension__has_dim_value(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) override { return p->has_dim_value(); }
  bool TensorShapeProto_Dimension__has_dim_param(const ONNX_NAMESPACE::TensorShapeProto_Dimension* p) override { return p->has_dim_param(); }

  // TensorShapeProto_Dimensions (wrapped)
  std::unique_ptr<TensorShapeProto_Dimension_Iterator> TensorShapeProto_Dimensions__begin(const ONNX_NAMESPACE::TensorShapeProto_Dimensions* p) override {
    return std::make_unique<TensorShapeProto_Dimension_Iterator_Impl>(p->begin());
  }

  std::unique_ptr<TensorShapeProto_Dimension_Iterator> TensorShapeProto_Dimensions__end(const ONNX_NAMESPACE::TensorShapeProto_Dimensions* p) override {
    return std::make_unique<TensorShapeProto_Dimension_Iterator_Impl>(p->end());
  }

  // TensorShapeProto (wrapped)
  int TensorShapeProto__dim_size(const ONNX_NAMESPACE::TensorShapeProto* p) override { return p->dim_size(); }
  const ONNX_NAMESPACE::TensorShapeProto_Dimensions& TensorShapeProto__dim(const ONNX_NAMESPACE::TensorShapeProto* p) override { return p->dim(); }
  const ONNX_NAMESPACE::TensorShapeProto_Dimension& TensorShapeProto__dim(const ONNX_NAMESPACE::TensorShapeProto* p, int index) override { return p->dim(index); }
  ONNX_NAMESPACE::TensorShapeProto_Dimension* TensorShapeProto__mutable_dim(ONNX_NAMESPACE::TensorShapeProto* p, int index) override { return p->mutable_dim(index); }
  void TensorShapeProto__clear_dim(ONNX_NAMESPACE::TensorShapeProto* p) override { return p->clear_dim(); }
  ONNX_NAMESPACE::TensorShapeProto_Dimension* TensorShapeProto__add_dim(ONNX_NAMESPACE::TensorShapeProto* p) override { return p->add_dim(); }

  // ValueInfoProto (wrapped)
  const ONNX_NAMESPACE::TypeProto& ValueInfoProto__type(const ONNX_NAMESPACE::ValueInfoProto* p) override { return p->type(); }
  ONNX_NAMESPACE::TypeProto* ValueInfoProto__mutable_type(ONNX_NAMESPACE::ValueInfoProto* p) override { return p->mutable_type(); }
  virtual void ValueInfoProto__operator_assign(ONNX_NAMESPACE::ValueInfoProto* p, const ONNX_NAMESPACE::ValueInfoProto& v) override { *p = v; }

  // ValueInfoProtos (wrapped)
  ONNX_NAMESPACE::ValueInfoProto* ValueInfoProtos__Add(ONNX_NAMESPACE::ValueInfoProtos* p) override { return p->Add(); }

  const ONNX_NAMESPACE::ValueInfoProto& ValueInfoProtos__operator_array(const ONNX_NAMESPACE::ValueInfoProtos* p, int index) override { return (*p)[index]; }

  // ComputeCapability (wrapped)
  std::unique_ptr<ComputeCapability> ComputeCapability__construct(std::unique_ptr<IndexedSubGraph> t_sub_graph) override { return std::make_unique<ComputeCapability>(std::move(t_sub_graph)); }
  void ComputeCapability__operator_delete(ComputeCapability* p) override { delete p; }
  std::unique_ptr<IndexedSubGraph>& ComputeCapability__SubGraph(ComputeCapability* p) override { return p->sub_graph; }

  // DataTransferManager (wrapped)
  Status DataTransferManager__CopyTensor(const DataTransferManager* p, const Tensor& src, Tensor& dst) override { return p->CopyTensor(src, dst); }
#if !defined(DISABLE_SPARSE_TENSORS)
  Status DataTransferManager__CopySparseTensor(const DataTransferManager* p, const SparseTensor& src, SparseTensor& dst) override { return p->CopySparseTensor(src, dst); }
  Status DataTransferManager__CopySparseTensors(const DataTransferManager* p, const std::vector<IDataTransfer::SparseSrcDstPair>& src_dst_pairs) override { return p->CopySparseTensors(src_dst_pairs); };
#endif
  const IDataTransfer* DataTransferManager__GetDataTransfer(const DataTransferManager* p, const OrtDevice& src_device, const OrtDevice& dst_device) override { return p->GetDataTransfer(src_device, dst_device); }

  // IDataTransfer (direct)
  Status IDataTransfer__CopyTensor(const IDataTransfer* p, const Tensor& src, Tensor& dst) override { return p->IDataTransfer::CopyTensor(src, dst); }
  Status IDataTransfer__CopyTensors(const IDataTransfer* p, const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) override { return p->IDataTransfer::CopyTensors(src_dst_pairs); }
#if !defined(DISABLE_SPARSE_TENSORS)
  Status IDataTransfer__CopySparseTensors(const IDataTransfer* p, const std::vector<IDataTransfer::SparseSrcDstPair>& src_dst_pairs) override {
    return p->CopySparseTensors(src_dst_pairs);
  }
#endif

  // IndexedSubGraph_MetaDef (wrapped)
  std::unique_ptr<IndexedSubGraph_MetaDef> IndexedSubGraph_MetaDef__construct() override { return std::make_unique<IndexedSubGraph::MetaDef>(); }
  void IndexedSubGraph_MetaDef__operator_delete(IndexedSubGraph_MetaDef* p) override { delete p; }

  std::string& IndexedSubGraph_MetaDef__name(IndexedSubGraph_MetaDef* p) override { return p->name; }
  std::string& IndexedSubGraph_MetaDef__domain(IndexedSubGraph_MetaDef* p) override { return p->domain; }
  int& IndexedSubGraph_MetaDef__since_version(IndexedSubGraph_MetaDef* p) override { return p->since_version; }
  ONNX_NAMESPACE::OperatorStatus& IndexedSubGraph_MetaDef__status(IndexedSubGraph_MetaDef* p) override { return p->status; }
  std::vector<std::string>& IndexedSubGraph_MetaDef__inputs(IndexedSubGraph_MetaDef* p) override { return p->inputs; }
  std::vector<std::string>& IndexedSubGraph_MetaDef__outputs(IndexedSubGraph_MetaDef* p) override { return p->outputs; }
  std::vector<std::string>& IndexedSubGraph_MetaDef__constant_initializers(IndexedSubGraph_MetaDef* p) override { return p->constant_initializers; }
  NodeAttributes& IndexedSubGraph_MetaDef__attributes(IndexedSubGraph_MetaDef* p) override { return p->attributes; }
  std::string& IndexedSubGraph_MetaDef__doc_string(IndexedSubGraph_MetaDef* p) override { return p->doc_string; }

  // IndexedSubGraph (wrapped)
  std::unique_ptr<IndexedSubGraph> IndexedSubGraph__construct() override { return std::make_unique<IndexedSubGraph>(); }
  void IndexedSubGraph__operator_delete(IndexedSubGraph* p) override { delete p; }

  std::vector<onnxruntime::NodeIndex>& IndexedSubGraph__Nodes(IndexedSubGraph* p) override { return p->nodes; }

  void IndexedSubGraph__SetMetaDef(IndexedSubGraph* p, std::unique_ptr<IndexedSubGraph_MetaDef>&& meta_def_) override { return p->SetMetaDef(std::move(meta_def_)); }
  const IndexedSubGraph_MetaDef* IndexedSubGraph__GetMetaDef(const IndexedSubGraph* p) override { return p->GetMetaDef(); }

  // KernelDef (wrapped)
  void KernelDef__operator_delete(KernelDef* p) override { delete p; }
  void KernelDef__SinceVersion(const KernelDef* p, int* start, int* end) override { return p->SinceVersion(start, end); }
  const std::string& KernelDef__Domain(const KernelDef* p) override { return p->Domain(); }
  const std::string& KernelDef__OpName(const KernelDef* p) override { return p->OpName(); }
  int KernelDef__ExecQueueId(const KernelDef* p) override { return p->ExecQueueId(); }

  // KernelDefBuilder (wrapped)
  std::unique_ptr<KernelDefBuilder> KernelDefBuilder__construct() override { return std::make_unique<KernelDefBuilder>(); }
  void KernelDefBuilder__operator_delete(KernelDefBuilder* p) override { delete p; }

  void KernelDefBuilder__SetName(KernelDefBuilder* p, const char* op_name) override { p->SetName(op_name); }
  void KernelDefBuilder__SetDomain(KernelDefBuilder* p, const char* domain) override { p->SetDomain(domain); }
  void KernelDefBuilder__SinceVersion(KernelDefBuilder* p, int since_version) override { p->SinceVersion(since_version); }
  void KernelDefBuilder__SinceVersion(KernelDefBuilder* p, int since_version_start, int since_version_end) override { p->SinceVersion(since_version_start, since_version_end); }
  void KernelDefBuilder__Provider(KernelDefBuilder* p, const char* provider_type) override { p->Provider(provider_type); }
  void KernelDefBuilder__TypeConstraint(KernelDefBuilder* p, const char* arg_name, MLDataType supported_type) override { p->TypeConstraint(arg_name, supported_type); }
  void KernelDefBuilder__TypeConstraint(KernelDefBuilder* p, const char* arg_name, const std::vector<MLDataType>& supported_types) override { p->TypeConstraint(arg_name, supported_types); }
  void KernelDefBuilder__InputMemoryType(KernelDefBuilder* p, OrtMemType type, int input_index) override { p->InputMemoryType(type, input_index); }
  void KernelDefBuilder__InputMemoryType(KernelDefBuilder* p, OrtMemType type, const std::vector<int>& input_indexes) override { p->InputMemoryType(type, input_indexes); }
  void KernelDefBuilder__OutputMemoryType(KernelDefBuilder* p, OrtMemType type, int input_index) override { p->OutputMemoryType(type, input_index); }
  void KernelDefBuilder__ExecQueueId(KernelDefBuilder* p, int queue_id) override { p->ExecQueueId(queue_id); }
  void KernelDefBuilder__MayInplace(KernelDefBuilder* p, int input_index, int output_index) override { p->MayInplace(input_index, output_index); }
  void KernelDefBuilder__Alias(KernelDefBuilder* p, int input_index, int output_index) override { p->Alias(input_index, output_index); }
  void KernelDefBuilder__Alias(KernelDefBuilder* p, const std::vector<std::pair<int, int>>& aliases) override { p->Alias(aliases); }
  void KernelDefBuilder__VariadicAlias(KernelDefBuilder* p, int input_offset, int output_offset) override { p->VariadicAlias(input_offset, output_offset); }
  void KernelDefBuilder__ExternalOutputs(KernelDefBuilder* p) override { p->ExternalOutputs(); }
  void KernelDefBuilder__AllocateInputsContiguously(KernelDefBuilder* p) override { p->AllocateInputsContiguously(); }
#ifdef ENABLE_STRIDED_TENSORS
  void KernelDefBuilder__MayStridedInput(KernelDefBuilder* p, int input_index) override { p->MayStridedInput(input_index); }
  void KernelDefBuilder__MayStridedOutput(KernelDefBuilder* p, int input_index, int output_index) override { p->MayStridedOutput(input_index, output_index); }
#endif

  std::unique_ptr<KernelDef> KernelDefBuilder__Build(KernelDefBuilder* p) override { return p->Build(); }

  // KernelRegistry (wrapped)
  std::shared_ptr<KernelRegistry> KernelRegistry__construct() override { return std::make_shared<KernelRegistry>(); }
  void KernelRegistry__operator_delete(KernelRegistry* p) override { delete p; }
  Status KernelRegistry__Register(KernelRegistry* p, KernelCreateInfo&& create_info) override { return p->Register(std::move(create_info)); }

  // PrimitiveDataTypeBase (wrapped)
  int32_t PrimitiveDataTypeBase__GetDataType(const PrimitiveDataTypeBase* p) override { return p->GetDataType(); }

  // DataTypeImpl (wrapped)
  MLDataType DataTypeImpl__GetType_Tensor() override { return DataTypeImpl::GetType<Tensor>(); }
#if !defined(DISABLE_SPARSE_TENSORS)
  MLDataType DataTypeImpl__GetType_SparseTensor() override { return DataTypeImpl::GetType<SparseTensor>(); }
#endif
  MLDataType DataTypeImpl__GetType_TensorSeq() override { return DataTypeImpl::GetType<TensorSeq>(); }
  MLDataType DataTypeImpl__GetTypeFromOnnxType(int onnx_type) override { return DataTypeImpl::TensorTypeFromONNXEnum(onnx_type)->GetElementType(); }
  MLDataType DataTypeImpl__GetType_bool() override { return DataTypeImpl::GetType<bool>(); }
  MLDataType DataTypeImpl__GetType_int8() override { return DataTypeImpl::GetType<int8_t>(); }
  MLDataType DataTypeImpl__GetType_uint8() override { return DataTypeImpl::GetType<uint8_t>(); }
  MLDataType DataTypeImpl__GetType_int16() override { return DataTypeImpl::GetType<int16_t>(); }
  MLDataType DataTypeImpl__GetType_uint16() override { return DataTypeImpl::GetType<uint16_t>(); }
  MLDataType DataTypeImpl__GetType_int32() override { return DataTypeImpl::GetType<int32_t>(); }
  MLDataType DataTypeImpl__GetType_uint32() override { return DataTypeImpl::GetType<uint32_t>(); }
  MLDataType DataTypeImpl__GetType_int64() override { return DataTypeImpl::GetType<int64_t>(); }
  MLDataType DataTypeImpl__GetType_uint64() override { return DataTypeImpl::GetType<uint64_t>(); }
  MLDataType DataTypeImpl__GetType_float() override { return DataTypeImpl::GetType<float>(); }
  MLDataType DataTypeImpl__GetType_double() override { return DataTypeImpl::GetType<double>(); }
  MLDataType DataTypeImpl__GetType_BFloat16() override { return DataTypeImpl::GetType<BFloat16>(); }
  MLDataType DataTypeImpl__GetType_MLFloat16() override { return DataTypeImpl::GetType<MLFloat16>(); }
  MLDataType DataTypeImpl__GetType_string() override { return DataTypeImpl::GetType<std::string>(); }

#if !defined(DISABLE_FLOAT8_TYPES)
  MLDataType DataTypeImpl__GetType_Float8E4M3FN() override { return DataTypeImpl::GetType<Float8E4M3FN>(); }
  MLDataType DataTypeImpl__GetType_Float8E4M3FNUZ() override { return DataTypeImpl::GetType<Float8E4M3FNUZ>(); }
  MLDataType DataTypeImpl__GetType_Float8E5M2() override { return DataTypeImpl::GetType<Float8E5M2>(); }
  MLDataType DataTypeImpl__GetType_Float8E5M2FNUZ() override { return DataTypeImpl::GetType<Float8E5M2FNUZ>(); }
#endif

  MLDataType DataTypeImpl__GetTensorType_bool() override { return DataTypeImpl::GetTensorType<bool>(); }
  MLDataType DataTypeImpl__GetTensorType_int8() override { return DataTypeImpl::GetTensorType<int8_t>(); }
  MLDataType DataTypeImpl__GetTensorType_uint8() override { return DataTypeImpl::GetTensorType<uint8_t>(); }
  MLDataType DataTypeImpl__GetTensorType_int16() override { return DataTypeImpl::GetTensorType<int16_t>(); }
  MLDataType DataTypeImpl__GetTensorType_uint16() override { return DataTypeImpl::GetTensorType<uint16_t>(); }
  MLDataType DataTypeImpl__GetTensorType_int32() override { return DataTypeImpl::GetTensorType<int32_t>(); }
  MLDataType DataTypeImpl__GetTensorType_uint32() override { return DataTypeImpl::GetTensorType<uint32_t>(); }
  MLDataType DataTypeImpl__GetTensorType_int64() override { return DataTypeImpl::GetTensorType<int64_t>(); }
  MLDataType DataTypeImpl__GetTensorType_uint64() override { return DataTypeImpl::GetTensorType<uint64_t>(); }
  MLDataType DataTypeImpl__GetTensorType_float() override { return DataTypeImpl::GetTensorType<float>(); }
  MLDataType DataTypeImpl__GetTensorType_double() override { return DataTypeImpl::GetTensorType<double>(); }
  MLDataType DataTypeImpl__GetTensorType_BFloat16() override { return DataTypeImpl::GetTensorType<BFloat16>(); }
  MLDataType DataTypeImpl__GetTensorType_MLFloat16() override { return DataTypeImpl::GetTensorType<MLFloat16>(); }

#if !defined(DISABLE_FLOAT8_TYPES)
  MLDataType DataTypeImpl__GetTensorType_Float8E4M3FN() override { return DataTypeImpl::GetTensorType<Float8E4M3FN>(); }
  MLDataType DataTypeImpl__GetTensorType_Float8E4M3FNUZ() override { return DataTypeImpl::GetTensorType<Float8E4M3FNUZ>(); }
  MLDataType DataTypeImpl__GetTensorType_Float8E5M2() override { return DataTypeImpl::GetTensorType<Float8E5M2>(); }
  MLDataType DataTypeImpl__GetTensorType_Float8E5M2FNUZ() override { return DataTypeImpl::GetTensorType<Float8E5M2FNUZ>(); }
#endif

#if !defined(DISABLE_SPARSE_TENSORS)
  MLDataType DataTypeImpl__GetSparseTensorType_bool() override { return DataTypeImpl::GetSparseTensorType<bool>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_int8() override { return DataTypeImpl::GetSparseTensorType<int8_t>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_uint8() override { return DataTypeImpl::GetSparseTensorType<uint8_t>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_int16() override { return DataTypeImpl::GetSparseTensorType<int16_t>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_uint16() override { return DataTypeImpl::GetSparseTensorType<uint16_t>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_int32() override { return DataTypeImpl::GetSparseTensorType<int32_t>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_uint32() override { return DataTypeImpl::GetSparseTensorType<uint32_t>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_int64() override { return DataTypeImpl::GetSparseTensorType<int64_t>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_uint64() override { return DataTypeImpl::GetSparseTensorType<uint64_t>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_float() override { return DataTypeImpl::GetSparseTensorType<float>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_double() override { return DataTypeImpl::GetSparseTensorType<double>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_string() override { return DataTypeImpl::GetSparseTensorType<std::string>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_BFloat16() override { return DataTypeImpl::GetSparseTensorType<BFloat16>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_MLFloat16() override { return DataTypeImpl::GetSparseTensorType<MLFloat16>(); }
#if !defined(DISABLE_FLOAT8_TYPES)
  MLDataType DataTypeImpl__GetSparseTensorType_Float8E4M3FN() override { return DataTypeImpl::GetSparseTensorType<Float8E4M3FN>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_Float8E4M3FNUZ() override { return DataTypeImpl::GetSparseTensorType<Float8E4M3FNUZ>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_Float8E5M2() override { return DataTypeImpl::GetSparseTensorType<Float8E5M2>(); }
  MLDataType DataTypeImpl__GetSparseTensorType_Float8E5M2FNUZ() override { return DataTypeImpl::GetSparseTensorType<Float8E5M2FNUZ>(); }
#endif
#endif

  const char* DataTypeImpl__ToString(MLDataType type) override { return DataTypeImpl::ToString(type); }
  bool DataTypeImpl__IsTensorType(const DataTypeImpl* p) override { return p->IsTensorType(); }
  bool DataTypeImpl__IsTensorSequenceType(const DataTypeImpl* p) override { return p->IsTensorSequenceType(); }
#if !defined(DISABLE_SPARSE_TENSORS)
  bool DataTypeImpl__IsSparseTensorType(const DataTypeImpl* p) override { return p->IsSparseTensorType(); }
#endif

  DeleteFunc DataTypeImpl__GetDeleteFunc(const DataTypeImpl* p) override { return p->GetDeleteFunc(); }

  const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorTypes() override { return DataTypeImpl::AllFixedSizeTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorTypesIRv4() override { return DataTypeImpl::AllFixedSizeTensorTypesIRv4(); }
  const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorTypesIRv9() override { return DataTypeImpl::AllFixedSizeTensorTypesIRv9(); }

  const std::vector<MLDataType>& DataTypeImpl__AllTensorTypes() override { return DataTypeImpl::AllTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllTensorTypesIRv4() override { return DataTypeImpl::AllTensorTypesIRv4(); }
  const std::vector<MLDataType>& DataTypeImpl__AllTensorTypesIRv9() override { return DataTypeImpl::AllTensorTypesIRv9(); }

  const std::vector<MLDataType>& DataTypeImpl__AllIEEEFloatTensorTypes() override { return DataTypeImpl::AllIEEEFloatTensorTypes(); }

  const std::vector<MLDataType>& DataTypeImpl__AllTensorAndSequenceTensorTypes() override { return DataTypeImpl::AllTensorAndSequenceTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllTensorAndSequenceTensorTypesIRv4() override { return DataTypeImpl::AllTensorAndSequenceTensorTypesIRv4(); }
  const std::vector<MLDataType>& DataTypeImpl__AllTensorAndSequenceTensorTypesIRv9() override { return DataTypeImpl::AllTensorAndSequenceTensorTypesIRv9(); }

  const std::vector<MLDataType>& DataTypeImpl__AllOptionalAndTensorAndSequenceTensorTypes() override { return DataTypeImpl::AllOptionalAndTensorAndSequenceTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllOptionalAndTensorAndSequenceTensorTypesIRv4() override { return DataTypeImpl::AllOptionalAndTensorAndSequenceTensorTypesIRv4(); }
  const std::vector<MLDataType>& DataTypeImpl__AllOptionalAndTensorAndSequenceTensorTypesIRv9() override { return DataTypeImpl::AllOptionalAndTensorAndSequenceTensorTypesIRv9(); }

  const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypes() override { return DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypesIRv4() override { return DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypesIRv4(); }
  const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeTensorAndSequenceTensorTypesIRv9() override { return DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypesIRv9(); }

  const std::vector<MLDataType>& DataTypeImpl__AllSequenceTensorTypes() override { return DataTypeImpl::AllSequenceTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllSequenceTensorTypesIRv4() override { return DataTypeImpl::AllSequenceTensorTypesIRv4(); }
  const std::vector<MLDataType>& DataTypeImpl__AllSequenceTensorTypesIRv9() override { return DataTypeImpl::AllSequenceTensorTypesIRv9(); }

  const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeSequenceTensorTypes() override { return DataTypeImpl::AllFixedSizeSequenceTensorTypes(); }
  const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeSequenceTensorTypesIRv4() override { return DataTypeImpl::AllFixedSizeSequenceTensorTypesIRv4(); }
  const std::vector<MLDataType>& DataTypeImpl__AllFixedSizeSequenceTensorTypesIRv9() override { return DataTypeImpl::AllFixedSizeSequenceTensorTypesIRv9(); }

  size_t DataTypeImpl__Size(const DataTypeImpl* p) override { return p->Size(); }
  const PrimitiveDataTypeBase* DataTypeImpl__AsPrimitiveDataType(const DataTypeImpl* p) override { return p->AsPrimitiveDataType(); }

  // Function (wrapped)
  const Graph& Function__Body(const Function* p) override { return p->Body(); }

  // Node (wrapped)
  const std::string& Node__Name(const Node* p) noexcept override { return p->Name(); }
  const std::string& Node__Description(const Node* p) noexcept override { return p->Description(); }
  const std::string& Node__Domain(const Node* p) noexcept override { return p->Domain(); }
  const std::string& Node__OpType(const Node* p) noexcept override { return p->OpType(); }
  int Node__SinceVersion(const Node* p) override { return p->SinceVersion(); }

  const Function* Node__GetFunctionBody(const Node* p) noexcept override { return p->GetFunctionBody(); }
  ProviderType Node__GetExecutionProviderType(const Node* p) const noexcept override { return p->GetExecutionProviderType(); }

  const std::vector<int>& Node__InputArgCount(const Node* p) override { return p->InputArgCount(); }
  ConstPointerContainer<std::vector<NodeArg*>> Node__ImplicitInputDefs(const Node* p) noexcept override { return p->ImplicitInputDefs(); }
  ConstPointerContainer<std::vector<NodeArg*>> Node__InputDefs(const Node* p) noexcept override { return p->InputDefs(); }
  ConstPointerContainer<std::vector<NodeArg*>> Node__OutputDefs(const Node* p) noexcept override { return p->OutputDefs(); }
  NodeIndex Node__Index(const Node* p) noexcept override { return p->Index(); }
  std::vector<gsl::not_null<const Graph*>> Node__GetSubgraphs(const Node* p) const noexcept override { return p->GetSubgraphs(); }

  void Node__ToProto(const Node* p, ONNX_NAMESPACE::NodeProto& proto, bool update_subgraphs = false) override { p->ToProto(proto, update_subgraphs); }

  const NodeAttributes& Node__GetAttributes(const Node* p) noexcept override { return p->GetAttributes(); }
  size_t Node__GetInputEdgesCount(const Node* p) noexcept override { return p->GetInputEdgesCount(); }
  size_t Node__GetOutputEdgesCount(const Node* p) noexcept override { return p->GetOutputEdgesCount(); }

  std::unique_ptr<Node__NodeIterator> Node__InputNodesBegin(const Node* p) noexcept override { return std::make_unique<Node__NodeIterator_Impl>(p->InputNodesBegin()); }
  std::unique_ptr<Node__NodeIterator> Node__InputNodesEnd(const Node* p) noexcept override { return std::make_unique<Node__NodeIterator_Impl>(p->InputNodesEnd()); }

  std::unique_ptr<Node__NodeIterator> Node__OutputNodesBegin(const Node* p) noexcept override { return std::make_unique<Node__NodeIterator_Impl>(p->OutputNodesBegin()); }
  std::unique_ptr<Node__NodeIterator> Node__OutputNodesEnd(const Node* p) noexcept override { return std::make_unique<Node__NodeIterator_Impl>(p->OutputNodesEnd()); }

  std::unique_ptr<Node__EdgeIterator> Node__OutputEdgesBegin(const Node* p) noexcept override { return std::make_unique<Node__EdgeIterator_Impl>(p->OutputEdgesBegin()); }
  std::unique_ptr<Node__EdgeIterator> Node__OutputEdgesEnd(const Node* p) noexcept override { return std::make_unique<Node__EdgeIterator_Impl>(p->OutputEdgesEnd()); }

  void Node__ForEachDef(const Node* p, std::function<void(const NodeArg&, bool is_input)> func, bool include_missing_optional_defs) override { p->ForEachDef(func, std::move(include_missing_optional_defs)); }
  const std::unordered_map<std::string, gsl::not_null<Graph*>>& Node__GetAttributeNameToMutableSubgraphMap(Node* p) noexcept override { return p->GetAttributeNameToMutableSubgraphMap(); }
  std::unordered_map<std::string, gsl::not_null<const Graph*>> Node__GetAttributeNameToSubgraphMap(const Node* p) const override { return p->GetAttributeNameToSubgraphMap(); }

  // NodeArg (wrapped)
  const std::string& NodeArg__Name(const NodeArg* p) noexcept override { return p->Name(); }
  const ONNX_NAMESPACE::TensorShapeProto* NodeArg__Shape(const NodeArg* p) override { return p->Shape(); }
  ONNX_NAMESPACE::DataType NodeArg__Type(const NodeArg* p) noexcept override { return p->Type(); }
  const NodeArgInfo& NodeArg__ToProto(const NodeArg* p) noexcept override { return p->ToProto(); }
  bool NodeArg__Exists(const NodeArg* p) const noexcept override { return p->Exists(); }
  const ONNX_NAMESPACE::TypeProto* NodeArg__TypeAsProto(const NodeArg* p) noexcept override { return p->TypeAsProto(); }

  // NodeAttributes (wrapped)
  std::unique_ptr<NodeAttributes> NodeAttributes__construct() override { return std::make_unique<NodeAttributes>(); }
  void NodeAttributes__operator_delete(NodeAttributes* p) noexcept override { delete p; }
  size_t NodeAttributes__size(const NodeAttributes* p) override { return p->size(); }
  void NodeAttributes__clear(NodeAttributes* p) noexcept override { return p->clear(); }
  size_t NodeAttributes__count(const NodeAttributes* p, const std::string& keyval) override { return p->count(keyval); }
  ONNX_NAMESPACE::AttributeProto& NodeAttributes__operator_array(NodeAttributes* p, const std::string& string) override { return (*p)[string]; }
  const ONNX_NAMESPACE::AttributeProto& NodeAttributes__at(const NodeAttributes* p, const std::string& string) override { return p->at(string); }
  void NodeAttributes__operator_assign(NodeAttributes* p, const NodeAttributes& v) override { *p = v; }

  std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__begin(const NodeAttributes* p) override {
    return std::make_unique<NodeAttributes_Iterator_Impl>(p->begin());
  }
  std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__end(const NodeAttributes* p) override {
    return std::make_unique<NodeAttributes_Iterator_Impl>(p->end());
  }
  std::unique_ptr<NodeAttributes_Iterator> NodeAttributes__find(const NodeAttributes* p, const std::string& key) override {
    return std::make_unique<NodeAttributes_Iterator_Impl>(p->find(key));
  }
  void NodeAttributes__insert(NodeAttributes* p, const NodeAttributes& v) override { return p->insert(v.begin(), v.end()); }
  void NodeAttributes__emplace(NodeAttributes* p, const std::string& k, const ONNX_NAMESPACE::AttributeProto& v) override { p->emplace(k, v); }
  void NodeAttributes__reserve(NodeAttributes* p, size_t size) override { p->reserve(size); }

  // Model (wrapped)
  void Model__operator_delete(Model* p) override { delete p; }
  Graph& Model__MainGraph(Model* p) override { return p->MainGraph(); }
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> Model__ToProto(Model* p) override { return std::make_unique<ONNX_NAMESPACE::ModelProto>(p->ToProto()); }

  // Graph (wrapped)
  std::unique_ptr<GraphViewer> Graph__CreateGraphViewer(const Graph* p) override { return std::make_unique<GraphViewer>(*p); }
  std::unique_ptr<ONNX_NAMESPACE::GraphProto> Graph__ToGraphProto(const Graph* p) override { return std::make_unique<ONNX_NAMESPACE::GraphProto>(p->ToGraphProto()); }
  void Graph__SetInputs(Graph* p, gsl::span<const NodeArg* const> inputs) override { p->SetInputs(inputs); }

  NodeArg& Graph__GetOrCreateNodeArg(Graph* p, const std::string& name, const ONNX_NAMESPACE::TypeProto* p_arg_type) override { return p->GetOrCreateNodeArg(name, p_arg_type); }
  void Graph__AddOuterScopeNodeArg(Graph* p, const std::string& name) override { p->AddOuterScopeNodeArg(name); }

  Status Graph__Resolve(Graph* p) override { return p->Resolve(); }
  void Graph__AddInitializedTensor(Graph* p, const ONNX_NAMESPACE::TensorProto& tensor) override { p->AddInitializedTensor(tensor); }
  Node& Graph__AddNode(Graph* p, const std::string& name, const std::string& op_type, const std::string& description, const gsl::span<NodeArg* const>& input_args, const gsl::span<NodeArg* const>& output_args, const NodeAttributes* attributes, const std::string& domain) override {
    return p->AddNode(name, op_type, description, input_args, output_args, attributes, domain);
  }

  const std::vector<const NodeArg*>& Graph__GetOutputs(const Graph* p) noexcept override { return p->GetOutputs(); }
  void Graph__SetOutputs(Graph* p, gsl::span<const NodeArg* const> outputs) override { p->SetOutputs(outputs); }

  const std::vector<const NodeArg*>& Graph__GetInputs(const Graph* p) noexcept override { return p->GetInputs(); }
  bool Graph__GetInitializedTensor(const Graph* p, const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) override { return p->GetInitializedTensor(tensor_name, value); }

  const Node* Graph__ParentNode(const Graph* p) const override { return p->ParentNode(); }
  const Graph* Graph__ParentGraph(const Graph* p) const override { return p->ParentGraph(); }
  Graph* Graph__MutableParentGraph(Graph* p) override { return p->MutableParentGraph(); }
  const std::string& Graph__Name(const Graph* p) const noexcept override { return p->Name(); }
  const Path& Graph__ModelPath(const Graph* p) const override { return p->ModelPath(); }
  const std::vector<const NodeArg*>& Graph__GetInputsIncludingInitializers(const Graph* p) const noexcept override { return p->GetInputsIncludingInitializers(); }
  bool Graph__IsSubgraph(const Graph* p) override { return p->IsSubgraph(); }
  int Graph__MaxNodeIndex(const Graph* p) const noexcept override { return p->MaxNodeIndex(); }
  Node* Graph__GetNode(Graph* p, NodeIndex node_index) noexcept override { return p->GetNode(node_index); }
  const Node* Graph__GetNode(const Graph* p, NodeIndex node_index) const override { return p->GetNode(node_index); }
  const NodeArg* Graph__GetNodeArg(const Graph* p, const std::string& name) const override { return p->GetNodeArg(name); }

  // GraphViewer (wrapped)
  void GraphViewer__operator_delete(GraphViewer* p) override { delete p; }
  std::unique_ptr<Model> GraphViewer__CreateModel(const GraphViewer* graph_viewer, const logging::Logger& logger) override {
    return std::make_unique<Model>(graph_viewer->Name(), true, ModelMetaData(), PathString(),
#if !defined(ORT_MINIMAL_BUILD)
                                   IOnnxRuntimeOpSchemaRegistryList({graph_viewer->GetSchemaRegistry()}), graph_viewer->DomainToVersionMap(),
#else
                                   IOnnxRuntimeOpSchemaRegistryList(), graph_viewer->DomainToVersionMap(),
#endif  // ORT_MINIMAL_BUILD
                                   std::vector<ONNX_NAMESPACE::FunctionProto>(), logger);
  }

  const std::string& GraphViewer__Name(const GraphViewer* p) noexcept override { return p->Name(); }
  const Path& GraphViewer__ModelPath(const GraphViewer* p) noexcept override { return p->ModelPath(); }

  const Node* GraphViewer__GetNode(const GraphViewer* p, NodeIndex node_index) override { return p->GetNode(node_index); }
  const NodeArg* GraphViewer__GetNodeArg(const GraphViewer* p, const std::string& name) override { return p->GetNodeArg(name); }

  bool GraphViewer__IsSubgraph(const GraphViewer* p) override { return p->IsSubgraph(); }
  const Graph& GraphViewer__GetGraph(const GraphViewer* p) const override { return p->GetGraph(); }
  bool GraphViewer__IsConstantInitializer(const GraphViewer* p, const std::string& name, bool check_outer_scope) override { return p->IsConstantInitializer(name, check_outer_scope); }
  const Node* GraphViewer__ParentNode(const GraphViewer* p) override { return p->ParentNode(); }
  int GraphViewer__NumberOfNodes(const GraphViewer* p) noexcept override { return p->NumberOfNodes(); }
  int GraphViewer__MaxNodeIndex(const GraphViewer* p) noexcept override { return p->MaxNodeIndex(); }

  const std::vector<const NodeArg*>& GraphViewer__GetInputs(const GraphViewer* p) noexcept override { return p->GetInputs(); }
  const std::vector<const NodeArg*>& GraphViewer__GetOutputs(const GraphViewer* p) noexcept override { return p->GetOutputs(); }
  const std::unordered_set<const NodeArg*>& GraphViewer__GetValueInfo(const GraphViewer* p) noexcept override { return p->GetValueInfo(); }

  const InitializedTensorSet& GraphViewer__GetAllInitializedTensors(const GraphViewer* p) override { return p->GetAllInitializedTensors(); }
  bool GraphViewer__GetInitializedTensor(const GraphViewer* p, const std::string& tensor_name, const ONNX_NAMESPACE::TensorProto*& value) override { return p->GetInitializedTensor(tensor_name, value); }

  const std::unordered_map<std::string, int>& GraphViewer__DomainToVersionMap(const GraphViewer* p) override { return p->DomainToVersionMap(); }

  const std::vector<NodeIndex>& GraphViewer__GetNodesInTopologicalOrder(const GraphViewer* p) override { return p->GetNodesInTopologicalOrder(); }
  const std::vector<const NodeArg*>& GraphViewer__GetInputsIncludingInitializers(const GraphViewer* p) noexcept override { return p->GetInputsIncludingInitializers(); }
  void GraphViewer__ToProto(const GraphViewer* p, ONNX_NAMESPACE::GraphProto& graph_proto, bool include_initializers, bool include_outer_scope_args) noexcept override {
    GraphViewerToProto(*p, graph_proto, include_initializers, include_outer_scope_args);
  }

  // Path (wrapped)
  PathString Path__ToPathString(const Path* p) noexcept override { return p->ToPathString(); }
  const std::vector<PathString>& Path__GetComponents(const Path* p) noexcept override { return p->GetComponents(); }
  bool Path__IsEmpty(const Path* p) noexcept override { return p->IsEmpty(); }

  // OpKernel (direct)
  const Node& OpKernel__Node(const OpKernel* p) override { return p->OpKernel::Node(); }

  // OpKernelContext (wrapped)
  const Tensor* OpKernelContext__Input_Tensor(const OpKernelContext* p, int index) override { return p->Input<Tensor>(index); }
#if !defined(DISABLE_SPARSE_TENSORS)
  const SparseTensor* OpKernelContext__Input_SparseTensor(const OpKernelContext* p, int index) override { return p->Input<SparseTensor>(index); }
#endif
  const TensorSeq* OpKernelContext__Input_TensorSeq(const OpKernelContext* p, int index) override { return p->Input<TensorSeq>(index); }
  const Tensor& OpKernelContext__RequiredInput_Tensor(const OpKernelContext* p, int index) override { return p->RequiredInput<Tensor>(index); }
  MLDataType OpKernelContext__InputType(const OpKernelContext* p, int index) override { return p->InputType(index); }
  Tensor* OpKernelContext__Output_Tensor(OpKernelContext* p, int index) override { return p->Output<Tensor>(index); }
  TensorSeq* OpKernelContext__Output_TensorSeq(OpKernelContext* p, int index) override { return p->Output<TensorSeq>(index); }
  Tensor* OpKernelContext__Output(OpKernelContext* p, int index, const TensorShape& shape) override { return p->Output(index, shape); }
#if !defined(DISABLE_SPARSE_TENSORS)
  SparseTensor* OpKernelContext__OutputSparse(OpKernelContext* p, int index, const TensorShape& shape) override { return p->OutputSparse(index, shape); }
#endif
  Tensor& OpKernelContext__RequiredOutput(OpKernelContext* p, int index, const TensorShape& shape) override { return p->RequiredOutput(index, shape); }
  int OpKernelContext__InputCount(const OpKernelContext* p) override { return p->InputCount(); }
  int OpKernelContext__OutputCount(const OpKernelContext* p) override { return p->OutputCount(); }
  Status OpKernelContext__GetTempSpaceAllocator(const OpKernelContext* p, AllocatorPtr* output) override { return p->GetTempSpaceAllocator(output); }
  Status OpKernelContext__GetTempSpaceCPUAllocator(const OpKernelContext* p, AllocatorPtr* output) override { return p->GetTempSpaceCPUAllocator(output); }
  bool OpKernelContext__GetUseDeterministicCompute(const OpKernelContext* p) override { return p->GetUseDeterministicCompute(); }
  bool OpKernelContext__TryGetInferredOutputShape(const OpKernelContext* p, int index, TensorShape& shape) override { return p->TryGetInferredOutputShape(index, shape); }
  bool OpKernelContext__TryGetInferredInputShape(const OpKernelContext* p, int index, TensorShape& shape) override { return p->TryGetInferredInputShape(index, shape); }
  Stream* OpKernelContext__GetComputeStream(const OpKernelContext* p) override { return p->GetComputeStream(); }

  // OpKernelInfo (wrapped)
  std::unique_ptr<OpKernelInfo> CopyOpKernelInfo(const OpKernelInfo& info) override { return onnxruntime::CopyOpKernelInfo(info); }
  void OpKernelInfo__operator_delete(OpKernelInfo* p) override { delete p; }
  AllocatorPtr OpKernelInfo__GetAllocator(const OpKernelInfo* p, OrtMemType mem_type) override { return p->GetAllocator(mem_type); }
  const IExecutionProvider* OpKernelInfo__GetExecutionProvider(const OpKernelInfo* p) override { return p->GetExecutionProvider(); }
  Status OpKernelInfo__GetAttr_int64(const OpKernelInfo* p, const std::string& name, int64_t* value) override { return p->GetAttr(name, value); }
  Status OpKernelInfo__GetAttr_float(const OpKernelInfo* p, const std::string& name, float* value) override { return p->GetAttr(name, value); }
  Status OpKernelInfo__GetAttr_string(const OpKernelInfo* p, const std::string& name, std::string* value) override { return p->GetAttr(name, value); }
  Status OpKernelInfo__GetAttr_TensorProto(const OpKernelInfo* p, const std::string& name, ONNX_NAMESPACE::TensorProto* value) override { return p->GetAttr(name, value); }
  Status OpKernelInfo__GetAttrs(const OpKernelInfo* p, const std::string& name, std::vector<int64_t>& values) override { return p->GetAttrs(name, values); }
  Status OpKernelInfo__GetAttrs(const OpKernelInfo* p, const std::string& name, std::vector<float>& values) override { return p->GetAttrs(name, values); }
  Status OpKernelInfo__GetAttrs(const OpKernelInfo* p, const std::string& name, std::vector<std::string>& values) override { return p->GetAttrs(name, values); }
  Status OpKernelInfo__GetAttrsAsSpan(const OpKernelInfo* p, const std::string& name, gsl::span<const int64_t>& values) override {
    return p->GetAttrsAsSpan(name, values);
  }

  const DataTransferManager& OpKernelInfo__GetDataTransferManager(const OpKernelInfo* p) noexcept override { return p->GetDataTransferManager(); }
  const KernelDef& OpKernelInfo__GetKernelDef(const OpKernelInfo* p) override { return p->GetKernelDef(); }
  bool OpKernelInfo__TryGetConstantInput(const OpKernelInfo* p, int input_index, const Tensor** constant_input_value) override { return p->TryGetConstantInput(input_index, constant_input_value); }

  uint32_t OpKernelInfo__GetInputCount(const OpKernelInfo* p) override { return p->GetInputCount(); }
  uint32_t OpKernelInfo__GetOutputCount(const OpKernelInfo* p) override { return p->GetOutputCount(); }
  const Node& OpKernelInfo__node(const OpKernelInfo* p) override { return p->node(); }

  // SessionState (wrapped)
  const DataTransferManager& SessionState__GetDataTransferMgr(const SessionState* p) override { return p->GetDataTransferMgr(); }

  // Tensor (wrapped)
  std::unique_ptr<Tensor> Tensor__construct(MLDataType p_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator) override {
    return std::make_unique<Tensor>(p_type, shape, std::move(allocator));
  }

  std::unique_ptr<Tensor> Tensor__construct(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& alloc, ptrdiff_t offset) override {
    return std::make_unique<Tensor>(p_type, shape, p_data, alloc, offset);
  }

  std::unique_ptr<Tensor> Tensor__construct_default() override {
    return std::make_unique<Tensor>();
  }

  virtual void Tensor__move_assign(Tensor& lhs, Tensor&& rhs) noexcept override {
    lhs = std::move(rhs);
  };

  void Tensor__operator_delete(Tensor* p) noexcept override { delete p; }

  void Tensor__InitOrtValue(MLDataType elt_type, const TensorShape& shape, std::shared_ptr<IAllocator> allocator, OrtValue& ort_value) override {
    Tensor::InitOrtValue(elt_type, shape, std::move(allocator), ort_value);
  }

  void Tensor__InitOrtValue(MLDataType p_type, const TensorShape& shape, void* p_data, const OrtMemoryInfo& location, OrtValue& ort_value) override {
    Tensor::InitOrtValue(p_type, shape, p_data, location, ort_value);
  }

  bool* Tensor__MutableData_bool(Tensor* p) override { return p->MutableData<bool>(); }
  int8_t* Tensor__MutableData_int8(Tensor* p) override { return p->MutableData<int8_t>(); }
  uint8_t* Tensor__MutableData_uint8(Tensor* p) override { return p->MutableData<uint8_t>(); }
  int16_t* Tensor__MutableData_int16(Tensor* p) override { return p->MutableData<int16_t>(); }
  uint16_t* Tensor__MutableData_uint16(Tensor* p) override { return p->MutableData<uint16_t>(); }
  int32_t* Tensor__MutableData_int32(Tensor* p) override { return p->MutableData<int32_t>(); }
  uint32_t* Tensor__MutableData_uint32(Tensor* p) override { return p->MutableData<uint32_t>(); }
  int64_t* Tensor__MutableData_int64(Tensor* p) override { return p->MutableData<int64_t>(); }
  uint64_t* Tensor__MutableData_uint64(Tensor* p) override { return p->MutableData<uint64_t>(); }
  float* Tensor__MutableData_float(Tensor* p) override { return p->MutableData<float>(); }
  double* Tensor__MutableData_double(Tensor* p) override { return p->MutableData<double>(); }
  BFloat16* Tensor__MutableData_BFloat16(Tensor* p) override { return p->MutableData<BFloat16>(); }
  MLFloat16* Tensor__MutableData_MLFloat16(Tensor* p) override { return p->MutableData<MLFloat16>(); }

#if !defined(DISABLE_FLOAT8_TYPES)
  Float8E4M3FN* Tensor__MutableData_Float8E4M3FN(Tensor* p) override { return p->MutableData<Float8E4M3FN>(); }
  Float8E4M3FNUZ* Tensor__MutableData_Float8E4M3FNUZ(Tensor* p) override { return p->MutableData<Float8E4M3FNUZ>(); }
  Float8E5M2* Tensor__MutableData_Float8E5M2(Tensor* p) override { return p->MutableData<Float8E5M2>(); }
  Float8E5M2FNUZ* Tensor__MutableData_Float8E5M2FNUZ(Tensor* p) override { return p->MutableData<Float8E5M2FNUZ>(); }
#endif

  const bool* Tensor__Data_bool(const Tensor* p) override { return p->Data<bool>(); }
  const int8_t* Tensor__Data_int8(const Tensor* p) override { return p->Data<int8_t>(); }
  const uint8_t* Tensor__Data_uint8(const Tensor* p) override { return p->Data<uint8_t>(); }
  const int16_t* Tensor__Data_int16(const Tensor* p) override { return p->Data<int16_t>(); }
  const uint16_t* Tensor__Data_uint16(const Tensor* p) override { return p->Data<uint16_t>(); }
  const int32_t* Tensor__Data_int32(const Tensor* p) override { return p->Data<int32_t>(); }
  const uint32_t* Tensor__Data_uint32(const Tensor* p) override { return p->Data<uint32_t>(); }
  const int64_t* Tensor__Data_int64(const Tensor* p) override { return p->Data<int64_t>(); }
  const uint64_t* Tensor__Data_uint64(const Tensor* p) override { return p->Data<uint64_t>(); }
  const float* Tensor__Data_float(const Tensor* p) override { return p->Data<float>(); }
  const double* Tensor__Data_double(const Tensor* p) override { return p->Data<double>(); }
  const BFloat16* Tensor__Data_BFloat16(const Tensor* p) override { return p->Data<BFloat16>(); }
  const MLFloat16* Tensor__Data_MLFloat16(const Tensor* p) override { return p->Data<MLFloat16>(); }

#if !defined(DISABLE_FLOAT8_TYPES)
  const Float8E4M3FN* Tensor__Data_Float8E4M3FN(const Tensor* p) override { return p->Data<Float8E4M3FN>(); }
  const Float8E4M3FNUZ* Tensor__Data_Float8E4M3FNUZ(const Tensor* p) override { return p->Data<Float8E4M3FNUZ>(); }
  const Float8E5M2* Tensor__Data_Float8E5M2(const Tensor* p) override { return p->Data<Float8E5M2>(); }
  const Float8E5M2FNUZ* Tensor__Data_Float8E5M2FNUZ(const Tensor* p) override { return p->Data<Float8E5M2FNUZ>(); }
#endif

  gsl::span<const int64_t> Tensor__DataAsSpan_int64(const Tensor* p) override { return p->DataAsSpan<int64_t>(); }

  void* Tensor__MutableDataRaw(Tensor* p, MLDataType type) override { return p->MutableDataRaw(type); }
  const void* Tensor__DataRaw(const Tensor* p, MLDataType type) override { return p->DataRaw(type); }
  void* Tensor__MutableDataRaw(Tensor* p) noexcept override { return p->MutableDataRaw(); }
  const void* Tensor__DataRaw(const Tensor* p) noexcept override { return p->DataRaw(); }

  bool Tensor__IsDataType_bool(const Tensor* p) noexcept override { return p->IsDataType<bool>(); }
  bool Tensor__IsDataType_int8(const Tensor* p) noexcept override { return p->IsDataType<int8_t>(); }
  bool Tensor__IsDataType_uint8(const Tensor* p) noexcept override { return p->IsDataType<uint8_t>(); }
  bool Tensor__IsDataType_int16(const Tensor* p) noexcept override { return p->IsDataType<int16_t>(); }
  bool Tensor__IsDataType_uint16(const Tensor* p) noexcept override { return p->IsDataType<uint16_t>(); }
  bool Tensor__IsDataType_int32(const Tensor* p) noexcept override { return p->IsDataType<int32_t>(); }
  bool Tensor__IsDataType_uint32(const Tensor* p) noexcept override { return p->IsDataType<uint32_t>(); }
  bool Tensor__IsDataType_int64(const Tensor* p) noexcept override { return p->IsDataType<int64_t>(); }
  bool Tensor__IsDataType_uint64(const Tensor* p) noexcept override { return p->IsDataType<uint64_t>(); }
  bool Tensor__IsDataType_float(const Tensor* p) noexcept override { return p->IsDataType<float>(); }
  bool Tensor__IsDataType_double(const Tensor* p) noexcept override { return p->IsDataType<double>(); }
  bool Tensor__IsDataType_MLFloat16(const Tensor* p) noexcept override { return p->IsDataType<MLFloat16>(); }
  bool Tensor__IsDataType_BFloat16(const Tensor* p) noexcept override { return p->IsDataType<BFloat16>(); }
  bool Tensor__IsDataTypeString(const Tensor* p) noexcept override { return p->IsDataTypeString(); }

#if !defined(DISABLE_FLOAT8_TYPES)
  bool Tensor__IsDataType_Float8E4M3FN(const Tensor* p) noexcept override { return p->IsDataType<Float8E4M3FN>(); }
  bool Tensor__IsDataType_Float8E4M3FNUZ(const Tensor* p) noexcept override { return p->IsDataType<Float8E4M3FNUZ>(); }
  bool Tensor__IsDataType_Float8E5M2(const Tensor* p) noexcept override { return p->IsDataType<Float8E5M2>(); }
  bool Tensor__IsDataType_Float8E5M2FNUZ(const Tensor* p) noexcept override { return p->IsDataType<Float8E5M2FNUZ>(); }
#endif

  const TensorShape& Tensor__Shape(const Tensor* p) override { return p->Shape(); }
  void Tensor__Reshape(Tensor* p, const TensorShape& new_shape) override { return p->Reshape(new_shape); }
  void Tensor__SetByteOffset(Tensor* p, ptrdiff_t byte_offset) override { p->SetByteOffset(byte_offset); }
  ptrdiff_t Tensor__ByteOffset(const Tensor* p) override { return p->ByteOffset(); }
  size_t Tensor__SizeInBytes(const Tensor* p) override { return p->SizeInBytes(); }
  const OrtMemoryInfo& Tensor__Location(const Tensor* p) override { return p->Location(); }
  int32_t Tensor__GetElementType(const Tensor* p) override { return p->GetElementType(); }
  MLDataType Tensor__DataType(const Tensor* p) override { return p->DataType(); }
#ifdef ENABLE_STRIDED_TENSORS
  gsl::span<const int64_t> Tensor__Strides(const Tensor* p) override { return p->Strides(); }
  bool Tensor__IsContiguous(const Tensor* p) override { return p->IsContiguous(); }
  void Tensor__SetShapeAndStrides(Tensor* p, const TensorShape& new_shape,
                                  gsl::span<const int64_t> new_strides) override {
    return p->SetShapeAndStrides(new_shape, new_strides);
  }
#endif

  // SparseTensor(wrapped)
#if !defined(DISABLE_SPARSE_TENSORS)
  const TensorShape& SparseTensor__DenseShape(const SparseTensor* p) override { return p->DenseShape(); }
  Status SparseTensor__Copy(const SparseTensor* p, const DataTransferManager& dtm, SparseTensor& dst) override { return p->Copy(dtm, dst); }
#endif

  void* Allocator__AllocateBufferWithOptions(IAllocator& allocator, size_t size, bool use_reserve, Stream* stream, WaitNotificationFn wait_fn) override { return AllocateBufferWithOptions(allocator, size, use_reserve, stream, wait_fn); }

  // TensorSeq(wrapped)
  MLDataType TensorSeq__DataType(const TensorSeq* p) noexcept override { return p->DataType(); }
  void TensorSeq__SetType(TensorSeq* p, MLDataType data_type) override { p->SetType(data_type); }
  size_t TensorSeq__Size(const TensorSeq* p) noexcept override { return p->Size(); }
  const Tensor& TensorSeq__Get(const TensorSeq* p, size_t i) override { return p->Get(i); }
  const OrtValue& TensorSeq__GetAt(const TensorSeq* p, size_t i) override { return p->GetAt(i); }
  void TensorSeq__Add(TensorSeq* p, const OrtValue& tensor) override { p->Add(tensor); }
  void TensorSeq__Add(TensorSeq* p, OrtValue&& tensor) override { p->Add(std::move(tensor)); }
  void TensorSeq__Add(TensorSeq* p, Tensor&& tensor) override { p->Add(std::move(tensor)); }
  void TensorSeq__Reserve(TensorSeq* p, size_t capacity) override { p->Reserve(capacity); }

#if defined(ENABLE_TRAINING) && defined(ORT_USE_NCCL)
  training::DistributedRunContext& GetDistributedRunContextInstance() override { return training::DistributedRunContext::GetInstance(); }
#endif

#if defined(USE_CUDA) || defined(USE_ROCM)
  PhiloxGenerator& PhiloxGenerator__Default() override { return PhiloxGenerator::Default(); }
#endif

#ifdef ENABLE_TRAINING_TORCH_INTEROP
  void contrib__PythonOpBase__Init(contrib::PythonOpBase* p, const OpKernelInfo& info) override { p->PythonOpBase::Init(info); }
  void contrib__PythonOpBase__Clear(contrib::PythonOpBase* p) override { p->PythonOpBase::Clear(); }
  void contrib__PythonOpBase__SetOutputs(const contrib::PythonOpBase* p, OpKernelContext* context, void* diff_ctx, std::vector<OrtValue>& returned_args) override {
    return p->PythonOpBase::SetOutputs(context, diff_ctx, returned_args);
  }
  void contrib__PythonOpBase__RunForward(const contrib::PythonOpBase* p, OpKernelContext* context, void** diff_ctx, std::vector<OrtValue>& returned_ortvalues) override {
    return p->PythonOpBase::RunForward(context, diff_ctx, returned_ortvalues);
  }

  void contrib__PythonOpGradBase__Init(contrib::PythonOpGradBase* p, const OpKernelInfo& info) override { return p->PythonOpGradBase::Init(info); }
  void contrib__PythonOpGradBase__RunBackward(const contrib::PythonOpGradBase* p, OpKernelContext* context, std::vector<OrtValue>& returned_ortvalues) override {
    return p->PythonOpGradBase::RunBackward(context, returned_ortvalues);
  }
  void contrib__PythonOpGradBase__SetOutputs(const contrib::PythonOpGradBase* p, OpKernelContext* context, std::vector<OrtValue>& returned_args) override { p->PythonOpGradBase::SetOutputs(context, returned_args); }

  language_interop_ops::torch::RefCountTracker& GetRefCountTrackerInstance() override { return language_interop_ops::torch::RefCountTracker::GetInstance(); }
  void RefCountTracker__DumpDetails(const language_interop_ops::torch::RefCountTracker* p, const std::string& phase_name) override {
    return p->language_interop_ops::torch::RefCountTracker::DumpDetails(phase_name);
  }
#endif

#if defined(USE_CANN)
  RandomGenerator& RandomGenerator__Default() override { return RandomGenerator::Default(); }

  std::unique_ptr<Model> cann__CreateModel(const GraphViewer& graph_viewer, const logging::Logger& logger) {
    std::unordered_map<std::string, int> domain_to_version_map;
    domain_to_version_map[kOnnxDomain] = graph_viewer.DomainToVersionMap().at(kOnnxDomain);

    return std::make_unique<Model>(graph_viewer.Name(), true, ModelMetaData(), PathString(),
#if !defined(ORT_MINIMAL_BUILD)
                                   IOnnxRuntimeOpSchemaRegistryList({graph_viewer.GetSchemaRegistry()}), domain_to_version_map,
#else
                                   IOnnxRuntimeOpSchemaRegistryList(), domain_to_version_map,
#endif  // ORT_MINIMAL_BUILD
                                   std::vector<ONNX_NAMESPACE::FunctionProto>(), logger);
  }
#endif

  void MurmurHash3__x86_128(const void* key, int len, uint32_t seed, void* out) override {
    MurmurHash3::x86_128(key, len, seed, out);
  }

#ifdef _WIN32
  std::string ToUTF8String(const std::wstring& s) override { return onnxruntime::ToUTF8String(s); }
  std::wstring ToWideString(const std::string& s) override { return onnxruntime::ToWideString(s); }
#endif

  ProviderHostCPU& GetProviderHostCPU() override { return onnxruntime::GetProviderHostCPU(); }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  Status LoadDynamicLibrary(onnxruntime::PathString library_name) override { return LoadDynamicLibraryFromProvider(library_name); };
#endif
} provider_host_;
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
struct ProviderSharedLibrary {
  void Ensure() {
    if (handle_)
      return;

    auto full_path = Env::Default().GetRuntimePath() +
                     PathString(LIBRARY_PREFIX ORT_TSTR("onnxruntime_providers_shared") LIBRARY_EXTENSION);
    ORT_THROW_IF_ERROR(Env::Default().LoadDynamicLibrary(full_path, true /*shared_globals on unix*/, &handle_));

    void (*PProvider_SetHost)(void*);
    ORT_THROW_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_, "Provider_SetHost", (void**)&PProvider_SetHost));

    PProvider_SetHost(&provider_host_);
  }

  void Unload() {
    if (handle_) {
      auto status = Env::Default().UnloadDynamicLibrary(handle_);
      if (!status.IsOK()) {
        LOGS_DEFAULT(ERROR) << status.ErrorMessage();
      }
      handle_ = nullptr;
    }
  }

  ProviderSharedLibrary() = default;
  ~ProviderSharedLibrary() {
    // assert(!handle_); // We should already be unloaded at this point (disabled until Python shuts down deterministically)
  }

 private:
  void* handle_{};

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProviderSharedLibrary);
};

static ProviderSharedLibrary s_library_shared;

bool InitProvidersSharedLibrary() try {
  s_library_shared.Ensure();
  return true;
} catch (const std::exception&) {
  return false;
}

struct ProviderLibrary {
  ProviderLibrary(const ORTCHAR_T* filename, bool unload = true) : filename_{filename}, unload_{unload} {}
  ~ProviderLibrary() {
    // assert(!handle_); // We should already be unloaded at this point (disabled until Python shuts down deterministically)
  }

  Provider& Get() {
    std::lock_guard<std::mutex> lock{mutex_};
    try {
      if (!provider_) {
        s_library_shared.Ensure();

        auto full_path = Env::Default().GetRuntimePath() + filename_;
        ORT_THROW_IF_ERROR(Env::Default().LoadDynamicLibrary(full_path, false, &handle_));

        Provider* (*PGetProvider)();
        ORT_THROW_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_, "GetProvider", (void**)&PGetProvider));

        provider_ = PGetProvider();
        provider_->Initialize();
      }
      return *provider_;
    } catch (const std::exception&) {
      Unload();  // If anything fails we unload the library and rethrow
      throw;
    }
  }

  void Unload() {
    // This will crash in the Mac unit test due to the ProviderLibrary global variable being destroyed before Unload() is called
    // Something has a global 'Environment or OrtEnv' variable that is being destroyed after other global variables have already been destroyed
    // std::lock_guard<std::mutex> lock{mutex_};

    if (handle_) {
      if (provider_)
        provider_->Shutdown();

      if (unload_) {
        auto status = Env::Default().UnloadDynamicLibrary(handle_);
        if (!status.IsOK()) {
          LOGS_DEFAULT(ERROR) << status.ErrorMessage();
        }
      }

      handle_ = nullptr;
      provider_ = nullptr;
    }
  }

 private:
  std::mutex mutex_;
  const ORTCHAR_T* filename_;
  bool unload_;
  Provider* provider_{};
  void* handle_{};

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProviderLibrary);
};

static ProviderLibrary s_library_cuda(LIBRARY_PREFIX ORT_TSTR("onnxruntime_providers_cuda") LIBRARY_EXTENSION
#ifndef _WIN32
                                      ,
                                      false /* unload - On Linux if we unload the cuda shared provider we crash */
#endif
);
// This lib is only for unittest.
static ProviderLibrary s_library_cuda_test(LIBRARY_PREFIX ORT_TSTR("onnxruntime_providers_cuda_ut") LIBRARY_EXTENSION
#ifndef _WIN32
                                           ,
                                           false /* unload - On Linux if we unload the cuda shared provider we crash */
#endif
);
static ProviderLibrary s_library_cann(LIBRARY_PREFIX ORT_TSTR("onnxruntime_providers_cann") LIBRARY_EXTENSION
#ifndef _WIN32
                                      ,
                                      false /* unload - On Linux if we unload the cann shared provider we crash */
#endif
);
static ProviderLibrary s_library_rocm(LIBRARY_PREFIX ORT_TSTR("onnxruntime_providers_rocm") LIBRARY_EXTENSION
#ifndef _WIN32
                                      ,
                                      false /* unload - On Linux if we unload the rocm shared provider we crash */
#endif
);
static ProviderLibrary s_library_dnnl(LIBRARY_PREFIX ORT_TSTR("onnxruntime_providers_dnnl") LIBRARY_EXTENSION);
static ProviderLibrary s_library_openvino(LIBRARY_PREFIX ORT_TSTR("onnxruntime_providers_openvino") LIBRARY_EXTENSION);
static ProviderLibrary s_library_tensorrt(LIBRARY_PREFIX ORT_TSTR("onnxruntime_providers_tensorrt") LIBRARY_EXTENSION
#ifndef _WIN32
                                          ,
                                          /*
                                           * unload - On CentOS if we unload the TensorRT shared provider we crash.
                                           *
                                           * The reason is TensorRT EP holds a thread local data which won't be destroyed
                                           * until thread exits which happens after TRT EP destruction. Upon thread exits,
                                           * the destructor of thread local data will be called but the address of destructor
                                           * is invalid since the destructor is defined in TRT EP which is already been
                                           * removed from the address space. Therefore, we will hit a segmentation fault.
                                           * So, here we won't unload the TensorRT shared provider and leave the library around.
                                           * This way the OS/CRT/etc doesn't ever clean up until the process exits.
                                           *
                                           * Interestingly, TensorRT shared library won't crash on Ubuntu and Windows when being unloaded.
                                           * The destructor of thread local data can be successfully called upon thread exits.
                                           * One thing worth attention is, on Unix, the function to unload a library is allowed to do nothing.
                                           * Please see here: https://pubs.opengroup.org/onlinepubs/007904975/functions/dlclose.html
                                           *
                                           */
                                          false
#endif
);
static ProviderLibrary s_library_migraphx(LIBRARY_PREFIX ORT_TSTR("onnxruntime_providers_migraphx") LIBRARY_EXTENSION);

void UnloadSharedProviders() {
  s_library_dnnl.Unload();
  s_library_openvino.Unload();
  s_library_tensorrt.Unload();
  s_library_cuda.Unload();
  s_library_cuda_test.Unload();
  s_library_cann.Unload();
  s_library_rocm.Unload();
  s_library_shared.Unload();
  s_library_migraphx.Unload();
}

// Used by test code
std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(const char* name) {
  if (auto* info = onnxruntime::TryGetProviderInfo_CUDA())
    return info->CreateCUDAPinnedAllocator(name);

  return nullptr;
}

std::unique_ptr<IAllocator> CreateROCMPinnedAllocator(const char* name) {
  if (auto* info = onnxruntime::TryGetProviderInfo_ROCM())
    return info->CreateROCMPinnedAllocator(name);

  return nullptr;
}

// Adapter to convert the legacy OrtCUDAProviderOptions to the latest OrtCUDAProviderOptionsV2
OrtCUDAProviderOptionsV2 OrtCUDAProviderOptionsToOrtCUDAProviderOptionsV2(const OrtCUDAProviderOptions* legacy_cuda_options) {
  OrtCUDAProviderOptionsV2 cuda_options_converted{};

  cuda_options_converted.device_id = legacy_cuda_options->device_id;
  cuda_options_converted.cudnn_conv_algo_search = legacy_cuda_options->cudnn_conv_algo_search;
  cuda_options_converted.gpu_mem_limit = legacy_cuda_options->gpu_mem_limit;
  cuda_options_converted.arena_extend_strategy = static_cast<onnxruntime::ArenaExtendStrategy>(legacy_cuda_options->arena_extend_strategy);
  cuda_options_converted.do_copy_in_default_stream = legacy_cuda_options->do_copy_in_default_stream;
  cuda_options_converted.has_user_compute_stream = legacy_cuda_options->has_user_compute_stream;
  cuda_options_converted.user_compute_stream = legacy_cuda_options->user_compute_stream;
  cuda_options_converted.default_memory_arena_cfg = legacy_cuda_options->default_memory_arena_cfg;
  // Use default value as this field is not available in OrtCUDAProviderOptions
  cuda_options_converted.cudnn_conv_use_max_workspace = 1;
  cuda_options_converted.enable_cuda_graph = 0;
  cuda_options_converted.prefer_nhwc = 0;
  cuda_options_converted.cudnn_conv1d_pad_to_nc1d = 0;
  cuda_options_converted.enable_skip_layer_norm_strict_mode = 0;
  cuda_options_converted.use_ep_level_unified_stream = 0;

  return cuda_options_converted;
}

std::shared_ptr<IExecutionProviderFactory> CudaProviderFactoryCreator::Create(const OrtCUDAProviderOptions* provider_options) {
  OrtCUDAProviderOptionsV2 cuda_options_converted = onnxruntime::OrtCUDAProviderOptionsToOrtCUDAProviderOptionsV2(provider_options);
  return s_library_cuda.Get().CreateExecutionProviderFactory(&cuda_options_converted);
}

std::shared_ptr<IExecutionProviderFactory> CudaProviderFactoryCreator::Create(const OrtCUDAProviderOptionsV2* provider_options) {
  return s_library_cuda.Get().CreateExecutionProviderFactory(provider_options);
}

std::shared_ptr<IExecutionProviderFactory> RocmProviderFactoryCreator::Create(const OrtROCMProviderOptions* provider_options) {
  return s_library_rocm.Get().CreateExecutionProviderFactory(provider_options);
}

std::shared_ptr<IExecutionProviderFactory>
CannProviderFactoryCreator::Create(const OrtCANNProviderOptions* provider_options) {
  return s_library_cann.Get().CreateExecutionProviderFactory(provider_options);
}

std::shared_ptr<IExecutionProviderFactory> DnnlProviderFactoryCreator::Create(int use_arena) {
  return s_library_dnnl.Get().CreateExecutionProviderFactory(use_arena);
}

std::shared_ptr<IExecutionProviderFactory> TensorrtProviderFactoryCreator::Create(int device_id) {
  return s_library_tensorrt.Get().CreateExecutionProviderFactory(device_id);
}

std::shared_ptr<IExecutionProviderFactory> MIGraphXProviderFactoryCreator::Create(int device_id) {
  return s_library_migraphx.Get().CreateExecutionProviderFactory(device_id);
}

// Adapter to convert the legacy OrtTensorRTProviderOptions to the latest OrtTensorRTProviderOptionsV2
OrtTensorRTProviderOptionsV2 OrtTensorRTProviderOptionsToOrtTensorRTProviderOptionsV2(const OrtTensorRTProviderOptions* legacy_trt_options) {
  OrtTensorRTProviderOptionsV2 trt_options_converted;

  trt_options_converted.device_id = legacy_trt_options->device_id;
  trt_options_converted.has_user_compute_stream = legacy_trt_options->has_user_compute_stream;
  trt_options_converted.user_compute_stream = legacy_trt_options->user_compute_stream;
  trt_options_converted.trt_max_partition_iterations = legacy_trt_options->trt_max_partition_iterations;
  trt_options_converted.trt_min_subgraph_size = legacy_trt_options->trt_min_subgraph_size;
  trt_options_converted.trt_max_workspace_size = legacy_trt_options->trt_max_workspace_size;
  trt_options_converted.trt_fp16_enable = legacy_trt_options->trt_fp16_enable;
  trt_options_converted.trt_int8_enable = legacy_trt_options->trt_int8_enable;
  trt_options_converted.trt_int8_calibration_table_name = legacy_trt_options->trt_int8_calibration_table_name;
  trt_options_converted.trt_int8_use_native_calibration_table = legacy_trt_options->trt_int8_use_native_calibration_table;
  trt_options_converted.trt_dla_enable = legacy_trt_options->trt_dla_enable;
  trt_options_converted.trt_dla_core = legacy_trt_options->trt_dla_core;
  trt_options_converted.trt_dump_subgraphs = legacy_trt_options->trt_dump_subgraphs;
  trt_options_converted.trt_engine_cache_enable = legacy_trt_options->trt_engine_cache_enable;
  trt_options_converted.trt_engine_cache_path = legacy_trt_options->trt_engine_cache_path;
  trt_options_converted.trt_engine_decryption_enable = legacy_trt_options->trt_engine_decryption_enable;
  trt_options_converted.trt_engine_decryption_lib_path = legacy_trt_options->trt_engine_decryption_lib_path;
  trt_options_converted.trt_force_sequential_engine_build = legacy_trt_options->trt_force_sequential_engine_build;
  // Add new provider option below
  // Use default value as this field is not available in OrtTensorRTProviderOptions
  trt_options_converted.trt_timing_cache_enable = 0;
  trt_options_converted.trt_force_timing_cache = 0;
  trt_options_converted.trt_detailed_build_log = 0;
  trt_options_converted.trt_context_memory_sharing_enable = 0;
  trt_options_converted.trt_layer_norm_fp32_fallback = 0;
  trt_options_converted.trt_build_heuristics_enable = 0;
  trt_options_converted.trt_sparsity_enable = 0;
  trt_options_converted.trt_builder_optimization_level = 3;
  trt_options_converted.trt_auxiliary_streams = -1;
  trt_options_converted.trt_tactic_sources = "";
  trt_options_converted.trt_extra_plugin_lib_paths = "";
  trt_options_converted.trt_profile_min_shapes = "";
  trt_options_converted.trt_profile_max_shapes = "";
  trt_options_converted.trt_profile_opt_shapes = "";
  trt_options_converted.trt_cuda_graph_enable = 0;

  return trt_options_converted;
}

std::shared_ptr<IExecutionProviderFactory> TensorrtProviderFactoryCreator::Create(const OrtTensorRTProviderOptions* provider_options) {
  OrtTensorRTProviderOptionsV2 trt_options_converted = onnxruntime::OrtTensorRTProviderOptionsToOrtTensorRTProviderOptionsV2(provider_options);
  return s_library_tensorrt.Get().CreateExecutionProviderFactory(&trt_options_converted);
}

std::shared_ptr<IExecutionProviderFactory> TensorrtProviderFactoryCreator::Create(const OrtTensorRTProviderOptionsV2* provider_options) {
  return s_library_tensorrt.Get().CreateExecutionProviderFactory(provider_options);
}

std::shared_ptr<IExecutionProviderFactory> MIGraphXProviderFactoryCreator::Create(const OrtMIGraphXProviderOptions* provider_options) {
  return s_library_migraphx.Get().CreateExecutionProviderFactory(provider_options);
}

// Adapter to convert the legacy OrtOpenVINOProviderOptions to ProviderOptions
ProviderOptions OrtOpenVINOProviderOptionsToOrtOpenVINOProviderOptionsV2(const OrtOpenVINOProviderOptions* legacy_ov_options) {
  ProviderOptions ov_options_converted_map;
  if (legacy_ov_options->device_type != nullptr)
    ov_options_converted_map["device_type"] = legacy_ov_options->device_type;

  ov_options_converted_map["enable_npu_fast_compile"] = legacy_ov_options->enable_npu_fast_compile;

  if (legacy_ov_options->device_id != nullptr)
    ov_options_converted_map["device_id"] = legacy_ov_options->device_id;

  ov_options_converted_map["num_of_threads"] = std::to_string(legacy_ov_options->num_of_threads);

  if (legacy_ov_options->cache_dir != nullptr)
    ov_options_converted_map["cache_dir"] = legacy_ov_options->cache_dir;

  std::stringstream context_string;

  if (legacy_ov_options->context != nullptr)
    context_string << legacy_ov_options->context;
  ov_options_converted_map["context"] = context_string.str();

  ov_options_converted_map["enable_opencl_throttling"] = legacy_ov_options->enable_opencl_throttling;
  ov_options_converted_map["enable_dynamic_shapes"] = legacy_ov_options->enable_dynamic_shapes;

  // Add new provider option below
  ov_options_converted_map["num_streams"] = "1";
  return ov_options_converted_map;
}

std::shared_ptr<IExecutionProviderFactory> OpenVINOProviderFactoryCreator::Create(const OrtOpenVINOProviderOptions* provider_options) {
  ProviderOptions ov_options_converted_map = onnxruntime::OrtOpenVINOProviderOptionsToOrtOpenVINOProviderOptionsV2(provider_options);
  return s_library_openvino.Get().CreateExecutionProviderFactory(&ov_options_converted_map);
}

std::shared_ptr<IExecutionProviderFactory> OpenVINOProviderFactoryCreator::Create(const ProviderOptions* provider_options_map) {
  // std::cout << provider_options_map.at("num_streams") << std::endl;
  return s_library_openvino.Get().CreateExecutionProviderFactory(provider_options_map);
}

std::shared_ptr<IExecutionProviderFactory> DnnlProviderFactoryCreator::Create(const OrtDnnlProviderOptions* dnnl_options) {
  return s_library_dnnl.Get().CreateExecutionProviderFactory(dnnl_options);
}

ProviderInfo_OpenVINO* GetProviderInfo_OpenVINO() {
  return reinterpret_cast<ProviderInfo_OpenVINO*>(s_library_openvino.Get().GetInfo());
}

ProviderInfo_TensorRT* TryGetProviderInfo_TensorRT() try {
  return reinterpret_cast<ProviderInfo_TensorRT*>(s_library_tensorrt.Get().GetInfo());
} catch (const std::exception& exception) {
  LOGS_DEFAULT(ERROR) << exception.what();
  return nullptr;
}

ProviderInfo_TensorRT& GetProviderInfo_TensorRT() {
  if (auto* info = TryGetProviderInfo_TensorRT())
    return *info;

  ORT_THROW("TensorRT Provider not available, can't get interface for it");
}

ProviderInfo_CUDA* TryGetProviderInfo_CUDA() try {
  return reinterpret_cast<ProviderInfo_CUDA*>(s_library_cuda.Get().GetInfo());
} catch (const std::exception& exception) {
  LOGS_DEFAULT(ERROR) << exception.what();
  return nullptr;
}

ProviderInfo_CUDA& GetProviderInfo_CUDA() {
  if (auto* info = TryGetProviderInfo_CUDA())
    return *info;

  ORT_THROW("CUDA Provider not available, can't get interface for it");
}

ProviderInfo_CUDA* TryGetProviderInfo_CUDA_Test() try {
  return reinterpret_cast<ProviderInfo_CUDA*>(s_library_cuda_test.Get().GetInfo());
} catch (const std::exception& exception) {
  LOGS_DEFAULT(ERROR) << exception.what();
  return nullptr;
}

ProviderInfo_CUDA& GetProviderInfo_CUDA_Test() {
  if (auto* info = TryGetProviderInfo_CUDA_Test())
    return *info;

  ORT_THROW("CUDA Provider not available, can't get interface for it");
}

ProviderInfo_CANN* TryGetProviderInfo_CANN() try {
  return reinterpret_cast<ProviderInfo_CANN*>(s_library_cann.Get().GetInfo());
} catch (const std::exception& exception) {
  LOGS_DEFAULT(ERROR) << exception.what();
  return nullptr;
}

ProviderInfo_CANN& GetProviderInfo_CANN() {
  if (auto* info = TryGetProviderInfo_CANN())
    return *info;

  ORT_THROW("CANN Provider not available, can't get interface for it");
}

ProviderInfo_Dnnl* TryGetProviderInfo_Dnnl() try {
  return reinterpret_cast<ProviderInfo_Dnnl*>(s_library_dnnl.Get().GetInfo());
} catch (const std::exception& exception) {
  LOGS_DEFAULT(ERROR) << exception.what();
  return nullptr;
}

ProviderInfo_Dnnl& GetProviderInfo_Dnnl() {
  if (auto* info = TryGetProviderInfo_Dnnl())
    return *info;

  ORT_THROW("oneDNN Provider not available, can't get interface for it");
}

ProviderInfo_ROCM* TryGetProviderInfo_ROCM() try {
  return reinterpret_cast<ProviderInfo_ROCM*>(s_library_rocm.Get().GetInfo());
} catch (const std::exception& exception) {
  LOGS_DEFAULT(ERROR) << exception.what();
  return nullptr;
}

ProviderInfo_ROCM& GetProviderInfo_ROCM() {
  if (auto* info = TryGetProviderInfo_ROCM())
    return *info;

  ORT_THROW("ROCM Provider not available, can't get interface for it");
}

void CopyGpuToCpu(
    void* dst_ptr,
    const void* src_ptr,
    const size_t size,
    const OrtMemoryInfo& dst_location,
    const OrtMemoryInfo& src_location) {
  if (auto* info = onnxruntime::TryGetProviderInfo_CUDA())
    return info->CopyGpuToCpu(dst_ptr, src_ptr, size, dst_location, src_location);
  if (auto* info = onnxruntime::TryGetProviderInfo_ROCM())
    return info->CopyGpuToCpu(dst_ptr, src_ptr, size, dst_location, src_location);
  ORT_THROW("GPU-to-CPU copy is not implemented.");
}

void cudaMemcpy_HostToDevice(void* dst, const void* src, size_t count) {
  if (auto* info = onnxruntime::TryGetProviderInfo_CUDA())
    return info->cudaMemcpy_HostToDevice(dst, src, count);
  if (auto* info = onnxruntime::TryGetProviderInfo_ROCM())
    return info->rocmMemcpy_HostToDevice(dst, src, count);
  ORT_THROW("cudaMemcpy_HostToDevice is not implemented.");
}

#ifdef ENABLE_NVTX_PROFILE
namespace profile {
void NvtxRangeCreator::BeginImpl() {
  GetProviderInfo_CUDA().NvtxRangeCreator__BeginImpl(this);
}

void NvtxRangeCreator::EndImpl() {
  GetProviderInfo_CUDA().NvtxRangeCreator__EndImpl(this);
}
}  // namespace profile
#endif

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P) && defined(ENABLE_TRAINING)
namespace cuda {
INcclService& INcclService::GetInstance() {
  return GetProviderInfo_CUDA().GetINcclService();
}
}  // namespace cuda
#endif

#if defined(USE_ROCM) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P) && defined(ENABLE_TRAINING)
namespace rocm {
INcclService& INcclService::GetInstance() {
  return GetProviderInfo_ROCM().GetINcclService();
}
}  // namespace rocm
#endif

void UpdateProviderInfo_Tensorrt(OrtTensorRTProviderOptionsV2* provider_options, const ProviderOptions& options) {
  s_library_tensorrt.Get().UpdateProviderOptions(reinterpret_cast<void*>(provider_options), options);
}

ProviderOptions GetProviderInfo_Tensorrt(const OrtTensorRTProviderOptionsV2* provider_options) {
  return s_library_tensorrt.Get().GetProviderOptions(reinterpret_cast<const void*>(provider_options));
}

void UpdateProviderInfo_Cuda(OrtCUDAProviderOptionsV2* provider_options, const ProviderOptions& options) {
  return s_library_cuda.Get().UpdateProviderOptions(reinterpret_cast<void*>(provider_options), options);
}

ProviderOptions GetProviderInfo_Cuda(const OrtCUDAProviderOptionsV2* provider_options) {
  return s_library_cuda.Get().GetProviderOptions(reinterpret_cast<const void*>(provider_options));
}

}  // namespace onnxruntime

void AddTensorRTCustomOpDomainToSessionOption(OrtSessionOptions* options, std::string extra_plugin_lib_paths) {
  auto is_already_in_domains = [&](std::string& domain_name, std::vector<OrtCustomOpDomain*>& domains) {
    for (auto ptr : domains) {
      if (domain_name == ptr->domain_) {
        return true;
      }
    }
    return false;
  };

  std::vector<OrtCustomOpDomain*> custom_op_domains;
  onnxruntime::ProviderInfo_TensorRT& provider_info = onnxruntime::GetProviderInfo_TensorRT();
  provider_info.GetTensorRTCustomOpDomainList(custom_op_domains, extra_plugin_lib_paths);
  for (auto ptr : custom_op_domains) {
    if (!is_already_in_domains(ptr->domain_, options->custom_op_domains_)) {
      options->custom_op_domains_.push_back(ptr);
    } else {
      LOGS_DEFAULT(WARNING) << "The custom op domain name " << ptr->domain_ << " is already in session option.";
    }
  }
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Dnnl, _In_ OrtSessionOptions* options, int use_arena) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::DnnlProviderFactoryCreator::Create(use_arena);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Dnnl: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tensorrt, _In_ OrtSessionOptions* options, int device_id) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::TensorrtProviderFactoryCreator::Create(device_id);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Tensorrt: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);

  std::string extra_plugin_lib_paths = onnxruntime::Env::Default().GetEnvironmentVar("trt_extra_plugin_lib_paths");
  AddTensorRTCustomOpDomainToSessionOption(options, extra_plugin_lib_paths);

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_MIGraphX, _In_ OrtSessionOptions* options, int device_id) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::MIGraphXProviderFactoryCreator::Create(device_id);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_MIGraphX: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_TensorRT, _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptions* tensorrt_options) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::TensorrtProviderFactoryCreator::Create(tensorrt_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "SessionOptionsAppendExecutionProvider_Tensorrt: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);

  AddTensorRTCustomOpDomainToSessionOption(options, "");

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_MIGraphX, _In_ OrtSessionOptions* options, _In_ const OrtMIGraphXProviderOptions* migraphx_options) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::MIGraphXProviderFactoryCreator::Create(migraphx_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "SessionOptionsAppendExecutionProvider_MIGraphX: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_OpenVINO, _In_ OrtSessionOptions* options, _In_ const OrtOpenVINOProviderOptions* provider_options) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::OpenVINOProviderFactoryCreator::Create(provider_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "SessionOptionsAppendExecutionProvider_OpenVINO: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_OpenVINO, _In_ OrtSessionOptions* options,
                    _In_ const char* device_type) {
  OrtOpenVINOProviderOptions provider_options{};
  provider_options.device_type = device_type;
  return OrtApis::SessionOptionsAppendExecutionProvider_OpenVINO(options, &provider_options);
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id) {
  OrtCUDAProviderOptions provider_options{};
  provider_options.device_id = device_id;

  return OrtApis::SessionOptionsAppendExecutionProvider_CUDA(options, &provider_options);
}

ORT_API_STATUS_IMPL(OrtApis::SetCurrentGpuDeviceId, _In_ int device_id) {
  API_IMPL_BEGIN
  if (auto* info = onnxruntime::TryGetProviderInfo_CUDA())
    return info->SetCurrentGpuDeviceId(device_id);
  if (auto* info = onnxruntime::TryGetProviderInfo_ROCM())
    return info->SetCurrentGpuDeviceId(device_id);
  return CreateStatus(ORT_FAIL, "CUDA and/or ROCM execution provider is either not enabled or not available.");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetCurrentGpuDeviceId, _In_ int* device_id) {
  API_IMPL_BEGIN
  if (auto* info = onnxruntime::TryGetProviderInfo_CUDA())
    return info->GetCurrentGpuDeviceId(device_id);
  if (auto* info = onnxruntime::TryGetProviderInfo_ROCM())
    return info->GetCurrentGpuDeviceId(device_id);
  return CreateStatus(ORT_FAIL, "CUDA and/or ROCM execution provider is either not enabled or not available.");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* cuda_options) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::CudaProviderFactoryCreator::Create(cuda_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Cuda: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_ROCM, _In_ OrtSessionOptions* options, int device_id) {
  OrtROCMProviderOptions provider_options{};
  provider_options.device_id = device_id;

  return OrtApis::SessionOptionsAppendExecutionProvider_ROCM(options, &provider_options);
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_ROCM, _In_ OrtSessionOptions* options, _In_ const OrtROCMProviderOptions* rocm_options) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::RocmProviderFactoryCreator::Create(rocm_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Rocm: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_TensorRT_V2, _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::TensorrtProviderFactoryCreator::Create(tensorrt_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_TensorRT: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);

  std::string extra_plugin_lib_paths = (tensorrt_options == nullptr || tensorrt_options->trt_extra_plugin_lib_paths == nullptr) ? "" : tensorrt_options->trt_extra_plugin_lib_paths;
  AddTensorRTCustomOpDomainToSessionOption(options, extra_plugin_lib_paths);

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorRTProviderOptions, _Outptr_ OrtTensorRTProviderOptionsV2** out) {
  API_IMPL_BEGIN
#ifdef USE_TENSORRT
  auto options = std::make_unique<OrtTensorRTProviderOptionsV2>();
  *out = options.release();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(out);
  return CreateStatus(ORT_FAIL, "TensorRT execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UpdateTensorRTProviderOptions,
                    _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  API_IMPL_BEGIN
#ifdef USE_TENSORRT
  onnxruntime::ProviderOptions provider_options_map;
  for (size_t i = 0; i != num_keys; ++i) {
    if (provider_options_keys[i] == nullptr || provider_options_keys[i][0] == '\0' ||
        provider_options_values[i] == nullptr || provider_options_values[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "key/value cannot be empty");
    }

    provider_options_map[provider_options_keys[i]] = provider_options_values[i];
  }

  onnxruntime::UpdateProviderInfo_Tensorrt(tensorrt_options,
                                           reinterpret_cast<const onnxruntime::ProviderOptions&>(provider_options_map));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateStatus(ORT_FAIL, "TensorRT execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

#if defined(USE_TENSORRT) || defined(USE_CUDA) || defined(USE_CANN) || defined(USE_DNNL) || defined(USE_ROCM)
static std::string BuildOptionsString(const onnxruntime::ProviderOptions::iterator& begin,
                                      const onnxruntime::ProviderOptions::iterator& end) {
  std::ostringstream options;
  auto it = begin;
  if (it != end) {
    options << it->first << "=" << it->second;
    ++it;
  }
  for (; it != end; ++it) {
    options << ";" << it->first << "=" << it->second;
  }
  return options.str();
}
#endif

ORT_API_STATUS_IMPL(OrtApis::GetTensorRTProviderOptionsAsString, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  API_IMPL_BEGIN
#ifdef USE_TENSORRT
  onnxruntime::ProviderOptions options = onnxruntime::GetProviderInfo_Tensorrt(tensorrt_options);
  std::string options_str = BuildOptionsString(options.begin(), options.end());
  *ptr = onnxruntime::StrDup(options_str, allocator);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "TensorRT execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UpdateTensorRTProviderOptionsWithValue,
                    _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _In_ const char* key,
                    _In_ void* value) {
  API_IMPL_BEGIN
#ifdef USE_TENSORRT
  // current provider option that has pointer data type (excluding const char*) is 'user_compute_stream'
  if (strcmp(key, "user_compute_stream") == 0) {
    tensorrt_options->has_user_compute_stream = 1;
    tensorrt_options->user_compute_stream = value;
  }
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(value);
  return CreateStatus(ORT_FAIL, "TensorRT execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorRTProviderOptionsByName,
                    _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _In_ const char* key,
                    _Outptr_ void** ptr) {
  API_IMPL_BEGIN
#ifdef USE_TENSORRT
  // current provider option that has pointer data type (excluding const char*) is 'user_compute_stream'
  if (strcmp(key, "user_compute_stream") == 0) {
    *ptr = tensorrt_options->user_compute_stream;
  } else {
    *ptr = nullptr;
  }
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "TensorRT execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseTensorRTProviderOptions, _Frees_ptr_opt_ OrtTensorRTProviderOptionsV2* ptr) {
#ifdef USE_TENSORRT
  if (ptr != nullptr) {
    delete[] ptr->trt_int8_calibration_table_name;
    delete[] ptr->trt_engine_cache_path;
    delete[] ptr->trt_timing_cache_path;
    delete[] ptr->trt_engine_decryption_lib_path;
    delete[] ptr->trt_tactic_sources;
    delete[] ptr->trt_extra_plugin_lib_paths;
    delete[] ptr->trt_profile_min_shapes;
    delete[] ptr->trt_profile_max_shapes;
    delete[] ptr->trt_profile_opt_shapes;
  }

  std::unique_ptr<OrtTensorRTProviderOptionsV2> p(ptr);
#else
  ORT_UNUSED_PARAMETER(ptr);
#endif
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CUDA_V2, _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptionsV2* cuda_options) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::CudaProviderFactoryCreator::Create(cuda_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Cuda: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateCUDAProviderOptions, _Outptr_ OrtCUDAProviderOptionsV2** out) {
  API_IMPL_BEGIN
#ifdef USE_CUDA
  auto options = std::make_unique<OrtCUDAProviderOptionsV2>();
  *out = options.release();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(out);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UpdateCUDAProviderOptions,
                    _Inout_ OrtCUDAProviderOptionsV2* cuda_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  API_IMPL_BEGIN
#ifdef USE_CUDA
  onnxruntime::ProviderOptions provider_options_map;
  for (size_t i = 0; i != num_keys; ++i) {
    if (provider_options_keys[i] == nullptr || provider_options_keys[i][0] == '\0' ||
        provider_options_values[i] == nullptr || provider_options_values[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "key/value cannot be empty");
    }

    provider_options_map[provider_options_keys[i]] = provider_options_values[i];
  }

  onnxruntime::UpdateProviderInfo_Cuda(cuda_options,
                                       reinterpret_cast<const onnxruntime::ProviderOptions&>(provider_options_map));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(cuda_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetCUDAProviderOptionsAsString, _In_ const OrtCUDAProviderOptionsV2* cuda_options, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  API_IMPL_BEGIN
#ifdef USE_CUDA
  onnxruntime::ProviderOptions options = onnxruntime::GetProviderInfo_Cuda(cuda_options);
  std::string options_str = BuildOptionsString(options.begin(), options.end());
  *ptr = onnxruntime::StrDup(options_str, allocator);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(cuda_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UpdateCUDAProviderOptionsWithValue,
                    _Inout_ OrtCUDAProviderOptionsV2* cuda_options,
                    _In_ const char* key,
                    _In_ void* value) {
  API_IMPL_BEGIN
#ifdef USE_CUDA
  if (strcmp(key, "user_compute_stream") == 0) {
    cuda_options->has_user_compute_stream = 1;
    cuda_options->user_compute_stream = value;
  }
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(cuda_options);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(value);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetCUDAProviderOptionsByName,
                    _In_ const OrtCUDAProviderOptionsV2* cuda_options,
                    _In_ const char* key,
                    _Outptr_ void** ptr) {
  API_IMPL_BEGIN
#ifdef USE_CUDA
  if (strcmp(key, "user_compute_stream") == 0) {
    *ptr = cuda_options->user_compute_stream;
  } else {
    *ptr = nullptr;
  }
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(cuda_options);
  ORT_UNUSED_PARAMETER(key);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "CUDA execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseCUDAProviderOptions, _Frees_ptr_opt_ OrtCUDAProviderOptionsV2* ptr) {
#ifdef USE_CUDA
  std::unique_ptr<OrtCUDAProviderOptionsV2> p(ptr);
#else
  ORT_UNUSED_PARAMETER(ptr);
#endif
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CANN,
                    _In_ OrtSessionOptions* options, _In_ const OrtCANNProviderOptions* cann_options) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::CannProviderFactoryCreator::Create(cann_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL, "SessionOptionsAppendExecutionProvider_CANN: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateCANNProviderOptions, _Outptr_ OrtCANNProviderOptions** out) {
  API_IMPL_BEGIN
#ifdef USE_CANN
  auto options = std::make_unique<OrtCANNProviderOptions>();
  options->device_id = 0;
  options->npu_mem_limit = SIZE_MAX;
  options->arena_extend_strategy = static_cast<onnxruntime::ArenaExtendStrategy>(0);
  options->enable_cann_graph = 1;
  options->dump_graphs = 0;
  options->default_memory_arena_cfg = nullptr;
  *out = options.release();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(out);
  return CreateStatus(ORT_FAIL, "CANN execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UpdateCANNProviderOptions,
                    _Inout_ OrtCANNProviderOptions* cann_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  API_IMPL_BEGIN
#ifdef USE_CANN
  onnxruntime::ProviderOptions provider_options_map;
  for (size_t i = 0; i != num_keys; ++i) {
    if (provider_options_keys[i] == nullptr || provider_options_keys[i][0] == '\0' ||
        provider_options_values[i] == nullptr || provider_options_values[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "key/value cannot be empty");
    }

    provider_options_map[provider_options_keys[i]] = provider_options_values[i];
  }

  onnxruntime::s_library_cann.Get().UpdateProviderOptions(reinterpret_cast<void*>(cann_options), provider_options_map);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(cann_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateStatus(ORT_FAIL, "CANN execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetCANNProviderOptionsAsString,
                    _In_ const OrtCANNProviderOptions* cann_options, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  API_IMPL_BEGIN
#ifdef USE_CANN
  onnxruntime::ProviderOptions options =
      onnxruntime::s_library_cann.Get().GetProviderOptions(reinterpret_cast<const void*>(cann_options));
  std::string options_str = BuildOptionsString(options.begin(), options.end());
  *ptr = onnxruntime::StrDup(options_str, allocator);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(cann_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "CANN execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseCANNProviderOptions, _Frees_ptr_opt_ OrtCANNProviderOptions* ptr) {
#ifdef USE_CANN
  std::unique_ptr<OrtCANNProviderOptions> p(ptr);
#else
  ORT_UNUSED_PARAMETER(ptr);
#endif
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_Dnnl,
                    _In_ OrtSessionOptions* options, _In_ const OrtDnnlProviderOptions* dnnl_options) {
  API_IMPL_BEGIN
  auto factory = onnxruntime::DnnlProviderFactoryCreator::Create(dnnl_options);
  if (!factory) {
    return OrtApis::CreateStatus(ORT_FAIL,
                                 "SessionOptionsAppendExecutionProvider_Dnnl: Failed to load shared library");
  }

  options->provider_factories.push_back(factory);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateDnnlProviderOptions, _Outptr_ OrtDnnlProviderOptions** out) {
  API_IMPL_BEGIN
#ifdef USE_DNNL
  auto options = std::make_unique<OrtDnnlProviderOptions>();
  options->use_arena = true;
  options->threadpool_args = nullptr;
  *out = options.release();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(out);
  return CreateStatus(ORT_FAIL, "oneDNN execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UpdateDnnlProviderOptions,
                    _Inout_ OrtDnnlProviderOptions* dnnl_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  API_IMPL_BEGIN
#ifdef USE_DNNL
  onnxruntime::ProviderOptions provider_options_map;
  for (size_t i = 0; i != num_keys; ++i) {
    if (provider_options_keys[i] == nullptr || provider_options_keys[i][0] == '\0' ||
        provider_options_values[i] == nullptr || provider_options_values[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "key/value cannot be empty");
    }

    provider_options_map[provider_options_keys[i]] = provider_options_values[i];
  }

  onnxruntime::s_library_dnnl.Get().UpdateProviderOptions(reinterpret_cast<void*>(dnnl_options), provider_options_map);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(dnnl_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateStatus(ORT_FAIL, "oneDNN execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetDnnlProviderOptionsAsString,
                    _In_ const OrtDnnlProviderOptions* dnnl_options, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  API_IMPL_BEGIN
#ifdef USE_DNNL
  onnxruntime::ProviderOptions options =
      onnxruntime::s_library_dnnl.Get().GetProviderOptions(reinterpret_cast<const void*>(dnnl_options));
  std::string options_str = BuildOptionsString(options.begin(), options.end());
  *ptr = onnxruntime::StrDup(options_str, allocator);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(dnnl_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "oneDNN execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseDnnlProviderOptions, _Frees_ptr_opt_ OrtDnnlProviderOptions* ptr) {
#ifdef USE_DNNL
  std::unique_ptr<OrtDnnlProviderOptions> p(ptr);
#else
  ORT_UNUSED_PARAMETER(ptr);
#endif
}

ORT_API_STATUS_IMPL(OrtApis::CreateROCMProviderOptions, _Outptr_ OrtROCMProviderOptions** out) {
  API_IMPL_BEGIN
#ifdef USE_ROCM
  auto options = std::make_unique<OrtROCMProviderOptions>();
  options->device_id = 0;
  options->miopen_conv_exhaustive_search = 0;
  options->gpu_mem_limit = std::numeric_limits<size_t>::max();
  options->arena_extend_strategy = 0;
  options->do_copy_in_default_stream = 1;
  options->has_user_compute_stream = 0;
  options->user_compute_stream = nullptr;
  options->default_memory_arena_cfg = nullptr;
  options->tunable_op_enable = 0;
  options->tunable_op_tuning_enable = 0;
  options->tunable_op_max_tuning_duration_ms = 0;

  *out = options.release();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(out);
  return CreateStatus(ORT_FAIL, "ROCm execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UpdateROCMProviderOptions,
                    _Inout_ OrtROCMProviderOptions* rocm_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  API_IMPL_BEGIN
#ifdef USE_ROCM
  onnxruntime::ProviderOptions provider_options_map;
  for (size_t i = 0; i != num_keys; ++i) {
    if (provider_options_keys[i] == nullptr || provider_options_keys[i][0] == '\0' ||
        provider_options_values[i] == nullptr || provider_options_values[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "key/value cannot be empty");
    }

    provider_options_map[provider_options_keys[i]] = provider_options_values[i];
  }

  onnxruntime::s_library_rocm.Get().UpdateProviderOptions(rocm_options, provider_options_map);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(rocm_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateStatus(ORT_FAIL, "ROCm execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetROCMProviderOptionsAsString, _In_ const OrtROCMProviderOptions* rocm_options,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr) {
  API_IMPL_BEGIN
#ifdef USE_ROCM
  onnxruntime::ProviderOptions options = onnxruntime::s_library_rocm.Get().GetProviderOptions(rocm_options);
  std::string options_str = BuildOptionsString(options.begin(), options.end());
  *ptr = onnxruntime::StrDup(options_str, allocator);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(rocm_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateStatus(ORT_FAIL, "ROCm execution provider is not enabled in this build.");
#endif
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseROCMProviderOptions, _Frees_ptr_opt_ OrtROCMProviderOptions* ptr) {
#ifdef USE_ROCM
  std::unique_ptr<OrtROCMProviderOptions> p(ptr);
#else
  ORT_UNUSED_PARAMETER(ptr);
#endif
}
